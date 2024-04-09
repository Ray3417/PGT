import os
import torch
import torch.nn.functional as F
from dataset.dataloader import get_loader
from config import param as option
from utils import set_seed, AvgMeter
from model.model import get_model
from torch.optim import lr_scheduler
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def structure_loss(pred, mask, epsilon=0.001):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    new_gts = (1-epsilon)*mask+epsilon/2
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def train_one_epoch(epoch, model, optimizer,scheduler, train_loader, option):
    model.train()
    loss_record = AvgMeter()

    progress_bar = tqdm(train_loader, desc='Epoch[{:03d}/{:03d}]'.format(epoch, option['epoch']))
    for pack in progress_bar:
        optimizer.zero_grad()
        images, gts = pack['image'].cuda(), pack['gt'].cuda()
        images = F.interpolate(images, size=option['trainsize'], mode='bilinear', align_corners=True)
        gts = F.interpolate(gts, size=option['trainsize'], mode='bilinear', align_corners=True)
        P5, P1 = model(img=images)
        loss = structure_loss(P1, gts) 
        # loss = structure_loss(P5, gts)
        # loss = structure_loss(P1, gts) + structure_loss(P5, gts)
        loss.backward()
        optimizer.step()

        loss_record.update(loss.data, option['batch_size'])
        progress_bar.set_postfix({'loss':f'{loss_record.show():.3f}', 'lr:':f"{optimizer.param_groups[0]['lr']:.2e}"})
    scheduler.step()
    return model, loss_record


if __name__ == "__main__":
    print('[INFO] Experiments saved in: ', option['training_info'])
    set_seed(option)
    train_loader = get_loader(option)
    model = get_model(option)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), option['lr_config']['lr'], betas=option['lr_config']['beta'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=option['lr_config']['decay_epoch'],
                                    gamma=option['lr_config']['decay_rate'])
    os.makedirs(option['log_path']+'tensorboard_log', exist_ok=True)
    writer = SummaryWriter(option['log_path']+'tensorboard_log')

    for epoch in range(1, (option['epoch']+1)):
        model, loss_record = train_one_epoch(epoch, model, optimizer,scheduler, train_loader, option)
        writer.add_scalar('loss', loss_record.show(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        save_path = option['ckpt_save_path']
        os.makedirs(save_path, exist_ok=True)
        if epoch % option['save_epoch'] == 0:
            save_name = save_path + '{:0>2d}_{:.3f}.pth'.format(epoch, loss_record.show())
            torch.save(model.state_dict(), save_name)
