import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import cv2
from dataset.augment import randomFlip, randomCrop, randomRotation, colorEnhance, randomPeper


class TrainDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = sorted([image_root + f for f in os.listdir(image_root)])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root)])
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        gt = Image.open(self.gts[index]).convert('L')
        image, gt = randomFlip(image, gt)
        image, gt = randomCrop(image, gt)
        image, gt = randomRotation(image, gt)
        image = colorEnhance(image)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return {'image': image, 'gt': gt}

    def __len__(self):
        return len(self.images)


def get_loader(option):
    dataset = TrainDataset(option['paths']['image_root'], option['paths']['gt_root'], trainsize=option['trainsize'])
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=option['batch_size'],
                                  shuffle=True,
                                  num_workers=option['num_workers'],
                                  pin_memory=True)
    return data_loader


class TestDataset:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root)]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = Image.open(self.images[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        self.index += 1
        return image, HH, WW, name


class EvalDataset(data.Dataset):
    def __init__(self, img_root, label_root):
        self.image_path = sorted([img_root + f for f in os.listdir(img_root)])
        self.label_path = sorted([label_root + f for f in os.listdir(label_root)])

    def __getitem__(self, item):
        pred_path = self.image_path[item]
        mask_path = self.label_path[item]
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
    
