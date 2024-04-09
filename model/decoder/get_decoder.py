


def get_decoder(option):
    if option['decoder']=='base':
        from .base_decoder import BaseDecoder
        return BaseDecoder(option)
    elif option['decoder']=='camoformer':
        from .decoder_p import Decoder
        return Decoder(128)
