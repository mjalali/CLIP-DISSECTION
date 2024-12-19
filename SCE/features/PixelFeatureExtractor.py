'''
    This code is borrowed from https://github.com/marcojira/fld
    Thanks for their great work
'''

import torch
import torchvision.transforms as transforms
from SCE.features.ImageFeatureExtractor import ImageFeatureExtractor


class MonoPixelFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "pixel"
        super().__init__(save_path, logger)

        self.features_size = 784
        # From https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L44
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485], std=[0.229])
            ]
        )
        self.model = None

        # self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        # self.model.eval()
        # self.model.to("cuda")
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        
        batch_size = img_batch.shape[0]
        # print(self.name)
        # print(img_batch.shape)
        # print(img_batch.view(batch_size, -1).shape)
        
        return img_batch.view(batch_size, -1)

class ColorPixelFeatureExtractor(ImageFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "pixel"
        super().__init__(save_path, logger)
        
        self.features_size = 784*3
        # From https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L44
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                # )
            ]
        )
        self.model = None

        # self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        # self.model.eval()
        # self.model.to("cuda")
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        
        batch_size = img_batch.shape[0]
        # print(self.name)
        # print(img_batch.shape)
        # print(img_batch.view(batch_size, -1).shape)
        
        return img_batch.view(batch_size, -1)
