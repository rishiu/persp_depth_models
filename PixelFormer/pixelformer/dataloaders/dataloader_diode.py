import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random
import cv2
from natsort import natsorted
from glob import glob

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])

class NewDataLoaderV2(object):
    def __init__(self, args, mode):
        ds = DIODEData("/usr/project/depth_models/depth_data/DIODE/data/outdoor/", transform=preprocessing_transforms(mode))
        self.data = DataLoader(
           ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
        )

class NewDataLoader(object):
    def __init__(self, args, mode):
        ds = DIODEData("/usr/project/depth_models/depth_data/DIODE/data/outdoor/", transform=preprocessing_transforms(mode))
        self.data = DataLoader(
           ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
        )

class DIODEData(Dataset):
    def __init__(self, root_dir, transform=None):
        super(DIODEData, self).__init__()

        self.img_paths = []
        self.depth_paths = []
        self.mask_paths = []

        for scene in natsorted(os.listdir(root_dir)):
            if os.path.isdir(root_dir+"/"+scene):
                for scan in natsorted(os.listdir(root_dir+"/"+scene)):
                    self.img_paths += natsorted(glob(f"{root_dir}/{scene}/{scan}/*.png"))
                    self.depth_paths += natsorted(glob(f"{root_dir}/{scene}/{scan}/*_depth.npy"))
                    self.mask_paths += natsorted(glob(f"{root_dir}/{scene}/{scan}/*_depth_mask.npy"))

        assert len(self.img_paths) == len(self.depth_paths) == len(self.mask_paths)
        self.data_len = len(self.img_paths)

        self.__transform = transform

    def __len__(self):
        return 2500#self.data_len

    def __getitem__(self, index):
        inp_path = self.img_paths[index]
        inp_img = np.array(Image.open(inp_path), dtype=np.float32)
        inp_img /= 255.


        depth_path = self.depth_paths[index]
        depth_img = np.load(depth_path).astype(np.float32)

        mask_path = self.mask_paths[index]
        mask_img = np.load(mask_path)

        sample = {}

        sample["image"] = inp_img
        sample["depth"] = depth_img
        sample["mask"] = mask_img
        sample["path"] = inp_path
        sample["focal"] = 0
        sample["has_valid_depth"] = True

        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': sample['path']}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
