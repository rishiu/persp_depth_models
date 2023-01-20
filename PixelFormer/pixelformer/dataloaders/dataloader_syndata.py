import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random
import cv2
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        root_dir_list = []
        with open(args.data_file, "r") as data_file:
            root_dir_list.append(data_file.readline().strip())
        self.training_samples = DPTData(args, root_dir_list, transform=preprocessing_transforms(mode))
        if args.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
        else:
            self.train_sampler = None

        self.data = DataLoader(self.training_samples, args.batch_size,
                                shuffle=(self.train_sampler is None),
                                num_workers=args.num_threads,
                                pin_memory=True,
                                sampler=self.train_sampler)

class DPTData(Dataset):
    """
    The dataset class for weather net training and validation.

    Parameters:
        root_dir_list (list) -- list of dirs for the dataset.
        is_train (bool) -- True for training set.
    """
    def __init__(self, args, root_dir_list, transform=None):
        super(DPTData, self).__init__()

        self.args = args
        self.img_paths = []
        self.depth_paths = []
        for root_dir in root_dir_list:
            self.img_paths += natsorted(glob(f"{root_dir}/img/*.jpg"))
            self.depth_paths += natsorted(glob(f"{root_dir}/depth/*.npy"))
        
        # number of images
        self.data_len = len(self.img_paths)
        print(self.data_len)
        print(self.img_paths[0])

        self.transform = transform

    def __len__(self):
        # return 16
        return self.data_len

    def get_scene_indices(self):
        return self.scene_indices

    def __getitem__(self, index):
        inp_path = self.img_paths[index]
        inp_img = Image.open(inp_path)
        filename = inp_path.split('/')[-1][:-4]

        depth_path = self.depth_paths[index]
        #depth_img = Image.open(depth_path)
        depth_img = np.load(depth_path).astype(np.float32)

        if self.args.do_random_rotate is True:
            random_angle = (random.random() - 0.5) * 2 * self.args.degree
            image = self.rotate_image(image, random_angle)
            depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

        # To numpy
        inp_img = np.array(inp_img, dtype=np.float32)
        depth_img = np.array(depth_img, dtype=np.float32)

        #fig, ax = plt.subplots(2)
        #im = ax[0].imshow(depth_img)
        #plt.colorbar(im)

        depth_img[depth_img != 0] = 1.0 / depth_img[depth_img != 0]

        # gim = ax[1].imshow(depth_img)
        # plt.colorbar(gim)
        # plt.savefig(f"./depth_gt{index}.jpg")
        # plt.close()

        if not np.isfinite(inp_img).all() or not np.isfinite(depth_img).all():
            print("Non finite!")

        #print(depth_img.shape)
        #print(inp_img.shape)

        inp_img *= 1/255.0
        #depth_img /= np.max(depth_img)
        # Crop Code
        h = inp_img.shape[0]
        w = inp_img.shape[1]

        dh = depth_img.shape[0]
        dw = depth_img.shape[1]
        # Crop out top 200
        #inp_img = inp_img[200:h, :, :]
        #lidar_img = lidar_img[200:h, :]
        # Resize to height 384
        #inp_img = cv2.resize(inp_img, (384, 384), interpolation=cv2.INTER_NEAREST)
        #lidar_img = cv2.resize(lidar_img, (1058, 384), interpolation=cv2.INTER_NEAREST)
        # Random Crop for square
        if h > 384 and w > 384 and dh > 384 and dw > 384:
            cc_x = random.randint(0, 512-384)
            cc_y = random.randint(0, 512-384)
            inp_img = inp_img[cc_y:cc_y+384, cc_x:cc_x+384,:]
            depth_img = np.expand_dims(depth_img, axis=2)
            depth_img = depth_img[cc_y:cc_y+384, cc_x:cc_x+384,:]
        else:
            inp_img = cv2.resize(inp_img, (384,384))
            depth_img = cv2.resize(depth_img, (384,384))
            depth_img = np.expand_dims(depth_img, axis=2)

        #inp_img = torch.from_numpy(inp_img).permute((2,0,1))
        #depth_img = torch.from_numpy(depth_img).permute((2,0,1))

        # Data augmentations: flip x, flip y, rotate by (90, 180, 270), combinations of flipping and rotating
        inp_img, depth_img = self.train_preprocess(inp_img, depth_img)

        # Dict for return
        # If using tanh as the last layer, the range should be [-1, 1]
        sample_dict = {
            'image': inp_img,
            'depth': depth_img,
            'file_name': filename,
            'focal': 1
        }

        if filename.find("vKITTI") < 0:
            depth_img *= 100

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)

        return sample_dict    

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug


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
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}
    
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
