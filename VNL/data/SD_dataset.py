import cv2
import json
import torch
import os.path
import numpy as np
import scipy.io as sio
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging
from PIL import Image
from natsort import natsorted
from glob import glob


logger = setup_logging(__name__)

class SDDataset():
    def initialize(self, opt):
        print(opt)
        #data_file = opt.data_file
        self.data_dir = opt.dataroot
        self.rgb_paths = natsorted(glob(f"{self.data_dir}/images/*.jpg"))
        self.depth_paths = natsorted(glob(f"{self.data_dir}/depth/*.npz"))

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_fname = self.rgb_paths[idx]
        depth_fname = self.depth_paths[idx]

        rgb_img = np.array(Image.open(rgb_fname)) / 255.
        depth_img = np.load(depth_fname)["depth"]

        rgb_img = np.transpose(rgb_img, (2,0,1)).astype(np.float32)
        depth_img = np.transpose(depth_img, (2,0,1))

        rgb_img = (rgb_img * 2.0) - 1.0
        #depth_img = (depth_img * 2.0) - 1.0

        rgb_img = torch.from_numpy(rgb_img)
        depth_img = torch.from_numpy(depth_img)

        return {"B": depth_img, "A": rgb_img, "B_bins": self.depth_to_bins(depth_img)}

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def name(self):
        return 'SD'