import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import cv2
import random
from tqdm import tqdm
from natsort import natsorted
from glob import glob

import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch.nn.functional as F

sys.path.append("../DPT/DPT/")
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from util.io import write_depth
from validate_kitti import KittiDepthSelectionV2
from validate_nyu import NyuDepthV2

sys.path.append("../vidar")
from vidar.metrics.depth import DepthEvaluationV2

# DataLoaders for Training and Validation set
class DPTData(Dataset):
  """
    The dataset class for weather net training and validation.

    Parameters:
        root_dir_list (list) -- list of dirs for the dataset.
        is_train (bool) -- True for training set.
  """
  def __init__(self, root_dir_list, is_train=True):
    super(DPTData, self).__init__()

    self.is_train = is_train
    self.img_paths = []
    self.depth_paths = []
    for root_dir in root_dir_list:
      self.img_paths += natsorted(glob(f"{root_dir}/img/*.jpg"))
      self.depth_paths += natsorted(glob(f"{root_dir}/new_depth/*.npy"))
    
    # number of images
    self.data_len = len(self.img_paths)
  
  def __len__(self):
    # return 16
    return 1000#self.data_len

  def get_scene_indices(self):
    return self.scene_indices
  
  def __getitem__(self, index):
    random.seed(index)
    inp_path = self.img_paths[index]
    inp_img = Image.open(inp_path)
    filename = inp_path.split('/')[-1][:-4]

    depth_path = self.depth_paths[index]
    #depth_img = Image.open(depth_path)
    depth_img = np.load(depth_path)

    # To numpy
    inp_img = np.array(inp_img, dtype=np.float32)
    depth_img = np.array(depth_img, dtype=np.float32)

    if not np.isfinite(inp_img).all() or not np.isfinite(depth_img).all():
      print("Non finite!")

    #print(depth_img.shape)
    #print(inp_img.shape)

    inp_img *= 1/255.0
    #depth_img /= np.max(depth_img)
    # Crop Code
    h = inp_img.shape[0]
    w = inp_img.shape[1]
    # Crop out top 200
    #inp_img = inp_img[200:h, :, :]
    #lidar_img = lidar_img[200:h, :]
    # Resize to height 384
    #inp_img = cv2.resize(inp_img, (384, 384), interpolation=cv2.INTER_NEAREST)
    #lidar_img = cv2.resize(lidar_img, (1058, 384), interpolation=cv2.INTER_NEAREST)
    # Random Crop for square
    cc_x = random.randint(0, 512-384)
    cc_y = random.randint(0, 512-384)
    inp_img = inp_img[cc_y:cc_y+384, cc_x:cc_x+384,:]
    depth_img = np.expand_dims(depth_img, axis=2)
    depth_img = depth_img[cc_y:cc_y+384, cc_x:cc_x+384,:]

    inp_img = (inp_img - .5) / .5
    
    inp_img = torch.from_numpy(inp_img).permute((2,0,1))
    depth_img = torch.from_numpy(depth_img).permute((2,0,1))

    # Data augmentations: flip x, flip y, rotate by (90, 180, 270), combinations of flipping and rotating
    if self.is_train:
      aug = random.randint(0, 1)
    else:
      aug = 0
    
    if aug==1 and False:
      inp_img = inp_img.flip(2)
      depth_img = depth_img.flip(2)

    mask = torch.ones_like(depth_img)
    mask[depth_img <= 0] = 0
    mask[depth_img > 10] = 0

    # Dict for return
    # If using tanh as the last layer, the range should be [-1, 1]
    sample_dict = {
        'image': inp_img,
        'depth': depth_img,
        'mask': mask,
        'file_name': filename
    }

    return sample_dict

def compute_scale_and_shift(prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )
    parser.add_argument(
        "--metrics",
        nargs='+',
        default=['abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3']
    )
    parser.add_argument(
        "--data_dir",
        required=True
    )
    parser.add_argument(
        "--gt_dir",
        required=False
    )

    args = parser.parse_args()
    return args

def main(args):
    device = torch.device("cuda")
    
    model = DPTDepthModel(
        invert=False,
        path=args.checkpoint,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False
    )

    model.to(device)

    h = 384
    w = 384

    transform = Compose(
        [
            Resize(
                w,
                h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_LINEAR
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet(),
        ]
    )

    model.eval()

    #ds = NyuDepthV2(args.data_dir, "../DPT/DPT/nyu_data/splits.mat", split="test", transform=transform)
    #dl = data.DataLoader(
    #    ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    #)

    #ds = KittiDepthSelectionV2(args.data_dir, args.gt_dir, transform)
    #dl = data.DataLoader(
    #    ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    #)

    #random.seed(42)

    ds = DPTData(["../../gen_data/CURRENT_FINAL_OUT/train/BASE/"], is_train=False)
    dl = data.DataLoader(
       ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    )

    evaluator = DepthEvaluationV2()

    overall_metrics = torch.zeros((1,8))
    with torch.no_grad():
        ss = 0
        ts = 0
        for i, batch in enumerate(tqdm(dl)):
            for k, v in batch.items():
                if k != "file_name" and k != "fname":
                    batch[k] = v.to(device)

            #print(batch["fname"][0])

            #print(torch.max(batch["image"]), torch.min(batch["image"]))

            #print(batch["image"].shape)

            pred = model.forward(batch["image"])

            pred = F.interpolate(
                pred.unsqueeze(1),
                size=batch["mask"].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            pred = pred.squeeze(1)

            #depth_disp = torch.zeros_like(batch["depth"])
            #depth_disp[batch["mask"] == 1] = 1.0 / batch["depth"][batch["mask"]==1]

            #s, t = compute_scale_and_shift(pred, depth_disp, batch["mask"])
            #ss += s
            #ts += t
            #pred = pred.cpu().numpy()

            #pred = np.transpose(pred, (1,2,0))

            #pred *= 256
            #if i % 500 == 0:
        #     pred_np = pred.cpu().numpy()
        #     pred_np = np.transpose(pred_np, (1,2,0))

        #     img_np = batch["image"].cpu().numpy()[0]
        #     img_np = np.transpose(img_np, (1,2,0))
        #     img_np = np.clip((img_np + 1.0) / 2.0, 0.0, 1.0)

        #     mask_np = batch["mask"].cpu().numpy()[0,0]
        #    # mask_np = np.transpose(mask_np, (1,2,0))

        #         # plt.imshow(img_np)
        #         # plt.savefig("./test_out/midas/midas_img_"+str(i)+".jpg", bbox_inches='tight')

        #     gt_np = batch["depth"].cpu().numpy()[0]
        #     gt_np = np.transpose(gt_np, (1,2,0))
    #           
    #                pred_np *= 256
            # fig, ax = plt.subplots(4)
            # iclb = ax[0].imshow(img_np)
            # gclb = ax[1].imshow(np.multiply(mask_np, gt_np[:,:,0]))
            # mclb = ax[2].imshow(mask_np)
            # dclb = ax[3].imshow(pred_np[:,:,0])
            # plt.colorbar(dclb)
            # plt.colorbar(gclb)
            # plt.colorbar(mclb)
            # plt.colorbar(iclb)
            # plt.savefig("./debug_out/ft/ft_"+batch["file_name"][0].replace("/","-"), bbox_inches='tight')
            # plt.close()

            #write_depth("./test_out/ft/"+batch["file_name"][0].replace("/","-"), pred_np, bits=2, absolute_depth=True)

            # _, gt_height, gt_width = batch["depth"].shape
            # top_margin = int(gt_height - 352)
            # left_margin = int((gt_width - 1216) / 2)
            # pred_depth_uncropped = torch.zeros((1, gt_height, gt_width), dtype=torch.float32, device=device)
            # pred_depth_uncropped[:,top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred
            # pred = pred_depth_uncropped

        #     #NYU
        #     #eval_mask = torch.zeros(batch["mask"].shape, device=device)
        #     #eval_mask[:,45:471,41:601] = 1

        #     #KITTI
        #     ##eval_mask = torch.zeros(batch["mask"].shape, device=device)
        #     #eval_mask[:,int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

        #    # print(batch["mask"].shape)

            final_mask = batch["mask"]#torch.logical_and(eval_mask, batch["mask"])

            metrics = evaluator.compute(batch["depth"].unsqueeze(1), pred, use_gt_scale=True, mask=final_mask, idx=i)
            overall_metrics += metrics
    print(ss / len(dl))
    print(ts / len(dl))
    print(('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3'))
    print(overall_metrics / len(dl))


if __name__ == "__main__":
    args = parse_args()
    main(args)
