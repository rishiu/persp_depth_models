import torch
import glob
import numpy as np
import imageio
import cv2
from natsort import natsorted
import os

import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Compose

from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt.models import DPTDepthModel

class KittiDepthSelectionV2(data.Dataset):
    def __init__(self, datapath, gtpath, transform=None):
        self.__img_list = []

        self.__transform = transform
        print(datapath + "/val_selection_cropped/image/")
        self.datapath = datapath

        img_names = []
        with open("/usr/project/depth_models/depth_data/eigen_test_files_with_gt.txt", "r") as in_file:
            for line in in_file:
                img_names.append(line.split(" ")[0])
        self.__img_list = img_names#natsorted(glob.glob(datapath + "/imgs/*.png"))
        
        gt_names = []
        pop_idx = []
        for idx, fname in enumerate(img_names):
            fname = fname.replace("/","-")
            file_dir = fname.split(".")[0]
            updir = file_dir.split('-')[0]
            file_dir = file_dir.replace(updir+'-', '')
            splitted_file = file_dir.split("-")
            real_dir = splitted_file[0]
            num = splitted_file[-1]

            gt_depth_path = os.path.join(gtpath, 'train', real_dir, 'proj_depth/groundtruth/image_02', num +'.png')
            if not os.path.isfile(gt_depth_path):
                gt_depth_path = os.path.join(gtpath, 'val', real_dir, 'proj_depth/groundtruth/image_02', num +'.png')
            if not os.path.isfile(gt_depth_path):
                pop_idx.append(idx)
                continue
            gt_names.append(gt_depth_path)
        
        new_img_list = []
        for i in range(len(self.__img_list)):
            if i not in pop_idx:
                new_img_list.append(self.__img_list[i])
        self.__img_list = new_img_list
        self.__depth_list = gt_names

        #self.__depth_list = natsorted(glob.glob(datapath+"/gt/*.png"))

        # self.__depth_list = [
        #     f.replace("/image/", "/groundtruth_depth/").replace(
        #         "_sync_image_", "_sync_groundtruth_depth_"
        #     )
        #     for f in self.__img_list
        # ]

        self.__length = len(self.__img_list)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image = np.array(imageio.imread(self.datapath+self.__img_list[index], pilmode="RGB"))
        image = image / 255

        # height, width, _ = image.shape
        # top = height - 352
        # left = (width - 1216) // 2
        # image = image[top : top + 352, left : left + 1216,:]

        # depth and mask
        depth_png = np.array(imageio.imread(self.__depth_list[index]), dtype=int)

        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255

        depth = depth_png.astype(np.float32) / 256.0

        mask = depth_png != 0

        # sample
        sample = {}
        sample["image"] = image
        sample["depth"] = depth
        sample["mask"] = mask
        sample["fname"] = self.__img_list[index]

        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample

class KittiDepthSelection(data.Dataset):
    def __init__(self, datapath, transform=None):
        self.__img_list = []

        self.__transform = transform
        print(datapath + "/val_selection_cropped/image/")

        self.__img_list = glob.glob(datapath + "/val_selection_cropped/image/*.png")
        
        self.__depth_list = [
            f.replace("/image/", "/groundtruth_depth/").replace(
                "_sync_image_", "_sync_groundtruth_depth_"
            )
            for f in self.__img_list
        ]

        self.__length = len(self.__img_list)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        # image
        image = np.array(imageio.imread(self.__img_list[index], pilmode="RGB"))
        image = image / 255

        # depth and mask
        depth_png = np.array(imageio.imread(self.__depth_list[index]), dtype=int)

        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert np.max(depth_png) > 255

        depth = depth_png.astype(np.float32) / 256.0

        mask = depth_png != 0

        # sample
        sample = {}
        sample["image"] = image
        sample["depth"] = depth
        sample["mask"] = mask

        # transforms
        if self.__transform is not None:
            sample = self.__transform(sample)

        return sample

class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=10):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

    def compute_scale_and_shift(self, prediction, target, mask):
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

    def __call__(self, prediction, target, mask):
        # transform predicted disparity to aligned depth
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        print(scale.shape, shift.shape)
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned

        # bad pixel
        err = torch.zeros_like(prediciton_depth, dtype=torch.float)

        err[mask == 1] = torch.max(
            prediciton_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediciton_depth[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self.__threshold).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p)


def validate(data_path):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    #model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")

    model = DPTDepthModel(
        path="weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    model.to(device)
    model.eval()
    
    transform = Compose(
            [
                Resize(
                    384, 384,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.5]*3, std=[0.5]*3),
                PrepareForNet(),
            ]
        )

    ds = KittiDepthSelectionV2(data_path, transform)
    dl = data.DataLoader(
        ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    )

    # validate
    metric = BadPixelMetric(depth_cap=80)

    loss_sum = 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            print(f"processing: {i + 1} / {len(ds)}")

            # to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # run model
            prediction = model.forward(batch["image"])

            # resize prediction to match target
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=batch["mask"].shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            prediction = prediction.squeeze(1)

            loss = metric(prediction, batch["depth"], batch["mask"])
            loss_sum += loss

    print(f"bad pixel: {loss_sum / len(ds)}")


if __name__ == "__main__":
    KITTI_DATA_PATH = "../../depth_data/KITTI_filtered/"

    validate(KITTI_DATA_PATH)