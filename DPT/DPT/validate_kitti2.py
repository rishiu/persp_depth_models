import torch
import glob
import numpy as np
import imageio
import cv2
from natsort import natsorted

import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Compose

from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dpt.models import DPTDepthModel
from validate_kitti import KittiDepthSelectionV2

class KittiDepthSelection(data.Dataset):
    def __init__(self, datapath, transform=None):
        self.__img_list = []

        self.__transform = transform
        print(datapath + "/val_selection_cropped/image/")

        self.__img_list = natsorted(glob.glob(datapath + "/imgs/*.png"))
        
        self.__depth_list = natsorted(glob.glob(datapath + "/gt/*.png"))
        
        """
        [
            f.replace("/image/", "/groundtruth_depth/").replace(
                "_sync_image_", "_sync_groundtruth_depth_"
            )
            for f in self.__img_list
        ]
        """

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

        print(prediction.shape, target_disparity.shape, mask.shape)
        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        print(scale, shift)
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
        path="weights/dpt_large-midas-2f21e586.pt",
        backbone="vitl16_384",
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

    ds = KittiDepthSelection(data_path, transform)
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
                if k == "fname":
                    continue
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
    KITTI_DATA_PATH = "../../depth_data/KITTI_filtered/161_split/"

    validate(KITTI_DATA_PATH)