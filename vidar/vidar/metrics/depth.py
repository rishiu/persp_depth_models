# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

import torch
import numpy as np
import matplotlib.pyplot as plt

from vidar.metrics.base import BaseEvaluationV2, BaseEvaluation
from vidar.metrics.utils import create_crop_mask, scale_output
from vidar.utils.config import cfg_has
from vidar.utils.data import dict_remove_nones
from vidar.utils.depth import post_process_depth
from vidar.utils.distributed import on_rank_0
from vidar.utils.logging import pcolor
from vidar.utils.types import is_dict

class DepthEvaluationV2(BaseEvaluationV2):
    """
    Detph evaluation metrics

    Parameters
    ----------
    cfg : Config
        Configuration file
    """
    def __init__(self):
        super().__init__(
            name='depth', task='depth',
            metrics=('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3'),
        )

        self.min_depth = 1e-3#cfg.min_depth
        self.max_depth = 80#cfg.max_depth
        self.crop = 'garg'#cfg_has(cfg, 'crop', '')
        self.scale_output = 'none'#cfg_has(cfg, 'scale_output', 'resize')

        self.post_process = False#cfg_has(cfg, 'post_process', False)
        self.scaling = False#cfg_has(cfg, 'median_scaling', False)
        self.valid_threshold = None#cfg.has('valid_threshold', None)

        if self.post_process:
            self.modes += ['pp']
        if self.scaling:
            self.modes += ['gt']
        if self.post_process and self.median_scaling:
            self.modes += ['pp_gt']

    @staticmethod
    def reduce_fn(metrics, seen):
        """Reduce function"""
        valid = seen.view(-1) > 0
        return (metrics[valid] / seen.view(-1, 1)[valid]).mean(0)

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """Populate metrics function"""
        for metric in metrics:
            if metric.startswith(self.name):
                name, suffix = metric.split('|')
                for i, key in enumerate(self.metrics):
                    metrics_dict[f'{prefix}-{name}|{key}_{suffix}'] = \
                        metrics[metric][i].item()

    @on_rank_0
    def print(self, reduced_data, prefixes):
        """Print function"""
        print()
        print(self.horz_line)
        print(self.metr_line.format(*((self.name.upper(),) + self.metrics)))
        for n, metrics in enumerate(reduced_data):
            if sum([self.name in key for key in metrics.keys()]) == 0:
                continue
            print(self.horz_line)
            print(self.wrap(pcolor('*** {:<114}'.format(prefixes[n]), **self.font1)))
            print(self.horz_line)
            for key, metric in sorted(metrics.items()):
                if self.name in key:
                    print(self.wrap(pcolor(self.outp_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

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

    def compute(self, gt, pred, use_gt_scale=True, mask=None, idx=None):
        """
        Compute depth metrics

        Parameters
        ----------
        gt : torch.Tensor
            Ground-truth depth maps [B,1,H,W]
        pred : torch.Tensor
            Predicted depth map [B,1,H,W]
        use_gt_scale : Bool
            Use median-scaling
        mask : torch.Tensor or None
            Mask to remove pixels from evaluation

        Returns
        -------
        metrics : torch.Tensor
            Depth metrics
        """
        # Match predicted depth map to ground-truth resolution
        #pred = scale_output(pred, gt, self.scale_output)
        # Create crop mask if requested
        #crop_mask = create_crop_mask(self.crop, gt)
        # For each batch sample
        metrics = []
        for i, (pred_i, gt_i) in enumerate(zip(pred, gt)):

            # Squeeze GT and PRED
            gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
            mask_i = torch.squeeze(mask[i]) if mask is not None else None

            #pred_i = 1 / (pred_i + 1e-8)
            #gt_i = 1 / (gt_i + 1e-8)

            # gt_i *= 100

            #pred_i[mask_i == 1] = 1.0 / pred_i[mask_i == 1]
            #pred_i = torch.nan_to_num(pred_i, posinf=0.0, neginf=0.0)
            #print(torch.min(pred_i), torch.max(pred_i))
            #print(torch.min(gt_i), torch.max(gt_i))
            #pred_i = pred_i.clamp(self.min_depth, self.max_depth)
            #pred_i[torch.isinf(pred_i)] = self.max_depth

            gt_i = gt_i.clamp(self.min_depth, self.max_depth)

            #print(torch.unique(gt_i))
            #gt_i[torch.isinf(gt_i)] = 0
            #gt_i[torch.isnan(gt_i)] = 0
            disp_cap = 1.0 / self.max_depth
            disp_min = 1.0 / self.min_depth

            if use_gt_scale:
                gt_disp = torch.zeros_like(gt_i)
                gt_disp[mask_i == 1] = 1.0 / gt_i[mask_i == 1]
                gt_disp[gt_disp < disp_cap] = disp_cap
                gt_disp[gt_disp > disp_min] = disp_min

                #pred_sc = pred_i[mask_i == 1.0]

                scale, shift = self.compute_scale_and_shift(pred_i[None,:,:], gt_disp[None,:,:], mask_i[None,:,:])

                pred_new = scale * pred_i + shift#pred_i * torch.median(gt_i) / torch.median(pred_i)
            else:
                pred_new = pred_i
            #disp_cap = 1.0 / self.max_depth
            #disp_min = 1.0 / self.min_depth
            #print(torch.min(pred_new), torch.max(pred_new))
            pred_new[pred_new < disp_cap] = disp_cap
            pred_new[pred_new > disp_min] = disp_min

            #pred_new[pred_new < self.min_depth] = self.min_depth
            #pred_new[pred_new > self.max_depth] = self.max_depth

            pred_depth = 1.0 / pred_new
            #pred_depth = pred_new

            if False and idx % 50 == 0:
                pred_np = pred_i.cpu().numpy()
                pred_new_np = pred_depth.cpu().numpy()

                gt_np = gt_i.cpu().numpy()
                mask_np = mask_i.cpu().numpy()
                
                #print(pred_np.shape, pred_new_np.shape, gt_np.shape, mask_np.shape)
                #gt_np = np.transpose(gt_np, (1,2,0))
    #            pred_np *= 256
                #pred_np = np.transpose(pred_np, (1,2,0))
                fig, ax = plt.subplots(5)
                gclb = ax[0].imshow(np.multiply(mask_np,gt_np))
                psclb = ax[1].imshow(np.multiply(mask_np,pred_new_np))
                dclb = ax[2].imshow(np.multiply(mask_np,pred_np))
                maskclb = ax[3].imshow(mask_np)
                diclb = ax[4].imshow(np.abs(np.multiply(mask_np,gt_np[:,:] - pred_new_np[:,:])))
                plt.colorbar(dclb)
                plt.colorbar(gclb)
                plt.colorbar(psclb)
                plt.colorbar(maskclb)
                plt.colorbar(diclb)
                plt.savefig("./debug_out/midas_"+str(idx)+".jpg", bbox_inches='tight')
                plt.close()
            #pred_depth = pred_new
            # Keep valid pixels (min/max depth and crop)
            #valid = (gt_i > self.min_depth) & (gt_i < self.max_depth)

            # Remove invalid predicted pixels as well
            #valid = valid & (pred_i > 0) & (gt_i > 0)
            # Apply crop mask if requested
            #valid = valid & crop_mask.bool() if crop_mask is not None else valid
            # Apply provided mask if available
            #valid = valid & mask_i.bool() if mask is not None else valid

            # Invalid evaluation
            #if self.valid_threshold is not None and valid.sum() < self.valid_threshold:
            #    return None

            #pred_i = 1 / pred_i

            # Keep only valid pixels
            gt_i, pred_depth = gt_i[mask_i == 1], pred_depth[mask_i == 1]
            # GT median scaling if needed
            
            # Clamp PRED depth values to min/max values

            # Calculate depth metrics
            #print(gt_i)
            thresh = torch.max((gt_i / pred_depth), (pred_depth / gt_i))
            #print(thresh)
            a1 = (thresh < 1.25).float().mean()
            a2 = (thresh < 1.25 ** 2).float().mean()
            a3 = (thresh < 1.25 ** 3).float().mean()

            diff_i = gt_i - pred_depth
            abs_rel = torch.mean(torch.abs(diff_i) / gt_i)
            sq_rel = torch.mean(diff_i ** 2 / gt_i)
            rmse = torch.sqrt(torch.mean(diff_i ** 2))
            rmse_log = torch.sqrt(torch.mean((torch.log(gt_i) - torch.log(pred_depth)) ** 2))

            err = torch.log(pred_depth) - torch.log(gt_i)
            silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

            metrics.append([abs_rel, sq_rel, rmse, rmse_log, silog, a1, a2, a3])

        # Return metrics
        return torch.tensor(metrics, dtype=gt.dtype)

    def evaluate(self, batch, output, flipped_output=None):
        """
        Evaluate predictions

        Parameters
        ----------
        batch : Dict
            Dictionary containing ground-truth information
        output : Dict
            Dictionary containing predictions
        flipped_output : Bool
            Optional flipped output for post-processing

        Returns
        -------
        metrics : Dict
            Dictionary with calculated metrics
        predictions : Dict
            Dictionary with additional predictions
        """
        metrics, predictions = {}, {}
        if self.name not in batch:
            return metrics, predictions
        # For each output item
        for key, val in output.items():
            # If it corresponds to this task
            if key.startswith(self.name) and 'debug' not in key:
                # Loop over every context
                val = val if is_dict(val) else {0: val}
                for ctx in val.keys():
                    # Loop over every scale
                    for i in range(1 if self.only_first else len(val[ctx])):

                        pred = val[ctx][i]
                        gt = batch[self.name][ctx]

                        if self.post_process:
                            pred_flipped = flipped_output[key][ctx][i]
                            pred_pp = post_process_depth(pred, pred_flipped, method='mean')
                        else:
                            pred_pp = None

                        if i > 0:
                            pred = self.interp_nearest(pred, val[ctx][0])
                            if self.post_process:
                                pred_pp = self.interp_nearest(pred_pp, val[ctx][0])

                        if pred.dim() == 4:
                            suffix = '(%s)' % str(ctx) + ('_%d' % i if not self.only_first else '')
                            for mode in self.modes:
                                metrics[f'{key}|{mode}{suffix}'] = \
                                    self.compute(
                                        gt=gt,
                                        pred=pred_pp if 'pp' in mode else pred,
                                        use_gt_scale='gt' in mode,
                                        mask=None,
                                    )
                        elif pred.dim() == 5:
                            for j in range(pred.shape[1]):
                                suffix = '(%s_%d)' % (str(ctx), j) + ('_%d' % i if not self.only_first else '')
                                for mode in self.modes:
                                    metrics[f'{key}|{mode}{suffix}'] = self.compute(
                                        gt=gt[:, j],
                                        pred=pred_pp[:, j] if 'pp' in mode else pred[:, j],
                                        use_gt_scale='gt' in mode,
                                        mask=None,
                                    )

        return dict_remove_nones(metrics), predictions

class DepthEvaluation(BaseEvaluation):
    """
    Detph evaluation metrics

    Parameters
    ----------
    cfg : Config
        Configuration file
    """
    def __init__(self, cfg):
        super().__init__(cfg,
            name='depth', task='depth',
            metrics=('abs_rel', 'sqr_rel', 'rmse', 'rmse_log', 'silog', 'a1', 'a2', 'a3'),
        )

        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth
        self.crop = cfg_has(cfg, 'crop', '')
        self.scale_output = cfg_has(cfg, 'scale_output', 'resize')

        self.post_process = cfg_has(cfg, 'post_process', False)
        self.median_scaling = cfg_has(cfg, 'median_scaling', False)
        self.valid_threshold = cfg.has('valid_threshold', None)

        if self.post_process:
            self.modes += ['pp']
        if self.median_scaling:
            self.modes += ['gt']
        if self.post_process and self.median_scaling:
            self.modes += ['pp_gt']

    @staticmethod
    def reduce_fn(metrics, seen):
        """Reduce function"""
        valid = seen.view(-1) > 0
        return (metrics[valid] / seen.view(-1, 1)[valid]).mean(0)

    def populate_metrics_dict(self, metrics, metrics_dict, prefix):
        """Populate metrics function"""
        for metric in metrics:
            if metric.startswith(self.name):
                name, suffix = metric.split('|')
                for i, key in enumerate(self.metrics):
                    metrics_dict[f'{prefix}-{name}|{key}_{suffix}'] = \
                        metrics[metric][i].item()

    @on_rank_0
    def print(self, reduced_data, prefixes):
        """Print function"""
        print()
        print(self.horz_line)
        print(self.metr_line.format(*((self.name.upper(),) + self.metrics)))
        for n, metrics in enumerate(reduced_data):
            if sum([self.name in key for key in metrics.keys()]) == 0:
                continue
            print(self.horz_line)
            print(self.wrap(pcolor('*** {:<114}'.format(prefixes[n]), **self.font1)))
            print(self.horz_line)
            for key, metric in sorted(metrics.items()):
                if self.name in key:
                    print(self.wrap(pcolor(self.outp_line.format(
                        *((key.upper(),) + tuple(metric.tolist()))), **self.font2)))
        print(self.horz_line)
        print()

    def compute(self, gt, pred, use_gt_scale=True, mask=None):
        """
        Compute depth metrics

        Parameters
        ----------
        gt : torch.Tensor
            Ground-truth depth maps [B,1,H,W]
        pred : torch.Tensor
            Predicted depth map [B,1,H,W]
        use_gt_scale : Bool
            Use median-scaling
        mask : torch.Tensor or None
            Mask to remove pixels from evaluation

        Returns
        -------
        metrics : torch.Tensor
            Depth metrics
        """
        # Match predicted depth map to ground-truth resolution
        pred = scale_output(pred, gt, self.scale_output)
        # Create crop mask if requested
        crop_mask = create_crop_mask(self.crop, gt)
        # For each batch sample
        metrics = []
        for i, (pred_i, gt_i) in enumerate(zip(pred, gt)):

            # Squeeze GT and PRED
            gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
            mask_i = torch.squeeze(mask[i]) if mask is not None else None

            # Keep valid pixels (min/max depth and crop)
            valid = (gt_i > self.min_depth) & (gt_i < self.max_depth)
            # Remove invalid predicted pixels as well
            valid = valid & (pred_i > 0)
            # Apply crop mask if requested
            valid = valid & crop_mask.bool() if crop_mask is not None else valid
            # Apply provided mask if available
            valid = valid & mask_i.bool() if mask is not None else valid

            # Invalid evaluation
            if self.valid_threshold is not None and valid.sum() < self.valid_threshold:
                return None

            # Keep only valid pixels
            gt_i, pred_i = gt_i[valid], pred_i[valid]
            # GT median scaling if needed
            if use_gt_scale:
                pred_i = pred_i * torch.median(gt_i) / torch.median(pred_i)
            # Clamp PRED depth values to min/max values
            pred_i = pred_i.clamp(self.min_depth, self.max_depth)

            # Calculate depth metrics

            thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
            a1 = (thresh < 1.25).float().mean()
            a2 = (thresh < 1.25 ** 2).float().mean()
            a3 = (thresh < 1.25 ** 3).float().mean()

            diff_i = gt_i - pred_i
            abs_rel = torch.mean(torch.abs(diff_i) / gt_i)
            sq_rel = torch.mean(diff_i ** 2 / gt_i)
            rmse = torch.sqrt(torch.mean(diff_i ** 2))
            rmse_log = torch.sqrt(torch.mean((torch.log(gt_i) - torch.log(pred_i)) ** 2))

            err = torch.log(pred_i) - torch.log(gt_i)
            silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

            metrics.append([abs_rel, sq_rel, rmse, rmse_log, silog, a1, a2, a3])

        # Return metrics
        return torch.tensor(metrics, dtype=gt.dtype)

    def evaluate(self, batch, output, flipped_output=None):
        """
        Evaluate predictions

        Parameters
        ----------
        batch : Dict
            Dictionary containing ground-truth information
        output : Dict
            Dictionary containing predictions
        flipped_output : Bool
            Optional flipped output for post-processing

        Returns
        -------
        metrics : Dict
            Dictionary with calculated metrics
        predictions : Dict
            Dictionary with additional predictions
        """
        metrics, predictions = {}, {}
        if self.name not in batch:
            return metrics, predictions
        # For each output item
        for key, val in output.items():
            # If it corresponds to this task
            if key.startswith(self.name) and 'debug' not in key:
                # Loop over every context
                val = val if is_dict(val) else {0: val}
                for ctx in val.keys():
                    # Loop over every scale
                    for i in range(1 if self.only_first else len(val[ctx])):

                        pred = val[ctx][i]
                        gt = batch[self.name][ctx]

                        if self.post_process:
                            pred_flipped = flipped_output[key][ctx][i]
                            pred_pp = post_process_depth(pred, pred_flipped, method='mean')
                        else:
                            pred_pp = None

                        if i > 0:
                            pred = self.interp_nearest(pred, val[ctx][0])
                            if self.post_process:
                                pred_pp = self.interp_nearest(pred_pp, val[ctx][0])

                        if pred.dim() == 4:
                            suffix = '(%s)' % str(ctx) + ('_%d' % i if not self.only_first else '')
                            for mode in self.modes:
                                metrics[f'{key}|{mode}{suffix}'] = \
                                    self.compute(
                                        gt=gt,
                                        pred=pred_pp if 'pp' in mode else pred,
                                        use_gt_scale='gt' in mode,
                                        mask=None,
                                    )
                        elif pred.dim() == 5:
                            for j in range(pred.shape[1]):
                                suffix = '(%s_%d)' % (str(ctx), j) + ('_%d' % i if not self.only_first else '')
                                for mode in self.modes:
                                    metrics[f'{key}|{mode}{suffix}'] = self.compute(
                                        gt=gt[:, j],
                                        pred=pred_pp[:, j] if 'pp' in mode else pred[:, j],
                                        use_gt_scale='gt' in mode,
                                        mask=None,
                                    )

        return dict_remove_nones(metrics), predictions

