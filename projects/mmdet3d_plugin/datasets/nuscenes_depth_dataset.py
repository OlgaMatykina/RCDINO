# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import numpy as np
from mmdet.datasets import DATASETS
from .nuscenes_dataset import CustomNuScenesDataset
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from mmcv.parallel import DataContainer as DC
import random
import math
from tqdm import tqdm
import torch.nn.functional as F


@DATASETS.register_module()
class CustomNuScenesDepthDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def evaluate(self,
                results,
                metric='epe:0-80',
                logger=None,
                jsonfile_prefix=None,
                result_names=['pts_bbox'],
                show=False,
                out_dir=None,
                pipeline=None):
        """Evaluation of model performance based on predictions and ground truth.

        Args:
            results (list[dict]): Testing results of the dataset, where each
                dictionary contains 'preds' (predictions) and 'gt' (ground truth).
            metric (str | list[str], optional): Metric to be evaluated. Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of JSON files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            result_names (list[str], optional): Names of the results. Default: ['pts_bbox'].
            show (bool, optional): Whether to visualize results. Default: False.
            out_dir (str, optional): Path to save visualization results. Default: None.
            pipeline (list[dict], optional): Raw data loading for showing. Default: None.

        Returns:
            dict: Results of each evaluation metric.
        """

        errors, errors_50, errors_70 = [], [], []
        rmses, rmses_50, rmses_70 = [], [], []
        
        # Iterate through results list
        for result in tqdm(results):
            preds = result['preds']
            gt = result['gt']

            # resize gt
            gt = gt.unsqueeze(0)
            gt = F.interpolate(gt, size=(16, 44), mode='bilinear', align_corners=False)
            gt = gt.squeeze(0).squeeze(0)

            # convert preds from softmax to matrix
            preds = torch.argmax(preds, dim=0)  # (16, 44)
            # Преобразуем индексы в значения глубины (2 м + индекс * 0.5 м)
            preds = 2.0 + preds * 0.5  # (16, 44) 

            # Reshape to 1D for error calculations
            preds, gt = (arr.reshape(-1).cpu().numpy() for arr in (preds, gt))

            # Define masks based on ground truth values
            mask1 = (gt > 0) & (gt <= 80)
            mask2 = (gt > 0) & (gt <= 50)
            mask3 = (gt > 0) & (gt <= 70)
            
            # Calculate the difference between predictions and ground truth
            diff = preds - gt
            
            # Calculate errors and RMS values
            diff80, rmse80 = self.get_error(diff, mask1)
            diff50, rmse50 = self.get_error(diff, mask2)
            diff70, rmse70 = self.get_error(diff, mask3)
            
            # Collect results
            errors.append(diff80)
            errors_50.append(diff50)
            errors_70.append(diff70)
            rmses.append(rmse80)
            rmses_50.append(rmse50)
            rmses_70.append(rmse70)
        
        # Aggregate results
        result = {
            'epe:0-80': float(np.mean(errors)),
            'rmse:0-80': float(np.mean(rmses)),
            'epe:0-50': float(np.mean(errors_50)),
            'rmse:0-50': float(np.mean(rmses_50)),
            'epe:0-70': float(np.mean(errors_70)),
            'rmse:0-70': float(np.mean(rmses_70)),
            'data_time': 0,
        }
        
        # Log results if logger is provided
        if logger:
            logger.info(', '.join([f'{k}: {v:.5f}' for k, v in result.items()]))
        
        # # Save results to JSON file if jsonfile_prefix is provided
        # if jsonfile_prefix:
        #     import json
        #     with open(f'{jsonfile_prefix}_results.json', 'w') as f:
        #         json.dump(result, f, indent=4)
        
        # # Show visualization if show is True or out_dir is specified
        # if show or out_dir:
        #     self.show(results, out_dir, show=show, pipeline=pipeline)

        return result
    
    def get_error(self, diffs, mask):
        mae  = np.mean(np.abs(diffs[mask]))
        rmse = np.sqrt(np.mean(diffs[mask]**2))
        return mae, rmse
