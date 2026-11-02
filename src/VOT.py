import os
import sys
import torch
import numpy as np
from typing import List, Dict
import cv2
from torchvision.transforms.functional import to_tensor

from utils import visualize_mask, visualize_bbox
from utils.pose_utils import LOST_BBOX

proj_path = os.path.join(os.path.dirname(__file__), '..')
cutie_path = os.path.abspath(os.path.join(proj_path, "Cutie"))
if cutie_path not in sys.path:
    sys.path.append(cutie_path)


class Tracker_2D:
    def __init__(self):
        pass

    def initialize(
            self, 
            frame: np.ndarray,
            init_info: Dict[str, np.ndarray], # mask, bbox, etc.  
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    ):
        return list(LOST_BBOX)

    def track(
            self, 
            frame: np.ndarray,
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    ):
        return list(LOST_BBOX)


class Cutie(Tracker_2D):
    def __init__(
            self,
            cutie_seg_threshold: float = 0.1,
            erosion_size: int = 5,
    ):
        super().__init__()
        self.cutie_seg_threshold = cutie_seg_threshold
        self.erosion_size = erosion_size

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import open_dict
        from cutie.model.cutie import CUTIE
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.inference.inference_core import InferenceCore

        weights = os.path.join(cutie_path, "weights", "cutie-base-mega.pth")
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Cutie 权重缺失：{weights}；运行 `python Cutie/cutie/utils/download_models.py` 下载")
        # 直接按 Cutie/cutie/config/eval_config.yaml 构建模型，不走 get_default_model()：
        # 后者每次都对权重算 md5、缺少 GUI 用的第二个权重时会联网下载，且 hydra.initialize 不可重入。
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3.2",
                                   config_dir=os.path.join(cutie_path, "cutie", "config"),
                                   job_name="eval_config"):
            cfg = compose(config_name="eval_config")
        with open_dict(cfg):
            cfg["weights"] = weights
        get_dataset_cfg(cfg)
        self.cutie = CUTIE(cfg).cuda().eval()
        self.cutie.load_weights(torch.load(weights, map_location="cuda"))
        self.cutie_processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.cutie_processor.max_internal_size = -1

    def initialize(
            self, 
            init_frame: np.ndarray,
            init_info: Dict[str, np.ndarray], # mask, bbox, etc.  
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    )->List[int]:
        with torch.no_grad():
            init_frame_tensor = to_tensor(init_frame).cuda().float()
            
            init_mask_tensor = torch.from_numpy(np.asarray(init_info['mask']).astype(np.uint8)).cuda()
            
            objects = np.unique(init_info['mask'])
            # background "0" does not count as an object
            objects = objects[objects != 0].astype(int).tolist()
            
            output_prob = self.cutie_processor.step(init_frame_tensor, init_mask_tensor, objects=objects)
            
            # convert output probabilities to an object mask
            mask = self.cutie_processor.output_prob_to_mask(self._threshold_prob(output_prob, self.cutie_seg_threshold))

            mask_np = mask.cpu().numpy()

        bbox_xywh = self._parse_output(mask_np)

        init_frame = init_frame.copy()

        if mask_visualization_path is not None:
            visualize_mask(init_frame, (mask_np > 0).astype(np.uint8), mask_visualization_path)
        if bbox_visualization_path is not None:
            visualize_bbox(init_frame, bbox_xywh, bbox_visualization_path)

        # del init_frame_tensor, init_mask_tensor, output_prob
        torch.cuda.empty_cache()

        return bbox_xywh

    def track_mask(self, frame: np.ndarray) -> np.ndarray:
        """跑一帧 Cutie，返回 H×W uint8 标签图（像素值 = 物体 id，0 为背景）。"""
        with torch.no_grad():
            frame_tensor = to_tensor(frame).cuda().float()
            output_prob = self.cutie_processor.step(frame_tensor)
            mask = self.cutie_processor.output_prob_to_mask(self._threshold_prob(output_prob, self.cutie_seg_threshold))
            mask_np = mask.cpu().numpy().astype(np.uint8)
        torch.cuda.empty_cache()
        return mask_np

    def track(
            self, 
            frame: np.ndarray, 
            mask_visualization_path: str = None,
            bbox_visualization_path: str = None,
    ) -> List[int]:
        mask_np = self.track_mask(frame)
        bbox_xywh = self._parse_output(mask_np)

        frame = frame.copy()

        if mask_visualization_path is not None:
            visualize_mask(frame, (mask_np > 0).astype(np.uint8), mask_visualization_path)
        if bbox_visualization_path is not None:
            visualize_bbox(frame, bbox_xywh, bbox_visualization_path)

        return bbox_xywh

    @staticmethod
    def _threshold_prob(output_prob: torch.Tensor, threshold: float) -> torch.Tensor:
        """返回提升了背景通道的副本：等价于给所有物体通道减去 threshold，使分割更保守。
        必须返回副本——Cutie 的 last_mask 是 output_prob 的视图，原地修改会污染下一帧的记忆读取。"""
        prob = output_prob.clone()
        prob[0] += threshold
        return prob

    def _parse_output(
            self, 
            mask_np: np.ndarray,
    ) -> List[int]:
        # Perform erosion on the mask
        kernel = np.ones((self.erosion_size, self.erosion_size), np.uint8)
        mask_np = cv2.erode(mask_np.astype(np.uint8), kernel, iterations=1)

        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            # 宽高含端点（像素索引 x_min..x_max 共 x_max - x_min + 1 个像素）
            w = x_max - x_min + 1
            h = y_max - y_min + 1
            bbox = [int(x_min), int(y_min), int(w), int(h)]
        else:
            bbox = list(LOST_BBOX)

        return bbox
