import cv2
import numpy as np
from typing import List


def visualize_mask(image: np.ndarray, mask: np.ndarray, save_path: str, alpha: float = 128 / 255):
    """在 RGB 图上把 mask>0 的区域叠加半透明白色后保存（向量化实现，与旧版 PIL 逐像素实现等价）。"""
    img = image.astype(np.float32)
    m = np.asarray(mask) > 0
    img[m] = img[m] * (1.0 - alpha) + 255.0 * alpha
    cv2.imwrite(save_path, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def visualize_bbox(image: np.ndarray, bbox: List[int], save_path: str):
    """在 RGB 图上画 [x, y, w, h] 框后保存；bbox 无效（跟丢）时保存原图以保持帧序列完整。"""
    if image is None:
        return
    canvas = np.ascontiguousarray(image).copy()
    if bbox is not None and len(bbox) == 4 and bbox[0] >= 0 and bbox[2] > 0 and bbox[3] > 0:
        x, y, w, h = bbox
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
