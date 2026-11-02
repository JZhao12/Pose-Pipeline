"""utils.visualization 的测试（纯 CPU）。"""
import cv2
import numpy as np
from PIL import Image

from utils.pose_utils import LOST_BBOX
from utils.visualization import visualize_bbox, visualize_mask


def _reference_visualize_mask(image, mask, save_path):
    """旧版逐像素 PIL 实现，原样保留作为对照基准。"""
    # Ensure mask is in 'RGBA' mode
    mask_rgba = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')

    # Create an alpha mask where the non-zero regions are semi-transparent
    alpha = 128  # Semi-transparent
    # Pillow >= 12.1 中 getdata 已弃用，优先用 get_flattened_data（行为相同）
    get_data = getattr(mask_rgba, "get_flattened_data", mask_rgba.getdata)
    mask_data = get_data()
    new_data = []
    for item in mask_data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            # Background pixel, make it fully transparent
            new_data.append((0, 0, 0, 0))
        else:
            # Mask pixel, set desired transparency
            new_data.append((item[0], item[1], item[2], alpha))
    mask_rgba.putdata(new_data)

    # Convert the original image to 'RGBA'
    image_rgba = Image.fromarray(image.astype(np.uint8)).convert('RGBA')

    # Overlay the mask onto the image
    overlaid_image = Image.alpha_composite(image_rgba, mask_rgba)

    # Convert back to 'RGB' mode if you don't need transparency in the saved image
    overlaid_image = overlaid_image.convert('RGB')

    overlaid_image.save(save_path)


def test_visualize_mask_matches_reference(tmp_path):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    mask = rng.integers(0, 2, size=(48, 64), dtype=np.uint8) * 255

    new_path = str(tmp_path / "new.png")
    ref_path = str(tmp_path / "ref.png")
    visualize_mask(image, mask, new_path)
    _reference_visualize_mask(image, mask, ref_path)

    new_img = cv2.imread(new_path).astype(np.int16)
    ref_img = cv2.imread(ref_path).astype(np.int16)
    assert new_img.shape == ref_img.shape == (48, 64, 3)
    assert np.abs(new_img - ref_img).max() <= 1


def test_visualize_bbox_lost_still_saves_unchanged_image(tmp_path):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    path = str(tmp_path / "bbox.png")

    visualize_bbox(image, list(LOST_BBOX), path)

    saved = cv2.imread(path)
    assert saved is not None
    assert saved.shape == image.shape
    # 输入是 RGB，保存为 PNG 后读回（BGR）再转 RGB，应与输入逐像素一致
    assert np.array_equal(cv2.cvtColor(saved, cv2.COLOR_BGR2RGB), image)
