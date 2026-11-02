"""Cutie._parse_output 的测试（不实例化 Cutie，不需要 GPU）。"""
import types

import numpy as np
import pytest

pytest.importorskip("torchvision")  # VOT 顶部会 import torchvision

from VOT import Cutie  # noqa: E402
from utils.pose_utils import LOST_BBOX  # noqa: E402


def _parse(mask):
    # _parse_output 只用到 self.erosion_size，用 SimpleNamespace 充当 self
    return Cutie._parse_output(types.SimpleNamespace(erosion_size=5), mask)


def test_empty_mask_returns_lost_bbox():
    bbox = _parse(np.zeros((100, 100), dtype=np.uint8))
    assert bbox == LOST_BBOX


def test_bbox_after_erosion_is_inclusive_python_ints():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:80] = 1
    bbox = _parse(mask)
    # 腐蚀 5x5 后行 22..57、列 32..77，宽高含端点：w = 77-32+1 = 46，h = 57-22+1 = 36
    assert bbox == [32, 22, 46, 36]
    assert all(type(v) is int for v in bbox)


def test_threshold_prob_returns_copy_and_boosts_background():
    import torch
    from VOT import Cutie
    prob = torch.tensor([[[0.45, 0.60]], [[0.55, 0.40]]])     # [bg, obj1] × 1×2
    out = Cutie._threshold_prob(prob, 0.1)
    assert out is not prob and out.data_ptr() != prob.data_ptr()
    assert torch.equal(prob, torch.tensor([[[0.45, 0.60]], [[0.55, 0.40]]]))   # 输入未被修改
    assert torch.allclose(out[0], prob[0] + 0.1) and torch.equal(out[1], prob[1])
    assert torch.argmax(prob, dim=0).tolist() == [[1, 0]]
    assert torch.argmax(out, dim=0).tolist() == [[0, 0]]                       # 0.55 < 0.45+0.1 → 判为背景
