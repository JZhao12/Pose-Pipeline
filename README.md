<div align="center">
<h2>Pose-Pipeline</h2>
<p>A complete pipeline for CAD-based novel-object 6-DoF pose estimation and tracking from an RGB-D camera</p>
</div>

Given a 3D model of an object and an RGB-D sequence, the pipeline segments the object in the first frame, registers its 6-DoF pose, and tracks it through the whole sequence:

```
RealSense RGB-D  ──►  CNOS (SAM 2 + DINOv3)  ──►  first-frame mask
                                                        │
      3D model (mesh.obj)  ──────────────────────────────┤
                                                        ▼
                     FoundationPose register ──► per-frame: Cutie 2D tracking
                                                          + error-state Kalman filter
                                                          + FoundationPose refinement
                                                        ▼
                              results/<name>/{ob_in_cam/*.txt, pose.npy, bbox_2d.txt, visualizations}
```

Tested on Ubuntu, Python 3.10, PyTorch 2.8.0 + CUDA 12.8, RTX 5070 (sm_120). One sequence of ~450 frames at 640×480 takes about 60–80 s end to end.

## Repository layout

| Path | What it is | In this repo? |
|---|---|---|
| `src/` | The pipeline itself (tracking loop, Kalman filter, Cutie wrapper, capture and video tools) | yes |
| `tests/` | CPU unit tests and an opt-in GPU smoke test | yes |
| `FoundationPose/` | NVIDIA FoundationPose with local build fixes (PR #369, C++17, CUDA 12.8). NVIDIA non-commercial license | yes (source only) |
| `Cutie/` | Cutie video object segmentation, with a small threshold patch | yes (source only) |
| `cnosv2/` | CNOS adapted to SAM 2.1 + DINOv3 for first-frame segmentation (`docs/cnosv2_changes_vs_upstream_298d1f3.patch` lists the changes vs upstream) | yes (source only) |
| `cnosv2/sam2_repo/`, `cnosv2/dinov3_repo/` | Unmodified upstream SAM 2 and DINOv3 at pinned commits | **no**, clone them (see below) |
| `*/weights/`, `*/checkpoints/` | Model weights | **no**, download (see below) |
| `offline/`, `results/` | Your recorded sequences and tracking outputs | **no** |

## Installation

<details><summary>Click to expand</summary>

Prerequisites: Linux, an NVIDIA GPU whose driver supports CUDA 12.8, the CUDA toolkit with `nvcc` (the FoundationPose extensions and nvdiffrast are compiled from source), `cmake`, a C++17 compiler, and conda or mamba.

### 1. Clone and fetch the pinned upstream repos

```bash
git clone https://github.com/JZhao12/Pose-Pipeline.git
cd Pose-Pipeline

git clone https://github.com/facebookresearch/sam2.git   cnosv2/sam2_repo
git -C cnosv2/sam2_repo   checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
git clone https://github.com/facebookresearch/dinov3.git cnosv2/dinov3_repo
git -C cnosv2/dinov3_repo checkout 54694f7627fd815f62a5dcc82944ffa6153bbb76
```

### 2. Environment

```bash
mamba create -n pipeline python=3.10 pip -y
mamba activate pipeline
```

### 3. Segmentation stack (CNOS + SAM 2 + DINOv3)

```bash
cd cnosv2
pip install -r requirements.txt            # pins torch==2.8.0 / torchvision==0.23.0 (cu128)
pip install git+https://github.com/facebookresearch/hand_tracking_toolkit.git
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -e "./sam2_repo[notebooks]"
pip install -r dinov3_repo/requirements.txt
cd ..
```

### 4. FoundationPose

```bash
mamba install conda-forge::eigen=3.4.0 conda-forge::pybind11 conda-forge::boost-cpp
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$CONDA_PREFIX/"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd FoundationPose
pip install "setuptools<70"
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.8.0cu128-cp310-cp310-linux_x86_64.whl

# Build the C++/CUDA extensions. TORCH_CUDA_ARCH_LIST is auto-detected from the installed GPU
# when unset; set it explicitly to cross-compile, e.g. 8.9 for RTX 40 series, 8.6 for RTX 30 series.
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 \
bash build_all_conda.sh
cd ..
```

### 5. Cutie, camera driver, tests

```bash
pip install -e Cutie
pip install pyrealsense2==2.53.1.4623     # only needed for src/run_realsense.py
pip install pytest                         # only needed for tests/
```
</details>

## Model weights

<details><summary>Click to expand</summary>

| Component | File(s) | Put under | Source |
|---|---|---|---|
| SAM 2.1 | `sam2.1_hiera_large.pt` | `cnosv2/sam2_repo/checkpoints/` | `cd cnosv2/sam2_repo/checkpoints && ./download_ckpts.sh` |
| DINOv3 | `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L/16 distilled, torch.hub format) | `cnosv2/dinov3_repo/weights/` | Request access at [Meta's DINOv3 download page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/); you receive signed URLs by e-mail, download with `wget` (the Hugging Face `model.safetensors` is a different format and will not load) |
| FoundationPose | `2023-10-28-18-33-37/` (refiner) and `2024-01-11-20-02-45/` (scorer) | `FoundationPose/weights/` | [Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) linked from the FoundationPose README |
| Cutie | `cutie-base-mega.pth` | `Cutie/weights/` | `cd Cutie && python cutie/utils/download_models.py` |

The paths for SAM 2 and DINOv3 are read from `cnosv2/configs/user/default.yaml` (`./sam2_repo`, `./dinov3_repo` relative to `cnosv2/`); change them there if you keep the repos elsewhere.
</details>

## Data layout

Each sequence lives in `offline/<name>/` (the convention is `<object>_<camera>`, e.g. `teabox_D435`):

```
offline/<name>/
├── rgb/000000.png, 000001.png, ...     8-bit RGB
├── depth/000000.png, ...               16-bit depth, raw sensor units
├── masks/000000.png                    first-frame object mask (0/255); only frame 0 is needed
├── mesh/mesh.obj (+ .mtl, textures)    object model, **in meters**
├── cam_K.txt                           3x3 intrinsics of the color camera
└── depth_scale.txt                     meters per depth unit (e.g. 0.001 for D435, 0.00025 for L515)
```

`run_realsense.py` creates everything except `masks/` and `mesh/`. Depth and mesh must both be metric: the loader converts raw depth with `depth_scale`, and FoundationPose compares it against the mesh directly. Frames are ordered by file name, so keep the zero-padded six-digit names that `run_realsense.py` produces.

## Usage

<details><summary>Click to expand</summary>

All commands run from the repository root with the `pipeline` environment active. The tracking scripts open an OpenCV window by default; add `--no_vis` on a headless machine.

### 1. Record a sequence (Intel RealSense)

```bash
python src/run_realsense.py --obj_path <name>
```
Space starts/stops recording, `q` quits. Frames are written to `offline/<name>/rgb` and `depth`, together with `cam_K.txt` and `depth_scale.txt`.

### 2. Provide a 3D model

Put a metric mesh at `offline/<name>/mesh/mesh.obj`. Any route works: BundleSDF (included in FoundationPose), iPhone Object Capture plus Blender clean-up, or an image-to-3D method.

### 3. Segment the first frame with CNOS

```bash
bash cnosv2/src/scripts/render_pipeline.sh <name>          # renders templates of the mesh (once per object)
bash cnosv2/src/scripts/inference_pipeline.sh <name> [000000]
```
This writes `offline/<name>/masks/000000.png`. Any other 0/255 mask of the object in frame 0 works just as well.

### 4. Track

```bash
python src/run_offline.py --testcase <name> \
    --est_refine_iter 10 --track_refine_iter 3 --kf_scale 0.05 [--no_vis]
python src/run_offline_video.py --testcase <name> [--fps 30]   # pose_visualization/ -> pose_vis.mp4
```

Several objects in the **same** sequence can be tracked jointly. Give each object its own `offline/<name>/` with the shared RGB-D frames plus its own `mesh/` and `masks/000000.png`; intrinsics are read from the first one:

```bash
python src/run_offline_multi.py --testcases "<name_a>,<name_b>" [--no_vis]
```
</details>

## Outputs

`results/<name>/`:

| File | Content |
|---|---|
| `ob_in_cam/<frame>.txt` | 4×4 object-to-camera pose per frame, translation in meters, in the mesh's own coordinate frame |
| `pose.npy` | the same poses stacked as `(N, 4, 4)`; frames where tracking failed are `NaN` |
| `bbox_2d.txt` | per-frame 2D box `x y w h` from Cutie; `-1 -1 0 0` when the object was lost |
| `mask_visualization/`, `bbox_visualization/`, `pose_visualization/` | per-frame overlays; `pose_vis.mp4` after `run_offline_video.py` |

In multi-object mode the combined `pose_visualization/` is written only to the first object's folder.

## How it works

- **Frame 0**: FoundationPose registers the pose from the RGB-D frame and the mask; Cutie is initialised with the same mask; the Kalman filter is initialised from the registered pose.
- **Every following frame**: Kalman `predict` → Cutie segments the object and its box centre is back-projected at the predicted depth to correct the filter's x/y → the filter mean seeds FoundationPose's refiner → the refined pose updates the filter. The saved pose is the refiner output; the filter only decides where refinement starts.
- **Filter** (`src/utils/kalman_filter_6d.py`): error-state Kalman filter over `[t, rotation vector, v_t, ω]`. Rotation is composed on SO(3), so there is no Euler wrap-around. Noise parameters are physical constants; `--kf_scale` scales the 6-D measurement noise (0.05 ≈ 3 mm / 0.5°). Translation is extrapolated with a constant-velocity model; rotation is **not** extrapolated (random walk). The latter was decided on real sequences: for near rotationally symmetric objects the rotation about the symmetry axis is unobservable, and extrapolating it roughly tripled rotation jitter.
- **Robustness**: a refinement seed or result with a non-finite entry or a depth outside 0.05–10 m is rejected; the frame is recorded as `NaN`, the filter rolls back one frame, and the run continues.

## Tests

```bash
python -m pytest -q -p no:cacheprovider tests/                         # CPU, ~2 s
PP_GPU_TESTS=1 python -m pytest -q -p no:cacheprovider tests/test_gpu_smoke.py   # runs 6 frames end to end, needs weights + data
```
The GPU smoke test expects `offline/teabox_D435` and `offline/medbox_D435` to exist; it builds a temporary sandbox and never writes to `results/`.

## Known limitations

- No re-detection: once a frame is marked `NaN` the filter keeps retrying from its last good state but does not re-register. Objects that leave the view and come back are not recovered automatically.
- Near rotationally symmetric objects jitter about their symmetry axis. The proper fix is to pass `symmetry_tfs` to FoundationPose; it is not wired through yet.
- FoundationPose's first-frame registration is not bit-wise deterministic across runs; on symmetric objects this can change per-frame rotation statistics by a few frames. Compare runs with a tolerance.
- Multi-object mode shares one RGB-D stream (the first test case) and has only been exercised on synthetic combinations.

## License

The pipeline code (`src/`, `tests/`, `docs/`) is released under the MIT License (see `LICENSE`). Each vendored component keeps its own license, and they are more restrictive than MIT:

| Component | License |
|---|---|
| FoundationPose | NVIDIA Source Code License, **non-commercial** (research and evaluation only) |
| Cutie | MIT |
| CNOS | MIT |
| SAM 2 | Apache 2.0 |
| DINOv3 | DINOv3 License (Meta); weights are gated |

Because FoundationPose is non-commercial, the pipeline as a whole can only be used non-commercially unless you replace that component.

## Acknowledgements

This pipeline builds on
[CNOS](https://github.com/nv-nguyen/cnos),
[FoundationPose](https://github.com/NVlabs/FoundationPose),
[FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus),
[SAM 2](https://github.com/facebookresearch/sam2),
[DINOv3](https://github.com/facebookresearch/dinov3) and
[Cutie](https://github.com/hkchengrex/Cutie).
