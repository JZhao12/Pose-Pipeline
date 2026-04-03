<div align="center">
<h2>
A Complete Pipeline for CAD-based Novel Object 6D Pose Estimation
</div>

## Installation
<details><summary>Click to expand</summary>

### 1. conda environment
```
conda create -n pipeline python=3.10 pip -y
conda activate pipeline
cd Pose-Pipeline

# segmentation
cd cnosv2
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/hand_tracking_toolkit.git
pip install git+https://github.com/facebookresearch/segment-anything.git

cd sam2_repo
pip install -e ".[notebooks]"
cd ..

cd dinov3_repo
pip install -r requirements.txt
cd ..
cd ..

# pose estimation
conda install conda-forge::eigen=3.4.0
conda install conda-forge::pybind11
conda install -c conda-forge boost-cpp
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/home/xxx/anaconda3/envs/pipeline/"
export LD_LIBRARY_PATH=/home/xxx/anaconda3/envs/pipeline/lib:$LD_LIBRARY_PATH

cd FoundationPose
pip install "setuptools<70"
python -m pip install -r requirements.txt --no-build-isolation
python -m pip install --quiet --no-cache-dir --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --quiet --no-cache-dir kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/pytorch3d-0.7.9%2Bpt2.8.0cu128-cp310-cp310-linux_x86_64.whl

# follow guidance from here: https://github.com/NVlabs/FoundationPose/pull/369
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

# Cutie
cd ..
cd Cutie
pip install -e .

# Others
pip install pyrealsense2==2.53.1.4623
pip install spyder-kernels
```

### 2. model weights
```
# sam2
cd cnosv2/sam2_repo/checkpoints && \
./download_ckpts.sh && \
cd ..

# dinov3
Download from hf and put it under the folder cnosv2/dinov3_repo/weights/

# FoundationPose
Download all network weights and put them under the folder FoundationPose/weights/

# Cutie
cd Cutie
python cutie/utils/download_models.py
```
</details>

##  Usage
<details><summary>Click to expand</summary>

### 1. record data sequence
```
cd Pose-Pipeline
conda activate pipeline
python src/run_realsense.py --obj_path xxx
```

### 2. get 3D model
There are multiple ways to obtain an object 3D model, try any one of them:
+ run BundleSDF in FoundationPose for reconstruction;
+ AR Code Object Capture in iPhone, Blender to finetune;
+ image-to-3d methods like SAM 3D Objects/Wonder3D/Hunyuan3D;

Then put the 3D model under the folder offline/xxx/mesh/

### 3. get segmentation mask
```
cd cnosv2
bash ./src/scripts/render_pipeline.sh xxx
bash ./src/scripts/inference_pipeline.sh xxx
cd ..
```

### 4. run pose estimation
```
python src/run_offline.py --testcase xxx
python src/run_offline_video.py --testcase xxx
```
</details>

## License
Unless specified, this pipeline is released under the MIT license.

## Acknowledgement

The pipeline makes use of multiple existing works that are openly available at:
+ [CNOS](https://github.com/nv-nguyen/cnos)
+ [FoundationPose](https://github.com/NVlabs/FoundationPose)
+ [FoundationPose++](https://github.com/teal024/FoundationPose-plus-plus)
+ [SAM 2](https://github.com/facebookresearch/sam2)
+ [DINOv3](https://github.com/facebookresearch/dinov3)

