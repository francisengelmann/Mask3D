# Mask3D 🎭

This is the code for the offical Mask3D demo at [mask3d.org](https://www.mask3d.org). Mask3D is a method for 3D semantic instance segmentation. This code base is derived from the original [Mask3d](https://github.com/JonasSchult/Mask3D) code and updated to more recent libraries.

## Installation Instructions

Install `uv` if not already done:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Set environment variables for your specific GPU and CPU:
```sh
export TORCH_CUDA_ARCH_LIST="7.5" # TITAN RTX (adapt for your own GPU)
export MAX_JOBS=24  # adapt to your cpu
```

Clone this repository:
```sh
git clone git@github.com:francisengelmann/Mask3D.git
cd Mask3D
uv venv
source .venv/bin/activate
```

Install required packages:
```sh
uv pip install ninja cython numpy cmake
uv pip install torch torchvision
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.10.0+cu128.html
uv pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

Test if everything worked:
```sh
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
python -c "import torch_scatter; print('torch_scatter OK')"
python -c "import detectron2; print('detectron2 OK')"
```

Compile third party packages:
```sh
cd third_party
git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
cd ..
chmod +x patch_minkowski_cuda12.sh
# Apply patches to Minkowski engine
./patch_minkowski_cuda12.sh
cd third_party/MinkowskiEngine
sudo apt install libopenblas-dev
python setup.py install --force_cuda --blas=openblas

cd ..
git clone https://github.com/ScanNet/ScanNet.git
cd ScanNet/Segmentator
git checkout 3e5726500896748521a6ceb81271b0f5b2c0e7d2
make

cd ../..
cd pointnet2
python setup.py install

cd ../..
uv pip install pytorch-lightning
```
<!-- uv pip install . -->


Install more packages:
```sh
uv pip install loguru hydra-core einops trimesh open3d albumentations dotenv pyviz3d imageio plyfile wandb volumentations click
```

Download the model weights for the ScanNet200 test set:
```sh
bash download_checkpoint.sh
```

## Ongoing development
- `./scripts/scannet200/run_scannet200_benchmark_eval.sh` - fully vibe-coded inference
- `run.py` - more hand-coded version to understand what's going on


# Prior Instructions from the Original Mask3D Repo


## Code structure
We adapt the codebase of [Mix3D](https://github.com/kumuji/mix3d) which provides a highly modularized framework for 3D Semantic Segmentation based on the MinkowskiEngine.

```
├── mix3d
│   ├── main_instance_segmentation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- Mask3D modules
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
```

#### ScanNet / ScanNet200
First, we apply Felzenswalb and Huttenlocher's Graph Based Image Segmentation algorithm to the test scenes using the default parameters.
Please refer to the [original repository](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for details.
Put the resulting segmentations in `./data/raw/scannet_test_segments`.
```
python -m datasets.preprocessing.scannet_preprocessing preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="data/processed/scannet" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=false/true
```

#### S3DIS
The S3DIS dataset contains some smalls bugs which we initially fixed manually. We will soon release a preprocessing script which directly preprocesses the original dataset. For the time being, please follow the instructions [here](https://github.com/JonasSchult/Mask3D/issues/8#issuecomment-1279535948) to fix the dataset manually. Afterwards, call the preprocessing script as follows:

```
python -m datasets.preprocessing.s3dis_preprocessing preprocess \
--data_dir="PATH_TO_Stanford3dDataset_v1.2" \
--save_dir="data/processed/s3dis"
```

#### STPLS3D
```
python -m datasets.preprocessing.stpls3d_preprocessing preprocess \
--data_dir="PATH_TO_STPLS3D" \
--save_dir="data/processed/stpls3d"
```

### Training and testing :train2:
Train Mask3D on the ScanNet dataset:
```bash
python main_instance_segmentation.py
```
Please refer to the [config scripts](https://github.com/JonasSchult/Mask3D/tree/main/scripts) (for example [here](https://github.com/JonasSchult/Mask3D/blob/main/scripts/scannet/scannet_val.sh#L15)) for detailed instructions how to reproduce our results.
In the simplest case the inference command looks as follows:
```bash
python main_instance_segmentation.py \
general.checkpoint='PATH_TO_CHECKPOINT.ckpt' \
general.train_mode=false
```

## Trained checkpoints :floppy_disk:
We provide detailed scores and network configurations with trained checkpoints.

### [S3DIS](http://buildingparser.stanford.edu/dataset.html) (pretrained on ScanNet train+val)
Following PointGroup, HAIS and SoftGroup, we finetune a model pretrained on ScanNet ([config](./scripts/scannet/scannet_pretrain_for_s3dis.sh) and [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/scannet_pretrained.ckpt)).
| Dataset | AP | AP_50 | AP_25 | Config | Checkpoint :floppy_disk: | Scores :chart_with_upwards_trend: | Visualizations :telescope:
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Area 1 | 69.3 | 81.9 | 87.7 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/area1_scannet_pretrained.ckpt) | [scores](./docs/detailed_scores/s3dis/scannet_pretrained/s3dis_area1_scannet_pretrained.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/scannet_pretrained/area_1/)
| Area 2 | 44.0 | 59.5 | 66.5 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/area2_scannet_pretrained.ckpt) | [scores](./docs/detailed_scores/s3dis/scannet_pretrained/s3dis_area2_scannet_pretrained.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/scannet_pretrained/area_2/)
| Area 3 | 73.4 | 83.2 | 88.2 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/area3_scannet_pretrained.ckpt) | [scores](./docs/detailed_scores/s3dis/scannet_pretrained/s3dis_area3_scannet_pretrained.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/scannet_pretrained/area_3/)
| Area 4 | 58.0 | 69.5 | 74.9 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/area4_scannet_pretrained.ckpt) | [scores](./docs/detailed_scores/s3dis/scannet_pretrained/s3dis_area4_scannet_pretrained.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/scannet_pretrained/area_4/)
| Area 5 | 57.8 | 71.9 | 77.2 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/area5_scannet_pretrained.ckpt) | [scores](./docs/detailed_scores/s3dis/scannet_pretrained/s3dis_area5_scannet_pretrained.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/scannet_pretrained/area_5/)
| Area 6 | 68.4 | 79.9 | 85.2 | [config](scripts/s3dis/s3dis_pretrained.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/scannet_pretrained/area6_scannet_pretrained.ckpt) | [scores](./docs/detailed_scores/s3dis/scannet_pretrained/s3dis_area6_scannet_pretrained.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/scannet_pretrained/area_6/)

### [S3DIS](http://buildingparser.stanford.edu/dataset.html) (from scratch)

| Dataset | AP | AP_50 | AP_25 | Config | Checkpoint :floppy_disk: | Scores :chart_with_upwards_trend: | Visualizations :telescope:
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Area 1 | 74.1 | 85.1 | 89.6 | [config](scripts/s3dis/s3dis_from_scratch.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/from_scratch/area1_from_scratch.ckpt) | [scores](./docs/detailed_scores/s3dis/from_scratch/s3dis_area1_from_scratch.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/from_scratch/area_1/)
| Area 2 | 44.9 | 57.1 | 67.9 | [config](scripts/s3dis/s3dis_from_scratch.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/from_scratch/area2_from_scratch.ckpt) | [scores](./docs/detailed_scores/s3dis/from_scratch/s3dis_area2_from_scratch.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/from_scratch/area_2/)
| Area 3 | 74.4 | 84.4 | 88.1 | [config](scripts/s3dis/s3dis_from_scratch.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/from_scratch/area3_from_scratch.ckpt) | [scores](./docs/detailed_scores/s3dis/from_scratch/s3dis_area3_from_scratch.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/from_scratch/area_3/)
| Area 4 | 63.8 | 74.7 | 81.1 | [config](scripts/s3dis/s3dis_from_scratch.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/from_scratch/area4_from_scratch.ckpt) | [scores](./docs/detailed_scores/s3dis/from_scratch/s3dis_area4_from_scratch.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/from_scratch/area_4/)
| Area 5 | 56.6 | 68.4 | 75.2 | [config](scripts/s3dis/s3dis_from_scratch.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/from_scratch/area5_from_scratch.ckpt) | [scores](./docs/detailed_scores/s3dis/from_scratch/s3dis_area5_from_scratch.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/from_scratch/area_5/)
| Area 6 | 73.3 | 83.4 | 87.8 | [config](scripts/s3dis/s3dis_from_scratch.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/s3dis/from_scratch/area6_from_scratch.ckpt) | [scores](./docs/detailed_scores/s3dis/from_scratch/s3dis_area6_from_scratch.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/s3dis/from_scratch/area_6/)

### [ScanNet v2](https://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d?metric=ap)

| Dataset | AP | AP_50 | AP_25 | Config | Checkpoint :floppy_disk: | Scores :chart_with_upwards_trend: | Visualizations :telescope:
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ScanNet val  | 55.2 | 73.7 | 83.5 | [config](scripts/scannet/scannet_val.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_val.ckpt) | [scores](./docs/detailed_scores/scannet_val.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/scannet/val/)
| ScanNet test | 56.6 | 78.0 | 87.0 | [config](scripts/scannet/scannet_benchmark.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet/scannet_benchmark.ckpt) | [scores](http://kaldir.vc.in.tum.de/scannet_benchmark/result_details?id=1081) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/scannet/test/)

### [ScanNet 200](https://kaldir.vc.in.tum.de/scannet_benchmark/scannet200_semantic_instance_3d)

| Dataset | AP | AP_50 | AP_25 | Config | Checkpoint :floppy_disk: | Scores :chart_with_upwards_trend: | Visualizations :telescope:
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ScanNet200 val | 27.4 | 37.0 | 42.3 | [config](scripts/scannet200/scannet200_val.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt) | [scores](./docs/detailed_scores/scannet200_val.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/scannet200/val/)
| ScanNet200 test | 27.8 | 38.8 | 44.5 | [config](scripts/scannet200/scannet200_benchmark.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt) | [scores](https://kaldir.vc.in.tum.de/scannet_benchmark/result_details?id=1242) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/scannet200/test/)

### [STPLS3D](https://www.stpls3d.com/)

| Dataset | AP | AP_50 | AP_25 | Config | Checkpoint :floppy_disk: | Scores :chart_with_upwards_trend: | Visualizations :telescope:
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| STPLS3D val | 57.3 | 74.3 | 81.6 | [config](scripts/stpls3d/stpls3d_val.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/stpls3d/stpls3d_val.ckpt) | [scores](./docs/detailed_scores/stpls3d.txt) | [visualizations](https://omnomnom.vision.rwth-aachen.de/data/mask3d/visualizations/stpls3d/)
| STPLS3D test | 63.4 | 79.2 | 85.6 | [config](scripts/stpls3d/stpls3d_benchmark.sh) | [checkpoint](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/stpls3d/stpls3d_benchmark.zip) | [scores](https://codalab.lisn.upsaclay.fr/competitions/4646#results) | visualizations

## BibTeX :pray:
```
@article{Schult23ICRA,
  title     = {{Mask3D: Mask Transformer for 3D Semantic Instance Segmentation}},
  author    = {Schult, Jonas and Engelmann, Francis and Hermans, Alexander and Litany, Or and Tang, Siyu and Leibe, Bastian},
  booktitle = {{International Conference on Robotics and Automation (ICRA)}},
  year      = {2023}
}
```
