# KITE: Keypoint-Conditioned Policies for Semantic Manipulation
## [Keypoint Training Repo]

*Priya Sundaresan, Suneel Belkhale, Dorsa Sadigh, Jeannette Bohg*

[[Project]](http://tinyurl.com/kite-site)
[[arXiv]](https://arxiv.org/abs/2306.16605)

## Description
* KITE is a framework for semantic manipulation using keypoints as a mechanism for grounding language instructions in a visual scene, and a library of keypoint-conditioned skills for execution.
* This repo provides the code for training an (image, language) --> keypoint model
* See [our simulated semantic grasping demo](https://github.com/priyasundaresan/kite_semantic_grasping.git) for an example of how this model can be used for downstream semantic manipulation

## Getting Started
* Clone this repo:
```
git clone https://github.com/priyasundaresan/kite_keypoint_training.git
```
* Create a conda environment, either via `conda env create -f env.yml` or via the following:
```
conda create -n kite python=3.10
conda activate kite
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install tqdm
pip install ftfy
pip install regex
pip install opencv-python
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib
```

## Training and Inference
* Copy the data from `/iliad/u/priyasun/kite_keypoint_training/data` to your repo
* Run the following to train on the example `semantic_grasping_dset` dataset:
```
python train.py
```
* After training, run the following to visualize predictions
```
python inference.py
```
* This will save keypoint heatmap visualizations to the folder `preds`
