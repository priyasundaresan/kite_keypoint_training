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
* Go to the `docker` directory:
```
cd /path/to/kite_keypoint_training/docker
```
* Build the Docker image to install all dependencies:
```
./docker_build.py
```
* After this step, run `docker images` to confirm that the image has built. You should see the following:
```
REPOSITORY            TAG       IMAGE ID       CREATED       SIZE
lang-manip-training   latest    bf3a316e74c5   10 minutes ago   4.14GB
```

## Training and Inference
* Go to the `docker` directory and launch a container:
```
cd /path/to/kite_keypoint_training/docker
./docker_run.py
```
* You should now be inside the Docker container. Run the following to train on the example `semantic_grasping_dset` dataset:
```
python train.py
```
* After training, run the following to visualize predictions
```
python analysis.py
```
* This will save heatmap visualizations to the folder `preds`
* Run `Ctrl A+D` to exit the container
