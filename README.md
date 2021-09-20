# Decoupled R-CNN

- This code is an official implementation of "IoU Regression with H+L-Sampling for Accurate
Detection Confidence" based on the open source object detection toolbox [mmdetection](https://github.com/open-mmlab/mmdetection). 

## Introduction
It is a common paradigm in object detection frameworks that the samples in training
and testing have consistent distributions for the two main tasks: Classification and bounding box
regression. This paradigm is popular in sampling strategy for training an object detector due to
its intuition and practicability. For the task of localization quality estimation, there exist two ways
of sampling: The same sampling with the main tasks and the uniform sampling by manually
augmenting the ground-truth. The first method of sampling is simple but inconsistent for the task
of quality estimation. The second method of uniform sampling contains all IoU level distributions
but is more complex and difficult for training. In this paper, we propose an H+L-Sampling strategy,
selecting the high and low IoU samples simultaneously, to effectively and simply train the branch of
quality estimation. This strategy inherits the effectiveness of consistent sampling and reduces the
training difficulty of uniform sampling. Finally, we introduce accurate detection confidence, which
combines the classification probability and the localization accuracy, as the ranking keyword of NMS.
Extensive experiments show the effectiveness of our method in solving the misalignment between
classification confidence and localization accuracy and improving the detection performance.

## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.6+
- PyTorch 1.3 or higher
- CUDA 9.0 or higher
- GCC 5+
- mmcv

### Install mmdetection

a. Create a conda virtual environment and activate it.
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install Pytorch and torchvision.

```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

c. Install mmcv.

```shell
 pip install mmcv-full==1.1.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

d. Clone the mmdetection repository.

```shell
git clone --branch v.2.4.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
```

e. Install build requirements and then install mmdetection.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Train and Inference
All our model is trained on 4 TITAN X GPUs with a total batch size of 8 (2 images per GPU). The learning rate is initialized as 0.01.

##### Train with a single GPU
```shell
python tools/train.py ${CONFIG_FILE}
```

##### Train with multiple GPUs
```shell
./tools/dist_train.sh ${CONFIG_FILE} 4 [optional arguments]
```

#####  Test with a single GPU

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

#####  Test with multiple GPUs

```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}]
```

- CONFIG_FILE about D2Det is in [configs/decoupled_rcnn](configs/decoupled_rcnn), please refer to [getting_started.md](docs/getting_started.md) for more details.


## Results

We provide some models with different backbones and results of object detection on MS COCO validation.


## Acknowledgement
Many thanks to the open source codes, i.e., [mmdetection](https://github.com/open-mmlab/mmdetection).
