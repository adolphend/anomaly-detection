# anomaly-detection

## Review

Considered review and reading in the following file

[./awesome_list.md]

## Setup

use either pip or conda to install the following requirements:

- python >= 3.8.1
- catalyst >= 21.7
- pillow >= 8.3
- torchvision >= 0.10.0

for conda, `conda env create -f environment.yml`

## baseline

Multiple researcher have worked on the project as can be seen on the (awesome list)[./awesome_list], from https://arxiv.org/pdf/2005.14140.pdf; they use feature extraction pretrained models; I would like to investigate if finetuning this on the current dataset can improve the model accuracy. 

Since it is an unsupervised/semi supervised problem, I would like to fine-tune using VAE. For this experiment, I would not investigate all the layers only the embedding produced at the end of the layer, I will do the experiment on ResNet18.

## Running the code

```
python main.py --cuda $CUDA --seed $SEED --data_path $DATA_PATH --batchsize $BATCHSIZE --lr $LR --epochs $EPOCHS --logs $LOGS
```

## Results

metal nut, resnet ROCAUC: 0.922
metal nut, resnet.requires_grad=False ROCAUC: 0.849
metal nut, full trained ROCAUC: 0.812

Status: **It did not improve**
