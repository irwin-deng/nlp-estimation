Repository based on https://github.com/jfc43/self-training-ensembles

## Requirements
* Python 3.10
* To install requirements: `pip install -r requirements.txt`

## Downloading Datasets
* [Amazon](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/): Download unprocessed.tar.gz and extract to `/dataset/amazon/`. Subdirectories (such as `/apparel`) should be directly under `/dataset/amazon`

## Overview of the Code
* `train_model.py`: train standard models via supervised learning.
* `eval_pipeline.py`: evaluate various methods on all tasks. 

## Running Experiments

### Training

You can run the following scrips to pre-train all models needed for the experiments. 
* `run_all_cifar10_model_training.sh`: train all supervised learning models. 
* `run_all_cifar10_ensemble_training.sh`: train all ensemble models.

### Evaluation

* `run_all_cifar10_evaluation.sh`: evaluate all methods on all tasks. 

### Methods

The CIFAR-10 dataset contains 40,000 train examples and 10,000 test examples. The CIFAR-10-C dataset contains 19 different corruptions of the test dataset. For each category of corruption, we will use the 10,000 examples with a corruption level of 5.

In each of the experiments below, the labeled dataset consisting of the 40,000 CIFAR-10 train examples is split 80-20 into a train and validation set. The unlabeled dataset contains the full 10,000 examples from a a CIFAR-10-C corruption category.

- Baseline: The random-initialization algorithm proposed in [this paper](https://arxiv.org/pdf/2106.15728.pdf)
- Accuracy-weighted: Ensemble models are weighted by accuracy on the labeled validation set
- Similarity-weighted: Ensemble models are weighted by accuracy on the labeled validation set, with more emphasis on examples that are similar to test examples according to L2 distance between BERT CLS vectors

### Results 

Below is a table of errors (accuracy predicted by the ensemble minus the actual accuracy of the model). Full evaluation log is [here](evaluation_log.txt)

|Unlabeled Dataset  |Baseline|Accuracy-weighted|Similarity-weighted|
|-------------------|--------|-----------------|-------------------|
|Brightness         |-60.69% |-61.23%          |-61.39%            |
|Contrast           |-14.59% |-14.59%          |-14.59%            |
|Defocus blur       |-34.76% |-34.73%          |-34.75%            |
|Elastic transform  |-56.78% |-57.09%          |-57.48%            |
|Fog                |-19.44% |-19.41%          |-19.49%            |
|Frost              |-52.44% |-46.33%          |-49.47%            |
|Gaussian blur      |-28.96% |-28.96%          |-28.96%            |
|Gaussian noise     |-36.60% |-36.64%          |-36.61%            |
|Glass blur         |-46.01% |-38.59%          |-38.59%            |
|Impulse noise      |-7.68%  |-7.67%           |-7.67%             |
|JPEG compression   |-65.47% |-64.66%          |-64.54%            |
|Motion blur        |-38.92% |-38.93%          |-38.92%            |
|Pixelate           |-57.49% |-50.12%          |-50.03%            |
|Saturate           |-57.85% |-57.01%          |-57.02%            |
|Shot noise         |-40.07% |-40.11%          |-40.10%            |
|Snow               |-58.48% |-57.45%          |-60.25%            |
|Spatter            |-51.27% |-48.19%          |-48.07%            |
|Speckle noise      |-38.39% |-38.32%          |-38.32%            |
|Zoom blur          |-39.06% |-39.09%          |-39.09%            |
|                   |        |                 |                   |
|Mean Absolute Error|42.37%  |41.01%           |41.33%             |
|Std.  Dev. of MAE  |16.35%  |15.67%           |15.93%             |
