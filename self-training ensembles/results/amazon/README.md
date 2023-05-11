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
* `run_all_model_training.sh`: train all supervised learning models. 
* `run_all_ensemble_training.sh`: train all ensemble models.

### Evaluation

* `run_all_evaluation.sh`: evaluate all methods on all tasks. 

### Methods

The Amazon dataset contains 10 categories with 2000 reviews each. Each experiment uses a pair of categories, one as the labeled dataset and one as the unlabeled dataset.

The labeled dataset (n=2000) is split 80-20 into a train and validation set. The test dataset contains the full 2000 examples from the unlabeled dataset.

- Baseline: The random-initialization algorithm proposed in [this paper](https://arxiv.org/pdf/2106.15728.pdf)
- Accuracy-weighted: Ensemble models are weighted by accuracy on the labeled validation set
- Similarity-weighted: Ensemble models are weighted by accuracy on the labeled validation set, with more emphasis on examples that are similar to test examples according to L2 distance between BERT CLS vectors

### Results 

Below is a table of errors (accuracy predicted by the ensemble minus the actual accuracy of the model). Full evaluation log is [here](evaluation_log.txt)

|Labeled Dataset                   |Unlabeled Dataset                  |Accuracy-weighted        |Similarity-weighted  |Baseline        |
|----------------------------------|-----------------------------------|-------------------------|---------------------|----------------|
|Apparel                           |Books                              |2.00%                    |2.00%                |2.00%           |
|Apparel                           |DVD                                |-0.20%                   |-34.30%              |-34.30%         |
|Apparel                           |Electronics                        |-31.65%                  |-31.65%              |-31.70%         |
|Apparel                           |Health                             |-33.05%                  |-32.70%              |-32.50%         |
|Apparel                           |Kitchen                            |-34.50%                  |-17.60%              |-34.20%         |
|Apparel                           |Music                              |-34.90%                  |2.10%                |-34.90%         |
|Apparel                           |Sports                             |-33.45%                  |-33.35%              |-33.40%         |
|Apparel                           |Toys                               |-15.60%                  |-16.60%              |-37.30%         |
|Apparel                           |Video                              |-32.20%                  |-32.20%              |-32.20%         |
|Books                             |Apparel                            |-31.10%                  |-31.10%              |-31.10%         |
|Books                             |DVD                                |-29.05%                  |-28.25%              |-28.45%         |
|Books                             |Electronics                        |-19.80%                  |-19.80%              |-19.80%         |
|Books                             |Health                             |-20.50%                  |-20.50%              |-20.50%         |
|Books                             |Kitchen                            |-20.60%                  |-20.60%              |-20.60%         |
|Books                             |Music                              |-26.50%                  |-26.50%              |-26.50%         |
|Books                             |Sports                             |-22.90%                  |-22.90%              |-22.90%         |
|Books                             |Toys                               |-26.40%                  |-26.40%              |-26.40%         |
|Books                             |Video                              |-27.65%                  |-27.35%              |-27.55%         |
|DVD                               |Apparel                            |-14.60%                  |-14.60%              |-14.60%         |
|DVD                               |Books                              |-31.05%                  |-31.00%              |-31.70%         |
|DVD                               |Electronics                        |-17.60%                  |-17.60%              |-17.60%         |
|DVD                               |Health                             |-24.35%                  |-24.15%              |-24.30%         |
|DVD                               |Kitchen                            |-20.55%                  |-20.55%              |-20.50%         |
|DVD                               |Music                              |-9.40%                   |-9.00%               |-9.10%          |
|DVD                               |Sports                             |-22.80%                  |-22.75%              |-22.80%         |
|DVD                               |Toys                               |-16.80%                  |-16.85%              |-17.05%         |
|DVD                               |Video                              |-29.75%                  |-32.40%              |-29.55%         |
|Electronics                       |Apparel                            |-15.80%                  |-15.60%              |-15.95%         |
|Electronics                       |Books                              |10.00%                   |10.00%               |10.05%          |
|Electronics                       |DVD                                |-0.75%                   |-0.85%               |-33.60%         |
|Electronics                       |Health                             |-19.00%                  |-18.90%              |-20.05%         |
|Electronics                       |Kitchen                            |-16.90%                  |-16.40%              |-33.95%         |
|Electronics                       |Music                              |1.90%                    |1.80%                |1.90%           |
|Electronics                       |Sports                             |-17.05%                  |-16.30%              |-16.35%         |
|Electronics                       |Toys                               |-14.10%                  |-14.75%              |-14.10%         |
|Electronics                       |Video                              |-4.80%                   |-4.80%               |-36.60%         |
|Health                            |Apparel                            |-34.00%                  |-33.90%              |-34.20%         |
|Health                            |Books                              |7.10%                    |7.10%                |7.10%           |
|Health                            |DVD                                |-38.20%                  |-38.25%              |-38.20%         |
|Health                            |Electronics                        |-30.40%                  |-30.40%              |-30.45%         |
|Health                            |Kitchen                            |-33.70%                  |-32.95%              |-33.50%         |
|Health                            |Music                              |-33.20%                  |-33.20%              |-33.20%         |
|Health                            |Sports                             |-33.90%                  |-34.10%              |-34.50%         |
|Health                            |Toys                               |-37.45%                  |-37.85%              |-37.50%         |
|Health                            |Video                              |-41.20%                  |-41.20%              |-41.20%         |
|Kitchen                           |Apparel                            |-19.05%                  |-19.75%              |-20.35%         |
|Kitchen                           |Books                              |1.25%                    |1.20%                |31.90%          |
|Kitchen                           |DVD                                |5.75%                    |5.80%                |5.75%           |
|Kitchen                           |Electronics                        |-30.55%                  |-30.05%              |-30.35%         |
|Kitchen                           |Health                             |-35.60%                  |-34.20%              |-37.75%         |
|Kitchen                           |Music                              |2.10%                    |2.10%                |2.10%           |
|Kitchen                           |Sports                             |-33.75%                  |-33.45%              |-34.05%         |
|Kitchen                           |Toys                               |-14.55%                  |-14.95%              |-14.75%         |
|Kitchen                           |Video                              |3.20%                    |3.20%                |3.20%           |
|Music                             |Apparel                            |-19.30%                  |-19.30%              |-19.30%         |
|Music                             |Books                              |-5.00%                   |-5.05%               |-5.05%          |
|Music                             |DVD                                |-26.25%                  |-24.90%              |-25.40%         |
|Music                             |Electronics                        |-2.60%                   |-2.60%               |-2.60%          |
|Music                             |Health                             |-17.00%                  |-17.00%              |-17.00%         |
|Music                             |Kitchen                            |-16.20%                  |-16.20%              |-16.20%         |
|Music                             |Sports                             |-14.70%                  |-14.70%              |-14.65%         |
|Music                             |Toys                               |-21.15%                  |-21.10%              |-21.10%         |
|Music                             |Video                              |-20.70%                  |-20.60%              |-20.70%         |
|Sports                            |Apparel                            |-19.60%                  |-16.75%              |-19.70%         |
|Sports                            |Books                              |13.10%                   |13.10%               |13.05%          |
|Sports                            |DVD                                |2.60%                    |2.60%                |2.95%           |
|Sports                            |Electronics                        |-33.35%                  |-33.35%              |-33.25%         |
|Sports                            |Health                             |-16.20%                  |-13.15%              |-14.65%         |
|Sports                            |Kitchen                            |-24.70%                  |-25.00%              |-34.00%         |
|Sports                            |Music                              |-7.55%                   |-7.70%               |7.55%           |
|Sports                            |Toys                               |-24.90%                  |-30.70%              |-32.00%         |
|Sports                            |Video                              |1.00%                    |1.35%                |1.05%           |
|Toys                              |Apparel                            |-12.45%                  |-12.10%              |-12.60%         |
|Toys                              |Books                              |-25.90%                  |-13.20%              |-13.15%         |
|Toys                              |DVD                                |-23.20%                  |-23.20%              |-20.35%         |
|Toys                              |Electronics                        |2.70%                    |2.60%                |2.65%           |
|Toys                              |Health                             |-6.45%                   |-6.25%               |-6.40%          |
|Toys                              |Kitchen                            |-7.95%                   |-7.90%               |-8.25%          |
|Toys                              |Music                              |-32.35%                  |-4.30%               |-4.30%          |
|Toys                              |Sports                             |-7.20%                   |-6.90%               |-6.20%          |
|Toys                              |Video                              |-23.55%                  |-23.45%              |-17.75%         |
|Video                             |Apparel                            |-8.60%                   |-8.60%               |-8.60%          |
|Video                             |Books                              |-12.90%                  |-12.35%              |-13.00%         |
|Video                             |DVD                                |-19.45%                  |-38.50%              |-20.90%         |
|Video                             |Electronics                        |-10.80%                  |-10.80%              |-10.90%         |
|Video                             |Health                             |-12.95%                  |-12.90%              |-12.90%         |
|Video                             |Kitchen                            |-8.10%                   |-8.10%               |-8.10%          |
|Video                             |Music                              |-2.25%                   |-2.25%               |-2.05%          |
|Video                             |Sports                             |-4.70%                   |-4.70%               |-4.70%          |
|Video                             |Toys                               |-3.20%                   |-3.20%               |-3.20%          |
|                                  |                                   |                         |                     |                |
|                                  |Mean Absolute Error                |18.61%                   |18.16%               |20.14%          |
|                                  |Std.  Dev. of MAE                  |11.32%                   |11.38%               |11.60%          |
