## Directory structure
- archive: Contains previous executions of the experiment
- bootstrap.py: Contains code for bayesian bootstrapping experiments
- common.py: Contains objects shared between files
- similarity.py: Contains functions for determining similarity between examples
- utils.py: Contains code used in multiple files

## Description of experiments
We construct confidence intervals to estimate the accuracy of a topic classification model on an unlabeled dataset. We use the TE_WikiCate model and the yahoo (40,000 examples) and agnews (7,600 examples) datasets from https://github.com/CogComp/ZeroShotWiki. We construct 95% confidence intervals by sampling (with repetition) a batch of labeled examples of size equal to the labeled dataset, and then calculating the accuracy of the classifier on those labeled examples. This process is repeated 10 thousand times, and then the 2.5 and 97.5 percentile values are chosen as the bounds of the bootstrapped confidence interval. We also compute the true accuracy on the unlabeled dataset as a reference. We compare confidence intervals constructed using weighted sampling (by sampling more similar examples more often), or unweighted simple random sampling from the labeled dataset.

When performing weighted sampling, the probability of sampling a specific labeled example is proportional to the inverse distance between a randomly sampled unlabeled example and that labeled examples' vector encoding. This encoding is based on the output of [this model](https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all), which seems to do well with topic classification tasks.

## Experiments ran and results
10,000 confidence intervals were generate for each experiment:
1) Unlabeled and labeled datasets are from the same source
    1) Unlabeled dataset: 400 (1%) of examples from yahoo; labeled dataset: remaining 39,600 examples from yahoo. 10 thousand 95% confidence intervals were constructed (using different subsets of yahoo as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        * confidence intervals from weighted sampling (data for each trial is [here](archive/2023-03-06/results_L-yahoo-U-yahoo.csv)):
            * average width: 0.0065
            * proportion containing the true accuracy: 0.1331 (0.2864 too low, 0.5805 too high)
        * confidence intervals from unweighted sampling:
            * average width: 0.0067
            * proportion containing the true accuracy: 0.1736 (0.4074 too low, 0.419 too high)
    2) Unlabeled dataset: 76 (1%) of examples from agnews; labeled dataset: remaining 7,524 examples from agnews. 10 thousand 95% confidence intervals were constructed (using different subsets of agnews as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        * confidence intervals from weighted sampling (data for each trial is [here](archive/2023-03-06/results_L-agnews-U-agnews.csv)):
            * average width: 0.0178
            * proportion containing the true accuracy: 0.1747 (0.4185 too low, 0.4068 too high)
        * confidence intervals from unweighted sampling:
            * average width: 0.0178
            * proportion containing the true accuracy: 0.1184 (0.4793 too low, 0.4023 too high)
2) Unlabeled and labeled datasets are from different sources
    1) Unlabeled dataset: 400 (1%) of examples from yahoo; labeled dataset: all 7,600 examples from agnews. 10 thousand 95% confidence intervals were constructed (using different subsets of yahoo as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        * confidence intervals from weighted sampling (data for each trial is [here](archive/2023-03-06/results_L-agnews-U-yahoo.csv)):
            * average width: 0.0177
            * proportion containing the true accuracy: 0.0022 (0.9978 too low, 0.0 too high)
        * confidence intervals from unweighted sampling:
            * average width: 0.0177
            * proportion containing the true accuracy: 0.0019 (0.9981 too low, 0.0 too high)
    2) Unlabeled dataset: 76 (1%) of examples from agnews; labeled dataset: all 40,000 examples from yahoo. 10 thousand 95% confidence intervals were constructed (using different subsets of agnews as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        * confidence intervals from weighted sampling (data for each trial is [here](archive/2023-03-06/results_L-yahoo-U-agnews.csv)):
            * average width: 0.0065
            * proportion containing the true accuracy: 0.0016 (0.061 too low, 0.9374 too high)
        * confidence intervals from unweighted sampling:
            * average width: 0.0067
            * proportion containing the true accuracy: 0.048 (0.061 too low, 0.891 too high)

## To run the experiment
- Run "pip install -r requirements.txt"
- Train a BERT model by doing "python bert_classifier.py"
- Set the parameters in the "\_\_name\_\_ == '\_\_main\_\_'" section of bootstrap.py
- Run "python bootstrap.py"
