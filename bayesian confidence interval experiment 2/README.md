## Directory structure
- archive: Contains previous executions of the experiment
- bert_classifier.py: Contains the BERT classifier model that we are fine-tuning, and code for fine-tuning that model
- bootstrap.py: Contains code for bayesian bootstrapping experiments
- common.py: Contains objects shared between files
- similarity.py: Contains functions for determining similarity between examples
- utils.py: Contains code used in multiple files

## Description of experiments
We construct confidence intervals to estimate the accuracy of a fine-tuned BERT classifier on an unlabeled dataset. We construct 95% confidence intervals by sampling a batch of labeled examples of size equal to the length of the unlabeled dataset, and then calculating the accuracy of the classifier on those labeled examples. This process is repeated 10 thousand times, and then the 2.5 and 97.5 percentile values are chosen as the bounds of the bootstrapped confidence interval. We also compute the true accuracy on the unlabeled dataset as a reference. We compare confidence intervals constructed using weighted sampling (by sampling more similar examples more often), or unweighted simple random sampling from the labeled dataset.

When performing weighted sampling, the probability of sampling a specific labeled example is proportional to the inverse distance between a randomly sampled unlabeled example and that labeled examples' [CLS] vectors in the base BERT model. Sampling is done with repetition for both the labeled and unlabeled cases.

In addition, we compare the performance between a classifier fine-tuned on the MNLI train set and the SNLI train set.

## Experiments ran and results
1) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: remaining 9900 examples from SNLI-test. 1000 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
    1) classifier fine-tuned on MNLI-train: proportion of weighted CIs containing the true accuracy: 0.089, proportion of unweighted CIs containing the true accuracy: 0.089. Data for each trial is [here](archive/2023-02-01%202%20experiments%3B%20mnli-finetuned/results_snli-snli.csv)
    2) classifier fine-tuned on SNLI-train: proportion of weighted CIs containing the true accuracy: 0.129, proportion of unweighted CIs containing the true accuracy: 0.252. Data for each trial is [here](archive/2023-02-01%202%20experiments%3B%20snli-finetuned/results_snli-snli.csv)
2) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: MNLI-validation_matched. 1000 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
    1) classifier fine-tuned on MNLI-train: proportion of weighted CIs containing the true accuracy: 0.129, proportion of unweighted CIs containing the true accuracy: 0.129. Data for each trial is [here](archive/2023-02-01%202%20experiments%3B%20mnli-finetuned/results_mnli-snli.csv)
    2) classifier fine-tuned on SNLI-train: proportion of weighted CIs containing the true accuracy: 0.000, proportion of unweighted CIs containing the true accuracy: 0.000. Data for each trial is [here](archive/2023-02-01%202%20experiments%3B%20snli-finetuned/results_mnli-snli.csv).

## To run the experiment
- Run "pip install -r requirements.txt"
- Set the parameters in the "\_\_name\_\_ == '\_\_main\_\_'" section of bootstrap.py
- Run "python bootstrap.py"
- The fine-tuned model will be saved to "bert_classifier.pt"
