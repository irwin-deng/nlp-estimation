## Directory structure
- archive: Contains previous executions of the experiment
- bert_classifier.py: Contains the BERT classifier model that we are fine-tuning, and code for fine-tuning that model
- bootstrap.py: Contains code for bayesian bootstrapping experiments
- common.py: Contains objects shared between files
- similarity.py: Contains functions for determining similarity between examples
- utils.py: Contains code used in multiple files

## Description of experiments
We construct confidence intervals to estimate the accuracy of a fine-tuned BERT classifier on an unlabeled dataset. We construct 95% confidence intervals by sampling a batch of labeled examples, and then calculating the accuracy of the classifier on those labeled examples. This process is repeated 10 thousand times, and then the 2.5 and 97.5 percentile values are chosen as the bounds of the bootstrapped confidence interval. We also compute the true accuracy on the unlabeled dataset as a reference. We compare confidence intervals constructed using weighted sampling (by sampling more similar examples more often), or unweighted simple random sampling from the labeled dataset.

When performing weighted sampling, the probability of sampling a specific labeled example is proportional to the inverse distance between a randomly sampled unlabeled example and that labeled examples' [CLS] vectors. We test with both the base BERT model's [CLS] vectors as well as the [CLS] vectors of a BERT model fine-tuned on MNLI. Sampling is done with repetition for both the labeled and unlabeled cases.

In addition, we compare the performance between a classifier fine-tuned on the MNLI train set and the SNLI train set.

## Experiments ran and results
1) Bootstrap sample size equal to the size of the __unlabeled__ dataset
    1) Unlabeled dataset: SNLI-test; labeled dataset: MNLI-validation_matched. A single 95% confidence interval is constructed, and the accuracy on the unlabeled dataset is measured as a reference for the following experiments.
        1) classifier fine-tuned on MNLI-train: accuracy on unlabeled dataset: 0.790, weighted CI: [0.814, 0.829], unweighted CI: [0.828, 0.843]
        2) classifier fine-tuned on SNLI-train: accuracy on unlabeled dataset: 0.905, weighted CI: [0.739, 0.756], unweighted CI: [0.717, 0.735]
    2) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: remaining 9900 examples from SNLI-test. 10 thousand 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        1) classifier fine-tuned on MNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_snli-snli.csv)):
                * average width: 0.160
                * proportion containing the true accuracy: 0.9653 (0.0139 too low, 0.0208 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.160
                * proportion containing the true accuracy: 0.9647 (0.0145 too low, 0.0208 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_snli-snli.csv)):
                * average width: 0.160
                * proportion containing the true accuracy: 0.9649 (0.0143 too low, 0.0208 too high)
        2) classifier fine-tuned on SNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_snli-snli.csv)):
                * average width: 0.113
                * proportion containing the true accuracy: 0.9665 (0.0106 too low, 0.0106 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.119
                * proportion containing the true accuracy: 0.0106 (0.0145 too low, 0.0163 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_snli-snli.csv)):
                * average width: 0.119
                * proportion containing the true accuracy: 0.9724 (0.0106 too low, 0.0170 too high)
    3) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: MNLI-validation_matched. 10 thousand 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        1) classifier fine-tuned on MNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_mnli-snli.csv)):
                * average width: 0.143
                * proportion containing the true accuracy: 0.8136 (0.0014 too low, 0.185 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.142
                * proportion containing the true accuracy: 0.8136 (0.0014 too low, 0.185 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_mnli-snli.csv)):
                * average width: 0.143
                * proportion containing the true accuracy: 0.8138 (0.0012 too low, 0.185 too high)
        2) classifier fine-tuned on SNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_mnli-snli.csv)):
                * average width: 0.170
                * proportion containing the true accuracy: 0.0027 (0.9973 too low, 0.0000 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.171
                * proportion containing the true accuracy: 0.0027 (0.9973 too low, 0.0000 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3Dunlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_mnli-snli.csv)):
                * average width: 0.171
                * proportion containing the true accuracy: 0.0027 (0.9973 too low, 0.0000 too high)
2) Bootstrap sample size equal to the size of the __labeled__ dataset
    1) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: remaining 9900 examples from SNLI-test. 10 thousand 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        1) classifier fine-tuned on MNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_snli-snli.csv)):
                * average width: 0.016
                * proportion containing the true accuracy: 0.0932 (0.4737 too low, 0.4331 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.016
                * proportion containing the true accuracy: 0.0932 (0.4737 too low, 0.4331 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3Dlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_snli-snli.csv)):
                * average width: 0.015
                * proportion containing the true accuracy: 0.0932 (0.4737 too low, 0.4331 too high)
        2) classifier fine-tuned on SNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_snli-snli.csv)):
                * average width: 0.012
                * proportion containing the true accuracy: 0.1438 (0.3810 too low, 0.4752 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.012
                * proportion containing the true accuracy: 0.2644 (0.3810 too low, 0.3546 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3Dlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_snli-snli.csv)):
                * average width: 0.012
                * proportion containing the true accuracy: 0.2644 (0.3810 too low, 0.3546 too high)
    2) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: MNLI-validation_matched. 10 thousand 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
        1) classifier fine-tuned on MNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_mnli-snli.csv)):
                * average width: 0.016
                * proportion containing the true accuracy: 0.1100 (0.4737 too low, 0.4331 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.016
                * proportion containing the true accuracy: 0.1100 (0.4737 too low, 0.4331 too high)
            * confidence intervals from unweighted sampling using fine-tuned BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3Dlabeled%3B%2010k%20bootstrap%20iters%3B%20mnli-finetuned/results_mnli-snli.csv)):
                * average width: 0.016
                * proportion containing the true accuracy: 0.1100 (0.4737 too low, 0.4331 too high)
        2) classifier fine-tuned on SNLI-train:
            * confidence intervals from weighted sampling using base BERT [CLS] vectors (data for each trial is [here](archive/2023-02-05%20size%3nlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_mnli-snli.csv)):
                * average width: 0.018
                * proportion containing the true accuracy: 0.0000 (1.0000 too low, 0.0000 too high)
            * confidence intervals from weighted sampling using fine-tuned BERT [CLS] vectors:
                * average width: 0.018
                * proportion containing the true accuracy: 0.0000 (1.0000 too low, 0.0000 too high)
            * confidence intervals from unweighted sampling (data for each trial is [here](archive/2023-02-05%20size%3nlabeled%3B%2010k%20bootstrap%20iters%3B%20snli-finetuned/results_mnli-snli.csv)):
                * average width: 0.018
                * proportion containing the true accuracy: 0.0000 (1.0000 too low, 0.0000 too high)


## To run the experiment
- Run "pip install -r requirements.txt"
- Train a BERT model by doing "python bert_classifier.py"
- Set the parameters in the "\_\_name\_\_ == '\_\_main\_\_'" section of bootstrap.py
- Run "python bootstrap.py"
