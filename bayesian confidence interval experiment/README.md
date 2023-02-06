## Directory structure
- archive: Contains previous executions of the experiment
- bert_classifier.py: Contains the BERT classifier model that we are fine-tuning, and code for fine-tuning that model
- bootstrap.py: Contains code for bayesian bootstrapping experiments
- common.py: Contains objects shared between files
- similarity.py: Contains functions for determining similarity between examples
- utils.py: Contains code used in multiple files

## Description of experiments
We construct confidence intervals to estimate the accuracy of a fine-tuned BERT classifier on an unlabeled dataset. We construct 95% confidence intervals by sampling a batch of labeled examples of size equal to the length of the __unlabeled__ dataset, and then calculating the accuracy of the classifier on those labeled examples. This process is repeated 10 thousand times, and then the 2.5 and 97.5 percentile values are chosen as the bounds of the bootstrapped confidence interval. We also compute the true accuracy on the unlabeled dataset as a reference. We compare confidence intervals constructed using weighted sampling (by sampling more similar examples more often), or unweighted simple random sampling from the labeled dataset.

When performing weighted sampling, the probability of sampling a specific labeled example is proportional to the inverse distance between a randomly sampled unlabeled example and that labeled examples' [CLS] vectors in the base BERT model. Sampling is done with repetition for both the labeled and unlabeled cases.

In addition, we compare the performance between a classifier fine-tuned on the MNLI train set and the SNLI train set.

## Experiments ran and results
1) Unlabeled dataset: SNLI-test; labeled dataset: MNLI-validation_matched. A single 95% confidence interval is constructed, and the accuracy on the unlabeled dataset is measured as a reference for the following experiments.
    1) classifier fine-tuned on MNLI-train: accuracy on unlabeled dataset: 0.790, weighted CI: [0.814, 0.829], unweighted CI: [0.828, 0.843]
    2) classifier fine-tuned on SNLI-train: accuracy on unlabeled dataset: 0.905, weighted CI: [0.739, 0.756], unweighted CI: [0.717, 0.735]
2) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: remaining 9900 examples from SNLI-test. 10 thousand 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
    1) classifier fine-tuned on MNLI-train (data for each trial is [here](archive/2023-02-05%202%20experiments%2C%2010k%20bootstrap%20iters%3B%20finetuned%20on%20MNLI/results_snli-snli.csv)):
        * confidence intervals from weighted sampling:
            * average width: 0.160
            * proportion containing the true accuracy: 0.9653
            * proportion lower than the true accuracy: 0.0139
            * proportion higher than the true accuracy: 0.0208
        * confidence intervals from unweighted sampling:
            * average width: 0.160
            * proportion containing the true accuracy: 0.9649 
            * proportion lower than the true accuracy: 0.0143
            * proportion higher than the true accuracy: 0.0208
    2) classifier fine-tuned on SNLI-train (data for each trial is [here](archive/2023-02-05%202%20experiments%2C%2010k%20bootstrap%20iters%3B%20finetuned%20on%20SNLI/results_snli-snli.csv)):
        * confidence intervals from weighted sampling:
            * average width: 0.113
            * proportion containing the true accuracy: 0.9665
            * proportion lower than the true accuracy: 0.0106
            * proportion higher than the true accuracy: 0.0106
        * confidence intervals from unweighted sampling:
            * average width: 0.119
            * proportion containing the true accuracy: 0.9724 
            * proportion lower than the true accuracy: 0.0106
            * proportion higher than the true accuracy: 0.0170
3) Unlabeled dataset: 100 examples from SNLI-test; labeled dataset: MNLI-validation_matched. 10 thousand 95% confidence intervals were constructed (using different subsets of SNLI-test as the unlabeled datasets) to evaluate the proportion of time that the confidence intervals contained the true accuracy on the unlabeled dataset.
    1) classifier fine-tuned on MNLI-train (data for each trial is [here](archive/2023-02-05%202%20experiments%2C%2010k%20bootstrap%20iters%3B%20finetuned%20on%20MNLI/results_mnli-snli.csv)):
        * confidence intervals from weighted sampling:
            * average width: 0.143
            * proportion containing the true accuracy: 0.8136
            * proportion lower than the true accuracy: 0.0014
            * proportion higher than the true accuracy: 0.185
        * confidence intervals from unweighted sampling:
            * average width: 0.143
            * proportion containing the true accuracy: 0.8138 
            * proportion lower than the true accuracy: 0.0012
            * proportion higher than the true accuracy: 0.185
    2) classifier fine-tuned on SNLI-train (data for each trial is [here](archive/2023-02-05%202%20experiments%2C%2010k%20bootstrap%20iters%3B%20finetuned%20on%20SNLI/results_mnli-snli.csv)):
        * confidence intervals from weighted sampling:
            * average width: 0.170
            * proportion containing the true accuracy: 0.0027
            * proportion lower than the true accuracy: 0.9973
            * proportion higher than the true accuracy: 0.0000
        * confidence intervals from unweighted sampling:
            * average width: 0.171
            * proportion containing the true accuracy: 0.0027 
            * proportion lower than the true accuracy: 0.9973
            * proportion higher than the true accuracy: 0.0000

## To run the experiment
- Run "pip install -r requirements.txt"
- Set the parameters in the "\_\_name\_\_ == '\_\_main\_\_'" section of bootstrap.py
- Run "python bootstrap.py"
- The fine-tuned model will be saved to "bert_classifier.pt"
