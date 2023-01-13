Each `.tar.gz` file contains the code of a previous run, as well as the console output in `log.txt`

## Description of runs
- 2022-12-11: Initial implementation comparing weighted vs unweighted sampling. These are buggy
- 2023-01-12: Fixed some bugs with the training loop. These are still buggy
- 2023-01-13: Fixed more bugs with validation metrics. After 3 epochs, the accuracy on the test set is 0.72 for weighted sampling, and 0.78 for unweighted sampling. The weighted sampler used approximate k-nearest neighbors with k=10 to find k candidates in the labeled dataset for each unlabeled data point. During the training loop, it sampled among the k with probability proportional to the inverse Euclidean distance between the [CLS] vectors outputted from the base BERT model.
