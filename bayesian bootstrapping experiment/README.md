Directory structure:
- archive: Contains previous executions of the experiment
- bootstrap.py: Contains code for bayesian bootstrapping
- similarity.py: Contains functions for determining similarity between samples
- bert_model.py: Contains the BERT classifier model that we are fine-tuning

In this experiment, we attempt to fine-tune a BERT classifier to predict on an unlabeled dataset (SNLI), using data from a labeled dataset (MNLI). We split the unlabeled dataset into a train and validation set. We train our model by taking batches from the unlabeled train set and substituting labeled data points in place of our unlabeled batch.
To efficiently find the most similar labeled data points, we use PyNNDescent, which constructs an array of the k approximately most similar labeled data points to each unlabeled data point. Currently, the probability of sampling a specific labeled data point among the k is proportional to the inverse distance between the unlabeled and labeled data points' BERT [CLS] vectors.
For comparison, we can also perform an unweighted sampling by randomly choosing samples from the MNLI dataset.

To run the experiment:
- Run "pip install -r requirements.txt"
- Set the parameters accordingly in the "__name__ == '__main__'" section of bootstrap.py
- Run "python bootstrap.py"
- The fine-tuned model will be saved to "bert_classifier.pt"
