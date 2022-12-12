Directory structure:
- archive: Contains previous executions of the experiment
- bootstrap.py: Contains code for bayesian bootstrapping
- similarity.py: Contains functions for determining similarity between samples
- bert_model.py: Contains the BERT classifier model that we are fine-tuning

In this experiment, we attempt to fine-tune a BERT classifier on the SNLI dataset by
training the classifier with samples from the MNLI dataset. We can perform a weighted
sampling from the MNLI dataset by choosing samples from the MNLI dataset that most
closely match the SNLI train dataset. To find the most similar samples, we use PyNNDescent,
which gives the approximate nearest neighbors. For comparison, we can also perform an
unweighted sampling by randomly choosing samples from the MNLI dataset.

To run the experiment:
- Run "pip install -r requirements.txt"
- Set the parameters accordingly in the "__name__ == '__main__'" section of bootstrap.py
- Run "python bootstrap.py"
- The fine-tuned model will be saved to "bert_classifier.pt"
