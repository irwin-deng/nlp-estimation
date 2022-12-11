import torch
from transformers import BertForSequenceClassification

class BertClassifier(torch.nn.Module):
    def __init__(self, n_labels: int = 3):
        super(BertClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=n_labels)
    
    def forward(self, input_ids, attention_mask):
        return self.bert.forward(input_ids=input_ids,
            attention_mask=attention_mask, return_dict=False)[0]
