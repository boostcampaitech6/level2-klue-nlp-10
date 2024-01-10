from torch import nn
from transformers import Trainer
from loss_function import FocalLoss

class CustomTrainer(Trainer):
    def __init__(self, loss_name, num_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name
        self.num_labels = num_labels
        
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.loss_name == "CrossEntropy":
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.loss_name == "FocalLoss":
            loss_fct = FocalLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss