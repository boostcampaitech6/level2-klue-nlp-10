import torch
import torch.nn as nn 
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments

class BaseModel(nn.Module):
    def __init__(self, model_name, label_cnt, tokenizer):
        super(BaseModel, self).__init__()
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = label_cnt
        self.model =  AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

        # 추가된 토큰을 반영해서 모델 embedding size 업데이트
        self.model.resize_token_embeddings(len(tokenizer)) 

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs


