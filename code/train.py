import pickle as pickle
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import RE_Dataset
import numpy as np


from preprocessing import Preprocessor, Prompt, tokenized_dataset
from metrics import compute_metrics
from utils import set_seed, label_to_num
from split_data import Spliter
from model import BaseModel

def train():
    set_seed(42)
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/roberta-large"
    TRAIN_PATH = "../dataset/train/train.csv"
    LABEL_CNT = 30
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # [TODO] KFold 용 load dataset 및 구조 구상
    # Train Dev Split
    train_dataset, dev_dataset = Spliter.stratified_split(TRAIN_PATH)
    # Train, Dev Prompt 생성
    train_prompt = Prompt.sub_sep_obj_prompt(train_dataset)
    dev_prompt = Prompt.sub_sep_obj_prompt(dev_dataset)

    # Train, Dev 전처리
    train_sentence = Preprocessor.baseline_preprocessor(train_dataset)
    dev_sentence = Preprocessor.baseline_preprocessor(dev_dataset)

    # Train, Dev 라벨 생성
    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(tokenizer, train_prompt, train_sentence)
    tokenized_dev = tokenized_dataset(tokenizer, dev_prompt, dev_sentence)

    # make dataset for pytorch.
    re_train_dataset = RE_Dataset(tokenized_train, train_label)
    re_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE : ', device)
    

    # setting model hyperparameter
    model = BaseModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT)
    print('MODEL CONFIG')
    print(model.model.config)
    model.parameters
    model.to(device)
    


    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
      output_dir='./results',          # output directory
      save_total_limit=2,              # number of total save model.
      save_steps=500,                 # model saving step.
      num_train_epochs=20,              # total number of training epochs
      learning_rate=5e-5,               # learning_rate
      per_device_train_batch_size=16,  # batch size per device during training
      per_device_eval_batch_size=32,   # batch size for evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps = 500,            # evaluation step.
      load_best_model_at_end = True 
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=re_train_dataset,         # training dataset
      eval_dataset=re_dev_dataset,             # evaluation dataset
      compute_metrics=compute_metrics         # define metrics function
    )

    # train model
    trainer.train()
    model_state_dict = model.state_dict()
    torch.save({'model_state_dict' : model_state_dict}, './best_model/bestmodel.pth')


def main():
    train()

if __name__ == '__main__':
    main()
