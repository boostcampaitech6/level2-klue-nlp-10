import pickle as pickle
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import RE_Dataset
import numpy as np

from preprocessing import Preprocessor, Prompt, tokenized_dataset, get_entity_loc
from metrics import compute_metrics
from utils import set_seed, label_to_num
from split_data import Spliter
from model import BaseModel, MtbModel
from custom_trainer import CustomTrainer


def train():
    SEED = 42
    set_seed(SEED)
    # load model and tokenizer
    
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/roberta-large"
    TRAIN_PATH = "../dataset/train/train.csv"
    LABEL_CNT = 30
    P_CONFIG = {'prompt_kind' : 's_and_o',  # ['s_sep_o', 's_and_o', 'quiz']
                'preprocess_method' : 'typed_entity_marker', # ['baseline_preprocessor', 'entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct']
                'and_marker' : '와',       # ['와', '그리고', '&', '[SEP]']
                'add_question' : False,     # sentence 뒷 부분에 "sub_e 와 obj_e의 관계는 무엇입니까?""
                'only_sentence' : False,   # True : (sentence) / False : (prompt + sentence)
                'loss_name' : 'CrossEntropy',  # loss fuction 선택: 'CrossEntropy', 'FocalLoss'
                'matching_the_blank' : 'entity_start'} # [None, 'entity_start', 'entity_start_end']
    

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # No split으로 수정
    train_dataset, dev_dataset = Spliter.no_split(TRAIN_PATH)

    # Train, Dev Prompt 생성
    prompt = Prompt()
    train_prompt = prompt.make_prompt(train_dataset, kind=P_CONFIG['prompt_kind'], marker=P_CONFIG['preprocess_method'], and_marker=P_CONFIG['and_marker'])
    dev_prompt = prompt.make_prompt(dev_dataset, kind=P_CONFIG['prompt_kind'], marker=P_CONFIG['preprocess_method'], and_marker=P_CONFIG['and_marker'])

    # Train, Dev 전처리
    preprocessor = Preprocessor()
    train_sentence, tokenizer = getattr(preprocessor, P_CONFIG['preprocess_method'])(train_dataset, tokenizer, add_question=P_CONFIG['add_question'], and_marker=P_CONFIG['and_marker'])
    dev_sentence, tokenizer = getattr(preprocessor, P_CONFIG['preprocess_method'])(dev_dataset, tokenizer, add_question=P_CONFIG['add_question'], and_marker=P_CONFIG['and_marker'])

    # Train, Dev 라벨 생성
    train_label = label_to_num(train_dataset['label'].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    max_length = 1000 if P_CONFIG['matching_the_blank'] else 256
    tokenized_train = tokenized_dataset(tokenizer, train_prompt, train_sentence, max_length, only_sentence=P_CONFIG['only_sentence'])
    tokenized_dev = tokenized_dataset(tokenizer, dev_prompt, dev_sentence, max_length, only_sentence=P_CONFIG['only_sentence'])
    
    # Matching the blank 사용시 tokenizer에 matching_the_blanks_ids 정보 추가
    if P_CONFIG['matching_the_blank']:
        train_entitiy_marker_loc_ids = get_entity_loc(tokenizer=tokenizer, tokenized_sentences = tokenized_train, config=P_CONFIG)
        dev_entitiy_marker_loc_ids = get_entity_loc(tokenizer=tokenizer, tokenized_sentences = tokenized_dev, config=P_CONFIG)
        tokenized_train['matching_the_blanks_ids'] = torch.tensor(train_entitiy_marker_loc_ids, dtype=torch.int64)
        tokenized_dev['matching_the_blanks_ids'] = torch.tensor(dev_entitiy_marker_loc_ids, dtype=torch.int64)


    # make dataset for pytorch.
    re_train_dataset = RE_Dataset(tokenized_train, train_label)
    re_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('DEVICE : ', device)


    # setting model hyperparameter
    if P_CONFIG['matching_the_blank']:
        model = MtbModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT, tokenizer=tokenizer, mtb_type=P_CONFIG['matching_the_blank'])
    else:
        model = BaseModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT, tokenizer=tokenizer)
    print('MODEL CONFIG')
    print(model.model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
      output_dir='./results',          # output directory
      save_total_limit=1,              # number of total save model.
      save_steps=700,                 # model saving step.
      num_train_epochs=4,              # total number of training epochs
      learning_rate=5e-5,               # learning_rate
      per_device_train_batch_size=32,  # batch size per device during training
      per_device_eval_batch_size=32,   # batch size for evaluation
      warmup_steps=700,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./logs',            # directory for storing logs
      logging_steps=100,              # log saving step.
      evaluation_strategy='steps', # evaluation strategy to adopt during training
                                  # `no`: No evaluation during training.
                                  # `steps`: Evaluate every `eval_steps`.
                                  # `epoch`: Evaluate every end of epoch.
      eval_steps = 700,            # evaluation step.
      load_best_model_at_end = True 
    )

    trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    loss_name=P_CONFIG['loss_name'],
    num_labels=LABEL_CNT,
    args=training_args,                  # training arguments, defined above
    train_dataset=re_train_dataset,         # training dataset
    eval_dataset=re_dev_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,         # define metrics function
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]  # early_stopping 
    # early stopping사용을 원하지 않는다면 그냥 callbacks 줄을 주석 처리 하면됨
    )
  
    # train model
    trainer.train()
    # git에 올린 코드
    model_state_dict = model.state_dict()
    torch.save({'model_state_dict' : model_state_dict}, './best_model/bestmodel.pth')
    
def main():
    train()

if __name__ == '__main__':
    main()
