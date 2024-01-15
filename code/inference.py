from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from datasets import RE_Dataset, Recent_Dataset
from preprocessing import Preprocessor, Prompt, tokenized_dataset, get_entity_loc, Recent
from utils import set_seed, num_to_label
from model import BaseModel, MtbModel, RecentModel


def inference(model, tokenized_sent, device, mtb, recent):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=32, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    # matching the blank인 경우
    if mtb:
      with torch.no_grad():
        inputs = {'input_ids' : data['input_ids'].to(device),
                    'attention_mask' :data['attention_mask'].to(device),
                    'token_type_ids' :data['token_type_ids'].to(device),
                    'matching_the_blanks_ids' : data['matching_the_blanks_ids']}
        outputs = model(**inputs)
      logits = outputs['logits']

    elif recent:
      with torch.no_grad():
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device),
            restrict_num = data['restrict_num'].to(device)
            )
      logits = outputs['logits']
    
    # 일반 모델 예측일 경우
    else: 
      with torch.no_grad():
        outputs = model(
            input_ids=data['input_ids'].to(device),
            attention_mask=data['attention_mask'].to(device),
            token_type_ids=data['token_type_ids'].to(device)
            )
      logits = outputs[0]
    
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()




def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  set_seed(42)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  MODEL_NAME = "klue/roberta-large"
  TEST_PATH = "../dataset/test/test_data.csv"
  LABEL_CNT = 30
  P_CONFIG = {'prompt_kind' : 's_and_o',  
                'preprocess_method' : 'typed_entity_marker_punct', 
                'and_marker' : '와',    
                'add_question' : False,    
                'only_sentence' : False,   
                'loss_name' : 'CrossEntropy', 
                'recent' : True,
                'matching_the_blank' : None} 
    

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  ## load test datset
  test_dataset = pd.read_csv(TEST_PATH)

  if P_CONFIG['recent']:
      recent = Recent()
      restrict_num = recent.find_restrict_num(test_dataset)

  # Test dataset Prompt 생성
  prompt = Prompt()
  test_prompt = prompt.make_prompt(test_dataset, kind=P_CONFIG['prompt_kind'], marker=P_CONFIG['preprocess_method'], and_marker=P_CONFIG['and_marker'])
  
  
  # Test dataset Sentence 전처리
  preprocessor = Preprocessor()
  test_sentence, tokenizer = getattr(preprocessor, P_CONFIG['preprocess_method'])(test_dataset, tokenizer, add_question=P_CONFIG['add_question'], and_marker=P_CONFIG['and_marker'])

  # tokenizing Test dataset
  max_length = 1000 if P_CONFIG['matching_the_blank'] else 256
  tokenized_test = tokenized_dataset(tokenizer, test_prompt, test_sentence, max_length, only_sentence=P_CONFIG['only_sentence'])
  
  if P_CONFIG['matching_the_blank']:
    test_entitiy_marker_loc_ids = get_entity_loc(tokenizer=tokenizer, tokenized_sentences = tokenized_test, config=P_CONFIG)
    tokenized_test['matching_the_blanks_ids'] = torch.tensor(test_entitiy_marker_loc_ids, dtype=torch.int64)

  # Test label 준비
  test_label = list(map(int, test_dataset['label'].values))

  if P_CONFIG['recent']:
    re_test_dataset = Recent_Dataset(tokenized_test, test_label, restrict_num) 

  else:
    re_test_dataset = RE_Dataset(tokenized_test , test_label)

  # setting model hyperparameter
  if P_CONFIG['matching_the_blank']:
    model = MtbModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT, tokenizer=tokenizer, mtb_type=P_CONFIG['matching_the_blank'])

  elif P_CONFIG['recent']:
    model = RecentModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT, tokenizer=tokenizer, restrict_num=restrict_num)

  else:
    model = BaseModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT, tokenizer=tokenizer)

  checkpoint = torch.load(args.model_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.parameters
  model.to(device)

  ## predict answer
  pred_answer, output_prob = inference(model, re_test_dataset, device, P_CONFIG['matching_the_blank'], P_CONFIG['recent']) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id' : test_dataset['id'],'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/restrict_prediction.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--model_path', type=str, default="./best_model/bestmodel_restrict.pth")
    args = parser.parse_args()
    print(args)
    main(args)
    
