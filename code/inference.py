from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from datasets import RE_Dataset
from preprocessing import Preprocessor, Prompt, tokenized_dataset
from utils import set_seed, num_to_label
from model import BaseModel


def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
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
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  ## load test datset
  test_dataset = pd.read_csv(TEST_PATH)
  # Test dataset Prompt 생성
  test_prompt = Prompt.sub_sep_obj_prompt(test_dataset)
  # Test dataset Sentence 전처리
  test_sentence = Preprocessor.baseline_preprocessor(test_dataset)
  # tokenizing Test dataset
  tokenized_test = tokenized_dataset(tokenizer, test_prompt, test_sentence)
  # Test label 준비
  test_label = list(map(int, test_dataset['label'].values))
  re_test_dataset = RE_Dataset(tokenized_test , test_label)

  ## load my model
  model = BaseModel(model_name=MODEL_NAME, label_cnt=LABEL_CNT)
  checkpoint = torch.load(args.model_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.parameters
  model.to(device)

  ## predict answer
  pred_answer, output_prob = inference(model, re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id' : test_dataset['id'],'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission_cv5.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--model_path', type=str, default="./best_model/bestmodel_cv5.pth")
    args = parser.parse_args()
    print(args)
    main(args)
    
