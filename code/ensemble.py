import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from utils import num_to_label

gj = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_gyeungjae.csv")  # 34 - 72.2240	| 76.6593  : eopch3,s[SEP]question(&)
gj2 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_andchange.csv")  # 37 - 72.4976 | 76.4745 : eopch3,e_and_e_sep_s(&)
jw = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_jongwon.csv")  # 31 - 73.0623  | 79.0710 : epoch3 - typed_entity_marker_punct - question
sota = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_sota.csv") # 47 73.4566 | 77.4988 : epoch3 - sao - 와 - temp
cv = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_5fold_cv.csv") # 66 73.3461 | 79.8736 : sao - 와 - temp

final = gj.copy()
new_label_class = []
for idx in tqdm(range(len(gj))):
    gj_probs_list = ast.literal_eval(gj.loc[idx, 'probs'])
    gj2_probs_list = ast.literal_eval(gj2.loc[idx, 'probs'])
    jw_probs_list = ast.literal_eval(jw.loc[idx, 'probs'])
    sota_probs_list = ast.literal_eval(sota.loc[idx, 'probs'])
    cv_probs_list = ast.literal_eval(cv.loc[idx, 'probs'])
    ensemble_probs = [(0.2)*x + (0.2)*y + (0.2)*z + (0.2)*k + (0.2)*c for x, y, z, k, c in zip(gj_probs_list, gj2_probs_list, jw_probs_list, sota_probs_list, cv_probs_list)]
    label_class = np.argmax(np.array(ensemble_probs))
    new_label_class.append(label_class)
    final.loc[idx, 'probs'] = str(ensemble_probs)

final['pred_label'] = num_to_label(new_label_class)
print(final.head())

final.to_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble5.csv", index = False)