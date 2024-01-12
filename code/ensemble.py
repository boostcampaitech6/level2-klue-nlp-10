import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from utils import num_to_label

gj = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_gyeungjae.csv")  # 34 - 72.2240	| 76.6593  : eopch3,s[SEP]question(&)
gj2 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_andchange.csv")  # 36 - 72.5172 | 75.1629 : eopch3,e_e_and_e_sep_s(와)
jw = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_jongwon.csv")  # 31 - 73.0623  | 79.0710 : epoch3 - typed_entity_marker_punct - question
sota = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_sota.csv") # 47 73.4566 | 77.4988 : epoch3 - sao - 와 - temp
cv = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_5fold_cv.csv") # 66 73.3461 | 79.8736 : sao - 와 - temp
mtb = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_mtb.csv")  # 73 73.3858 | 79.2777 : mtb 적용
poh = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_poh_plus.csv")  # 70 72.1904 | 76.9549  : POH 경우만 데이터 증강 적용

final = gj.copy()
new_label_class = []
for idx in tqdm(range(len(gj))):
    gj_probs_list = ast.literal_eval(gj.loc[idx, 'probs'])
  #  gj2_probs_list = ast.literal_eval(gj2.loc[idx, 'probs'])
    jw_probs_list = ast.literal_eval(jw.loc[idx, 'probs'])
    sota_probs_list = ast.literal_eval(sota.loc[idx, 'probs'])
    cv_probs_list = ast.literal_eval(cv.loc[idx, 'probs'])
   # mtb_probs_list = ast.literal_eval(mtb.loc[idx, 'probs'])
    poh_probs_list = ast.literal_eval(poh.loc[idx, 'probs'])
    ensemble_probs = [(0.2)*x + (0.2)*y + (0.2)*z + (0.2)*a + (0.2)*b for x, y, z, a, b in zip(gj_probs_list, 
                                                                                                   #  gj2_probs_list, 
                                                                                                     jw_probs_list, 
                                                                                                     sota_probs_list, 
                                                                                                     cv_probs_list, 
                                                                                                     #mtb_probs_list, 
                                                                                                     poh_probs_list
                                                                                                 )]
    label_class = np.argmax(np.array(ensemble_probs))
    new_label_class.append(label_class)
    final.loc[idx, 'probs'] = str(ensemble_probs)

final['pred_label'] = num_to_label(new_label_class)
print(final.head())

final.to_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble5_0112.csv", index = False)