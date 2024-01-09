import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from utils import num_to_label

gj = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_gyeungjae.csv")
jw = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_jongwon.csv")

print(gj.head())
print(len(gj))

final = gj
new_label_class = []
for idx in tqdm(range(len(gj))):
    gj_probs_list = ast.literal_eval(gj.loc[idx, 'probs'])
    jw_probs_list = ast.literal_eval(jw.loc[idx, 'probs'])
    ensemble_probs = [(0.5)*x + (0.5)*y for x, y in zip(gj_probs_list, jw_probs_list)]
    label_class = np.argmax(np.array(ensemble_probs))
    new_label_class.append(label_class)
    final.loc[idx, 'probs'] = str(ensemble_probs)

final['pred_label'] = num_to_label(new_label_class)
print(final.head())

final.to_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble2.csv", index = False)




