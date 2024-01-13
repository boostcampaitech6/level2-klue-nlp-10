import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from utils import num_to_label

# mix best 2

best1 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_86.csv")  # 86
best2 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_68.csv")  # 68


final = best1.copy()
new_label_class = []
for idx in tqdm(range(len(best1))):
    best1_probs_list = ast.literal_eval(best1.loc[idx, 'probs'])
    best2_probs_list = ast.literal_eval(best2.loc[idx, 'probs'])
  
      #34 - 15% #36 - 17% #31 - 17% #47 - 17% #66 - 17% #84 - 17%
    ensemble_probs = [(0.5)*x + (0.5)*y  for x, y in zip(best1_probs_list, best2_probs_list)]
    label_class = np.argmax(np.array(ensemble_probs))
    new_label_class.append(label_class)
    final.loc[idx, 'probs'] = str(ensemble_probs)

final['pred_label'] = num_to_label(new_label_class)
print(final.head())

final.to_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble_best_mix2.csv", index = False)

