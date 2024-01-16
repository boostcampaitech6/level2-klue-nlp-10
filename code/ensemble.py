import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from utils import num_to_label

# mix best (#86 리더보드 결과 재현)

best1 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_31.csv")  # 31
best2 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_34.csv")  # 34
best3 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_36.csv")  # 36
best4 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_47.csv")  # 47 
best5 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_71.csv")  # 71
best6 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_66.csv")  # 66
best7 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_84.csv")  # 84

final = best1.copy()
new_label_class = []
for idx in tqdm(range(len(best1))):
    best1_probs_list = ast.literal_eval(best1.loc[idx, 'probs'])
    best2_probs_list = ast.literal_eval(best2.loc[idx, 'probs'])
    best3_probs_list = ast.literal_eval(best3.loc[idx, 'probs'])
    best4_probs_list = ast.literal_eval(best4.loc[idx, 'probs'])
    best5_probs_list = ast.literal_eval(best5.loc[idx, 'probs'])
    best6_probs_list = ast.literal_eval(best6.loc[idx, 'probs'])
    best7_probs_list = ast.literal_eval(best7.loc[idx, 'probs'])
    # #34 - 15% #36 - 17% #31 - 17% #47 - 17% #66 - 17% #84 - 17%  (BEST)
    # 31 34 36 66 84 71 = 17 17 17 17 16 16  -> 117개
    # 31 34 36 66 84 71 = 16 16 17 17 17 17  -> 125개
    # 31 34 36 66 84 93 71 112 = 14 14 14 14 14 10 10 10 -> 139개
    # 31 34 36 66 84 93 71 112 = 13 13 13 13 13 12 12 11 -> 147개
    # 31 34 36 84 71 -> 20 20 20 20 20 -> 151개
    # 31 34 36 84 71 -> 20 15 15 30 30 -> 172개
    # 31 34 36 84 71 66 -> 18 18 16 16 16 16 -> 120개
    # 31 34 36 84 71 66 105-> 14 14 15 15 14 14 14 -> 149개
    # 31 34 36 71 66 84 -> 17 17 16 16 17 17 -> 116개
    # 31 34 36 71 66 84 -> 16 16 17 17 17 17 -> 125개
    # 31 34 36 47 71 66 84 -> 15 15 15 15 14 14 15 -> 86
    ensemble_probs = [(0.15)*x + (0.15)*y + (0.15)*z + (0.15)*a + (0.14)*b + (0.14)*c + (0.15)*d
                      for x, y, z, a, b, c, d in zip(best1_probs_list, 
                                                    best2_probs_list, 
                                                    best3_probs_list, 
                                                    best4_probs_list, 
                                                    best5_probs_list,
                                                    best6_probs_list, best7_probs_list
                                                                )]
    label_class = np.argmax(np.array(ensemble_probs))
    new_label_class.append(label_class)
    final.loc[idx, 'probs'] = str(ensemble_probs)

final['pred_label'] = num_to_label(new_label_class)
print(final.head())

final.to_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble7_0116.csv", index = False) 