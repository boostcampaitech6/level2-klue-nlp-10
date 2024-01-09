import pandas as pd
from tqdm import tqdm
import ast
import numpy as np
from utils import num_to_label

cv1 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/submission_cv1.csv")
cv2 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/submission_cv2.csv")
cv3 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/submission_cv3.csv")
cv4 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/submission_cv4.csv")
cv5 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/submission_cv5.csv")


final = cv1.copy()

new_label_class = []
for idx in tqdm(range(len(final))):
    cv1_probs_list = np.array(ast.literal_eval(cv1.loc[idx, 'probs']))
    cv2_probs_list = np.array(ast.literal_eval(cv2.loc[idx, 'probs']))
    cv3_probs_list = np.array(ast.literal_eval(cv3.loc[idx, 'probs']))
    cv4_probs_list = np.array(ast.literal_eval(cv4.loc[idx, 'probs']))
    cv5_probs_list = np.array(ast.literal_eval(cv5.loc[idx, 'probs']))
    ensemble_probs = (cv1_probs_list + cv2_probs_list + cv3_probs_list + cv4_probs_list + cv5_probs_list)/5
    label_class = np.argmax(ensemble_probs)
    new_label_class.append(label_class)
    final.loc[idx, 'probs'] = str(ensemble_probs.tolist())

final['pred_label'] = num_to_label(new_label_class)
print(final.head())

final.to_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_5fold_cv.csv", index = False)