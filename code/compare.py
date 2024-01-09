import pandas as pd

cv_result = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_5fold_cv.csv")
ensembel2_result = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble2.csv")

cv_result_list = cv_result['pred_label'].to_list()
ensembel2_result_list = ensembel2_result['pred_label'].to_list()

cnt = 0
for cv, en in zip(cv_result_list, ensembel2_result_list):
    if cv != en:
        cnt += 1

print(cnt)


