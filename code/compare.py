import pandas as pd

best = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble4.csv")
e5 = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/code/prediction/output_ensemble5.csv")

a = best['pred_label'].to_list()
b = e5['pred_label'].to_list()

cnt = 0
idx = 0
for aa, bb in zip(a, b):
    if aa != bb:
        print(idx)
        cnt += 1
    idx += 1

print(cnt)