import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("/data/ephemeral/level2-klue-nlp-10/dataset/train/train.csv")

train['sub_type'] = train['subject_entity'].apply(lambda x :eval(x)['type'])
train['obj_type'] = train['object_entity'].apply(lambda x :eval(x)['type'])
train['type_match'] = train['sub_type'] + '-' + train['obj_type']
#train[train['type_match'] == 'PER-ORG'] = 'ORG-PER'

train['new_label'] = train['label']
train.loc[train['label'] == 'no_relation', 'new_label'] = "no_relation_" + train.loc[train['label'] == 'no_relation', 'type_match']

dataset = train[['id', 'sentence','subject_entity', 'object_entity', 'source', 'label', 'new_label']]
print(dataset.head())

# 5 fold dataset
skf = StratifiedKFold(n_splits = 5, shuffle=True, random_state=623)
n_iter = 0

features = dataset.iloc[:,:]
label = pd.DataFrame(dataset['new_label'])

for train_idx, val_idx in skf.split(features, label):
    n_iter += 1
    print(f'--------------------{n_iter}번째 KFold-------------------')
    print(f'train_idx_len : {len(train_idx)} / test_idx_len : {len(val_idx)}')

    train_set = features.iloc[train_idx, :]
    val_set = features.iloc[val_idx, :]

    print(train_idx[:10])
  # print(train_set['new_label'].value_counts())
    print(val_idx[:10])
   # print(val_set['label'].value_counts())
    print(f'{n_iter}번째 단일 fold 데이터 완성')

    train_set.to_csv(f"/data/ephemeral/level2-klue-nlp-10/dataset/fold5cv/train_cv_{n_iter}.csv", index=False)
    val_set.to_csv(f"/data/ephemeral/level2-klue-nlp-10/dataset/fold5cv/val_cv_{n_iter}.csv", index=False)