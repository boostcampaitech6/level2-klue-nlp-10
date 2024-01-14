#%%
import pandas as pd
import numpy as np
from googletrans import Translator
from tqdm.auto import tqdm
import time

#%%
train_path = '../dataset/train/train.csv'
train_df = pd.read_csv(train_path)
train_df_len = len(train_df)

trans_train_path = '../dataset/train/trans_train.csv'
trans_train = pd.read_csv(trans_train_path)

#%%
# back translation
translator = Translator()

def google_ko2en2ko(ko_text, translator):
    ko_en_result = translator.translate(ko_text, dest = 'en').text
    en_ko_result = translator.translate(ko_en_result, dest = 'ko').text
    return en_ko_result


sen1_list = train_df['sentence'] 

for idx, sentence in enumerate(tqdm(sen1_list)):
     while True:
        try:
            result = google_ko2en2ko(sentence, translator)
            train_df.loc[idx, 'sentence'] = result
            break  
        except Exception as e:
            print(f"번 째 index에서 에러 발생 {idx} {e}")
            time.sleep(1)  # 1초의 지연 후 다시 시도

train_df.to_csv('./dataset/train/trans_train.csv', index=False)



# test_text = "나는지금코드를보고있는중이다."
# print(google_ko2en2ko(test_text, translator))

# for idx, sentence in enumerate(tqdm(sen1_list)):
#     result =  google_ko2en2ko(sentence, translator)
#     test.loc[idx,'sentence_1'] = result

# for idx, sentence in enumerate(tqdm(sen2_list)):
#     result =  google_ko2en2ko(sentence, translator)
#     test.loc[idx,'sentence_2'] = result

#%%
smooth_trans = trans_train[(trans_train['label'] != "no_relation") & (trans_train['label'] != "org:top_members/employees") & (trans_train['label'] != "per:employee_of")]
#augmented_df = pd.merge(train_df, smooth_trans)
smooth_trans['id'] = range(1, len(smooth_trans)+1)
smooth_trans.reset_index(drop=True, inplace=True)
smooth_trans.to_csv('../dataset/train/smooth_trans.csv', index=False)

augmented_df = pd.merge(train_df, smooth_trans, how='outer')
augmented_df['id'] = range(1, len(augmented_df)+1)
augmented_df.reset_index(drop=True, inplace=True)
augmented_df.to_csv('../dataset/train/augmented_train.csv', index=False)
# %%
import pandas as pd
import json
import ast

test_path = '../dataset/test/tmp.csv'
test_df = pd.read_csv(test_path)

#test_df = test_df['subject_entity']
subject_entity = test_df['subject_entity'].apply(lambda x: ast.literal_eval(x)['type'])

print(subject_entity)


# %%
import pandas as pd
import ast
test_path = '../dataset/test/test_data.csv'
test_df = pd.read_csv(test_path)
subject_entity = test_df[test_df['subject_entity'].apply(lambda x: ast.literal_eval(x)['type'] == 'LOC')]
print(subject_entity['subject_entity'])
# %%

import pandas as pd
import ast
train_path = '../dataset/train/train.csv'
train_df = pd.read_csv(train_path)
train_df = train_df[train_df['subject_entity'].apply(lambda x: ast.literal_eval(x)['type'] == 'PER') & train_df['object_entity'].apply(lambda x: ast.literal_eval(x)['type'] == 'ORG')]
train_df.to_csv('../dataset/train/per:org.csv', index=False)
print(train_df)

# %%
import pandas as pd

dict = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}

per_poh_path = '../dataset/train/per:poh.csv'
per_poh_df = pd.read_csv(per_poh_path)
per_poh_df = per_poh_df['label'].value_counts().reset_index()
per_poh_df.columns = ['relation', 'count']
per_poh_df['ids'] = per_poh_df['relation'].map(dict)
print("per_poh_df: \n", per_poh_df)
print()

per_per_path = '../dataset/train/per:per.csv'
per_per_df = pd.read_csv(per_per_path)
per_per_df = per_per_df['label'].value_counts().reset_index()
per_per_df.columns = ['relation', 'count']
per_per_df['ids'] = per_per_df['relation'].map(dict)
print("per_per_df: \n", per_per_df)
print()

per_org = '../dataset/train/per:org.csv'
per_org = pd.read_csv(per_org)
per_org = per_org['label'].value_counts().reset_index()
per_org.columns = ['relation', 'count']
per_org['ids'] = per_org['relation'].map(dict)
print("per_org: \n", per_org)
print()

per_noh_path = '../dataset/train/per:noh.csv'
per_noh_df = pd.read_csv(per_noh_path)
per_noh_df = per_noh_df['label'].value_counts().reset_index()
per_noh_df.columns = ['relation', 'count']
per_noh_df['ids'] = per_noh_df['relation'].map(dict)
print("per_noh_df: \n", per_noh_df)
print()

per_loc_path = '../dataset/train/per:loc.csv'
per_loc_df = pd.read_csv(per_loc_path)
per_loc_df = per_loc_df['label'].value_counts().reset_index()
per_loc_df.columns = ['relation', 'count']
per_loc_df['ids'] = per_loc_df['relation'].map(dict)
print("per_loc: \n", per_loc_df)
print()

per_dat_path = '../dataset/train/per:dat.csv'
per_dat_df = pd.read_csv(per_dat_path)
per_dat_df = per_dat_df['label'].value_counts().reset_index()
per_dat_df.columns = ['relation', 'count']
per_dat_df['ids'] = per_dat_df['relation'].map(dict)
print("per_dat_df: \n", per_dat_df)
print()

org_poh = '../dataset/train/org:poh.csv'
org_poh = pd.read_csv(org_poh)
org_poh = org_poh['label'].value_counts().reset_index()
org_poh.columns = ['relation', 'count']
org_poh['ids'] = org_poh['relation'].map(dict)
print("org_poh: \n", org_poh)
print()

org_per = '../dataset/train/org:per.csv'
org_per = pd.read_csv(org_per)
org_per = org_per['label'].value_counts().reset_index()
org_per.columns = ['relation', 'count']
org_per['ids'] = org_per['relation'].map(dict)
print("org_per: \n", org_per)
print()

org_org = '../dataset/train/org:org.csv'
org_org = pd.read_csv(org_org)
org_org = org_org['label'].value_counts().reset_index()
org_org.columns = ['relation', 'count']
org_org['ids'] = org_org['relation'].map(dict)
print("org_org: \n", org_org)
print()

org_noh = '../dataset/train/org:noh.csv'
org_noh = pd.read_csv(org_noh)
org_noh = org_noh['label'].value_counts().reset_index()
org_noh.columns = ['relation', 'count']
org_noh['ids'] = org_noh['relation'].map(dict)
print("org_noh: \n", org_noh)
print()

org_loc = '../dataset/train/org:loc.csv'
org_loc = pd.read_csv(org_loc)
org_loc = org_loc['label'].value_counts().reset_index()
org_loc.columns = ['relation', 'count']
org_loc['ids'] = org_loc['relation'].map(dict)
print("org_loc: \n", org_loc)
print()

org_dat = '../dataset/train/org:dat.csv'
org_dat = pd.read_csv(org_dat)
org_dat = org_dat['label'].value_counts().reset_index()
org_dat.columns = ['relation', 'count']
org_dat['ids'] = org_dat['relation'].map(dict)
print("org_dat: \n", org_dat)
print()
# %%

import pandas as pd
import json
import ast

test_path = '../dataset/test/test_data.csv'
test_df = pd.read_csv(test_path)

aug_path = '../dataset/train/augment_2.csv'
aug_df = pd.read_csv(aug_path)

aug_df['sub_type'] = test_df['subject_entity'].apply(lambda x: ast.literal_eval(x)['type'])
aug_df['obj_type'] = test_df['object_entity'].apply(lambda x: ast.literal_eval(x)['type'])
aug_df.drop(['probs'], axis=1, inplace=True)

#aug_df.to_csv('../dataset/train/test_analysis.csv', index=False)

aug_df = aug_df[aug_df['pred_label'] != 'no_relation']
count_mismatch = aug_df[aug_df['pred_label'].apply(lambda x: x.split(':')[0] != aug_df['sub_type'])]
result_df = pd.DataFrame({'Count_Mismatch': [count_mismatch]})
print(result_df)

# %%
import pandas as pd

augment_2 = pd.read_csv('../dataset/train/augment_2.csv')
solo_sota = pd.read_csv('../dataset/train/solo_sota.csv')
ensemble_sota = pd.read_csv('../dataset/train/ensemble_sota.csv')

augment_2.drop(['probs'], axis=1, inplace=True)
solo_sota.drop(['probs'], axis=1, inplace=True)
ensemble_sota.drop(['probs'], axis=1, inplace=True)

new_df = pd.DataFrame()
new_df['id'] = augment_2['id']
new_df['aug'] = augment_2['pred_label']
new_df['solo_sota'] = solo_sota['pred_label']
new_df['ensemble_sota'] = ensemble_sota['pred_label']
new_df.to_csv('../dataset/train/pred_analysis.csv', index=False)

# %%
import pandas as pd

pred_data = pd.read_csv('../dataset/train/pred_analysis.csv')
dif_data = pred_data[(pred_data['aug'] != pred_data['ensemble_sota'])]
dif_data.drop(['solo_sota'], axis=1, inplace=True)
print(dif_data)
dif_data.to_csv('../dataset/train/dif_data.csv', index=False)
# %%
import pandas as pd

aug = pd.DataFrame()
aug = dif_data['aug'].value_counts()
print("aug: \n", aug)
print()

# solo_sota = pd.DataFrame()
# solo_sota = dif_data['solo_sota'].value_counts()
# print("solo_sota: \n", solo_sota)
# print()

ensemble_sota = pd.DataFrame()
ensemble_sota = dif_data['ensemble_sota'].value_counts()
print("ensemble_sota: \n", ensemble_sota)
print()
# %%
import pandas as pd

train = '../dataset/train/train.csv'
train = pd.read_csv(train)
trans = '../dataset/train/no_relation_df.csv'
trans = pd.read_csv(trans)

augmented_df = pd.merge(trans, train, how='outer')
augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
augmented_df['id'] = range(1, len(augmented_df)+1)
augmented_df.reset_index(drop=True, inplace=True)
augmented_df.to_csv('../dataset/train/augmented_train3.csv', index=False)

# %%
import pandas as pd
import ast

trans = '../dataset/train/trans_train.csv'
trans_train = pd.read_csv(trans)

def filter_row(row):
    return row['subject_entity']['word'] in row['sentence'] and row['object_entity']['word'] in row['sentence']

print('원래 길이', len(trans_train))

# sentence에 subject entity와 object entity가 없으면 제거
trans_train['subject_entity'] = trans_train['subject_entity'].apply(ast.literal_eval)
trans_train['object_entity'] = trans_train['object_entity'].apply(ast.literal_eval)
filtered_trans1 = trans_train[trans_train.apply(filter_row, axis=1)]
print('sentence에 entity들 있는 거만 남기면', len(filtered_trans1))


# 분포를 맞춰주기 위해 no_relation, org:top_members/employees, per:employee_of을 제거
#filtered_trans2 = filtered_trans1[(filtered_trans1['label'] != "no_relation") & (trans_train['label'] != "org:top_members/employees") & (trans_train['label'] != "per:employee_of")]
#print('filter 후 길이', len(filtered_trans2))

# subject_entity와 object_entity의 start_idx 및 end_idx 수정해준다.
filtered_trans1['subject_entity'] = filtered_trans1.apply(lambda row: {
    'word': row['subject_entity']['word'],
    'start_idx': row['sentence'].find(row['subject_entity']['word']),
    'end_idx': row['sentence'].find(row['subject_entity']['word']) + len(row['subject_entity']['word'])-1,
    'type': row['subject_entity']['type']
}, axis=1)

filtered_trans1['object_entity'] = filtered_trans1.apply(lambda row: {
    'word': row['object_entity']['word'],
    'start_idx': row['sentence'].find(row['object_entity']['word']),
    'end_idx': row['sentence'].find(row['object_entity']['word']) + len(row['object_entity']['word'])-1,
    'type': row['object_entity']['type']
}, axis=1)

filtered_trans1['id'] = range(1, len(filtered_trans1)+1)
filtered_trans1.reset_index(drop=True, inplace=True)
filtered_trans1.to_csv('../dataset/train/filtered_trans1.csv', index=False)
# %%
filtered = pd.read_csv('../dataset/train/filtered_trans1.csv')
filtered_residence = filtered[filtered['label'] == 'per:place_of_residence']