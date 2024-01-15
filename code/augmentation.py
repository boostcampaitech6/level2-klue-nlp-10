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

train = pd.read_csv('../dataset/train/splitted_train.csv')
val = pd.read_csv('../dataset/test/splitted_val.csv')

train.drop(train.columns[0], axis=1, inplace=True)
train['id'] = range(1, len(train)+1)
train.reset_index(drop=True, inplace=True)
train.to_csv('../dataset/train/train_split.csv', index=False)

val.drop(val.columns[0], axis=1, inplace=True)
val['id'] = range(1, len(val)+1)
val.reset_index(drop=True, inplace=True)
val.to_csv('../dataset/train/val_split.csv', index=False)
# %%
import pandas as pd

val = pd.read_csv('../dataset/train/val_split.csv')
val['label'] = 100
val.to_csv("../dataset/test/val_test.csv")
print(val)


# %%
import pandas as pd

val = pd.read_csv('./prediction/val_test.csv') # 예측값
test = pd.read_csv('../dataset/train/val_split.csv') # 실제값

dif = test[val['pred_label'] != test['label']]
dif = dif['label'].value_counts()
print("예측 못한 값: \n", dif)

dif2 = val[val['pred_label'] != test['label']]
dif2 = dif2['pred_label'].value_counts()
print("\n\n~라고 틀리게 예측한 값: \n", dif2)

#
# A가 맞는데 B로 해석한 경우 시각화 코드 작성해야 함


# %%
import pandas as pd

val = pd.read_csv('./prediction/val_test.csv') # 예측값
test = pd.read_csv('../dataset/train/val_split.csv') # 실제값

val['real'] = test['label']
val = val.drop('probs', axis=1)
val.to_csv('../dataset/train/val_test.csv')
print(val)
# %%
import pandas as pd

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'no_relation') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("no_relation을 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:top_members/employees') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:top_members/employees 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:members') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:members 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:product') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:product 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:title') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:title 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:alternate_names') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:alternate_names 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:employee_of') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:employee_of 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:place_of_headquarters') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:place_of_headquarters 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:product') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:product 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:number_of_employees/members') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:number_of_employees/members 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:children') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:children 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:place_of_residence') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:place_of_residence 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:alternate_names') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:alternate_names 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:other_family') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:other_family 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:colleagues') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:colleagues 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:origin') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:origin 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:siblings') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:siblings 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:spouse') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:spouse 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:founded') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:founded 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:political/religious_affiliation') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:political/religious_affiliation 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:member_of') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:member_of 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:parents') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:parents 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:dissolved') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:dissolved 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:schools_attended') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:schools_attended 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:date_of_death') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:date_of_death 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:place_of_birth') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:place_of_birth 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:date_of_birth') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:date_of_birth 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:place_of_death') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:place_of_death 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'org:founded_by') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("org:founded_by 다른 걸로 잘못 예측: \n", dif)
print('\n\n')

dif = pd.read_csv('../dataset/train/val_test.csv') # 예측값, 실제값
dif = dif[(dif['real'] == 'per:religion') & (dif['real'] != dif['pred_label'])]
dif = dif['pred_label'].value_counts()
print("per:religion 다른 걸로 잘못 예측: \n", dif)
print('\n\n')
# %%
import pandas as pd

train = pd.read_csv('../dataset/train/train.csv')
trans = pd.read_csv('../dataset/train/filtered_trans1.csv')
trans = trans[(trans['label'] == 'org:product') | (trans['label'] == 'org:alternate_names') | (trans['label'] == 'per:place_of_residence') | (trans['label'] == 'per:origin')]

augmented_df = pd.merge(train, trans, how='outer')
augmented_df['id'] = range(1, len(augmented_df)+1)
# augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
augmented_df.to_csv('../dataset/train/augmented_train4.csv', index=False)
# %%

import pandas as pd

train = pd.read_csv('../dataset/train/train.csv')
trans = pd.read_csv('../dataset/train/filtered_trans1.csv')
trans = trans[(trans['label'] == 'per:place_of_residence')]

augmented_df = pd.merge(train, trans, how='outer')
augmented_df['id'] = range(1, len(augmented_df)+1)
# augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)
augmented_df.to_csv('../dataset/train/augmented_train5.csv', index=False)
# %%
import pandas as pd
import ast
test_path = '../dataset/test/test_data.csv'
test_df = pd.read_csv(test_path)
subject_entity = test_df[test_df['subject_entity'].apply(lambda x: ast.literal_eval(x)['type'] == 'LOC')]
print(subject_entity['object_entity'])
# %%
import pandas as pd

dict = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}

per_poh_path = '../dataset/train/per_poh.csv'
per_poh_df = pd.read_csv(per_poh_path)
per_poh_df = per_poh_df['label'].value_counts().reset_index()
per_poh_df.columns = ['relation', 'count']
per_poh_df['ids'] = per_poh_df['relation'].map(dict)
per_poh_df = per_poh_df.sort_values(by='ids')
print("per_poh_df: \n", per_poh_df)
print()

per_per_path = '../dataset/train/per_per.csv'
per_per_df = pd.read_csv(per_per_path)
per_per_df = per_per_df['label'].value_counts().reset_index()
per_per_df.columns = ['relation', 'count']
per_per_df['ids'] = per_per_df['relation'].map(dict)
per_per_df = per_per_df.sort_values(by='ids')
print("per_per_df: \n", per_per_df)
print()

per_org = '../dataset/train/per_org.csv'
per_org = pd.read_csv(per_org)
per_org = per_org['label'].value_counts().reset_index()
per_org.columns = ['relation', 'count']
per_org['ids'] = per_org['relation'].map(dict)
per_org = per_org.sort_values(by='ids')
print("per_org: \n", per_org)
print()

per_noh_path = '../dataset/train/per_noh.csv'
per_noh_df = pd.read_csv(per_noh_path)
per_noh_df = per_noh_df['label'].value_counts().reset_index()
per_noh_df.columns = ['relation', 'count']
per_noh_df['ids'] = per_noh_df['relation'].map(dict)
per_noh_df = per_noh_df.sort_values(by='ids')
print("per_noh_df: \n", per_noh_df)
print()

per_loc_path = '../dataset/train/per_loc.csv'
per_loc_df = pd.read_csv(per_loc_path)
per_loc_df = per_loc_df['label'].value_counts().reset_index()
per_loc_df.columns = ['relation', 'count']
per_loc_df['ids'] = per_loc_df['relation'].map(dict)
per_loc_df = per_loc_df.sort_values(by='ids')
print("per_loc: \n", per_loc_df)
print()

per_dat_path = '../dataset/train/per_dat.csv'
per_dat_df = pd.read_csv(per_dat_path)
per_dat_df = per_dat_df['label'].value_counts().reset_index()
per_dat_df.columns = ['relation', 'count']
per_dat_df['ids'] = per_dat_df['relation'].map(dict)
per_dat_df = per_dat_df.sort_values(by='ids')
print("per_dat_df: \n", per_dat_df)
print()

org_poh = '../dataset/train/org_poh.csv'
org_poh = pd.read_csv(org_poh)
org_poh = org_poh['label'].value_counts().reset_index()
org_poh.columns = ['relation', 'count']
org_poh['ids'] = org_poh['relation'].map(dict)
org_poh = org_poh.sort_values(by='ids')
print("org_poh: \n", org_poh)
print()

org_per = '../dataset/train/org_per.csv'
org_per = pd.read_csv(org_per)
org_per = org_per['label'].value_counts().reset_index()
org_per.columns = ['relation', 'count']
org_per['ids'] = org_per['relation'].map(dict)
org_per = org_per.sort_values(by='ids')
print("org_per: \n", org_per)
print()

org_org = '../dataset/train/org_org.csv'
org_org = pd.read_csv(org_org)
org_org = org_org['label'].value_counts().reset_index()
org_org.columns = ['relation', 'count']
org_org['ids'] = org_org['relation'].map(dict)
org_org = org_org.sort_values(by='ids')
print("org_org: \n", org_org)
print()

org_noh = '../dataset/train/org_noh.csv'
org_noh = pd.read_csv(org_noh)
org_noh = org_noh['label'].value_counts().reset_index()
org_noh.columns = ['relation', 'count']
org_noh['ids'] = org_noh['relation'].map(dict)
org_noh = org_noh.sort_values(by='ids')
print("org_noh: \n", org_noh)
print()

org_loc = '../dataset/train/org_loc.csv'
org_loc = pd.read_csv(org_loc)
org_loc = org_loc['label'].value_counts().reset_index()
org_loc.columns = ['relation', 'count']
org_loc['ids'] = org_loc['relation'].map(dict)
org_loc = org_loc.sort_values(by='ids')
print("org_loc: \n", org_loc)
print()

org_dat = '../dataset/train/org_dat.csv'
org_dat = pd.read_csv(org_dat)
org_dat = org_dat['label'].value_counts().reset_index()
org_dat.columns = ['relation', 'count']
org_dat['ids'] = org_dat['relation'].map(dict)
org_dat = org_dat.sort_values(by='ids')
print("org_dat: \n", org_dat)
print()
# %%
import pandas as pd

restrict = '../dataset/test/restrict_prediction.csv'
restrict = pd.read_csv(restrict)

sota = '../dataset/test/sota.csv'
sota = pd.read_csv(sota)

restrict = restrict.drop('probs', axis=1)
restrict['sota_label'] = sota['pred_label']
restrict = restrict[restrict['sota_label'] != restrict['pred_label']]
restrict.to_csv('../dataset/train/restrict_pred.csv', index=False)
print(restrict)
# %%
