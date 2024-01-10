#%%
import pandas as pd
import numpy as np
from googletrans import Translator
from tqdm.auto import tqdm
import time
import ast

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

# 번역된 문장에서 subject에 entity가 없는 경우 제거, subject_entity와 object_entity idx sentence에 맞게 조정

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
# POH만 가지도록

filtered1_trans_train_path = '../dataset/train/filtered_trans1.csv'
filtered1_trans_train = pd.read_csv(filtered1_trans_train_path)

filtered1_trans_train['object_entity'] = filtered1_trans_train['object_entity'].apply(ast.literal_eval)
filtered2_trans = filtered1_trans_train[(filtered1_trans_train['object_entity'].apply(lambda x: x.get('type') == 'POH'))]
print("filtered2길이: ", len(filtered2_trans))

filtered2_trans['id'] = range(1, len(filtered2_trans)+1)
filtered2_trans.reset_index(drop=True, inplace=True)
filtered2_trans.to_csv('../dataset/train/filtered_trans2.csv', index=False)

# %%
# 번역된 데이터(전처리) + 원본 데이터

trans_train_path = '../dataset/train/filtered_trans2.csv'
trans_train = pd.read_csv(trans_train_path)

augmented_df = pd.merge(train_df, trans_train, how='outer')
augmented_df['id'] = range(1, len(augmented_df)+1)
augmented_df.reset_index(drop=True, inplace=True)
augmented_df.to_csv('../dataset/train/augmented_train2.csv', index=False)
# %%
