#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

#%%
# path 및 df

train_path = '../dataset/train/train.csv'
train_df = pd.read_csv(train_path)

test_path = '../dataset/test/test_data.csv'
test_df = pd.read_csv(test_path)

# %%
# subj_entity의 라벨 종류 및 분포 / obj_entity의 라벨 종류 및 분포

def visualize_subj_label(train_df):
    train_df['subject_entity'] = train_df['subject_entity'].apply(eval)
    subj_label_df = pd.DataFrame(train_df['subject_entity'].apply(lambda x: x['type']).value_counts()).reset_index()
    subj_label_df.columns = ['subj_label', 'counts']
    print(subj_label_df)

    label = subj_label_df['subj_label']
    counts = subj_label_df['counts']

    plt.figure(figsize=(3, 3))
    plt.bar(label, counts, width=0.1, color='green')
    plt.xlabel('Label')
    plt.ylabel('Counts')
    plt.show()

def visualize_obj_label(train_df):
    train_df['object_entity'] = train_df['object_entity'].apply(eval)
    obj_label_df = pd.DataFrame(train_df['object_entity'].apply(lambda x: x['type']).value_counts()).reset_index()
    obj_label_df.columns = ['obj_label', 'counts']
    print(obj_label_df)

    label = obj_label_df['obj_label']
    counts = obj_label_df['counts']

    plt.figure(figsize=(12, 10))
    plt.bar(label, counts, width=0.1, color='red')
    plt.xlabel('Label')
    plt.ylabel('Counts')
    plt.show()


visualize_subj_label(train_df)
visualize_obj_label(train_df)

#%%

# test data에서 subj_entity의 라벨 종류 및 분포 / obj_entity의 라벨 종류 및 분포

def visualize_subj_label(train_df):
    train_df['subject_entity'] = train_df['subject_entity'].apply(eval)
    subj_label_df = pd.DataFrame(train_df['subject_entity'].apply(lambda x: x['type']).value_counts()).reset_index()
    subj_label_df.columns = ['subj_label', 'counts']
    print(subj_label_df)

    label = subj_label_df['subj_label']
    counts = subj_label_df['counts']

    plt.figure(figsize=(3, 3))
    plt.bar(label, counts, width=0.1, color='green')
    plt.xlabel('Label')
    plt.ylabel('Counts')
    plt.show()

def visualize_obj_label(train_df):
    train_df['object_entity'] = train_df['object_entity'].apply(eval)
    obj_label_df = pd.DataFrame(train_df['object_entity'].apply(lambda x: x['type']).value_counts()).reset_index()
    obj_label_df.columns = ['obj_label', 'counts']
    print(obj_label_df)

    label = obj_label_df['obj_label']
    counts = obj_label_df['counts']

    plt.figure(figsize=(12, 10))
    plt.bar(label, counts, width=0.1, color='red')
    plt.xlabel('Label')
    plt.ylabel('Counts')
    plt.show()


visualize_subj_label(test_df)
visualize_obj_label(test_df)

# %%
# relation 시각화

def visualize_relation(train_df):
    relation_df = pd.DataFrame(train_df['label'].value_counts()).reset_index()
    relation_df.columns = ['relation', 'counts']
    print(relation_df)

    relation = relation_df['relation']
    counts = relation_df['counts']

    plt.figure(figsize=(12, 10))
    plt.barh(relation, counts, color='red')
    plt.ylabel('Relation')
    plt.xlabel('Counts')
    plt.show()

visualize_relation(train_df)

# %%
# sentence 길이 시각화
# sentence 부분 가져오고, sentence 길이(10 단위), count

def visualize_sentence_len(train_df):
    # sentence 길이 및 개수 df 생성
    train_df['sentence_len'] = train_df['sentence'].apply(lambda x: (len(x) // 10) * 10)
    sentence_len_df = train_df['sentence_len'].value_counts().sort_index().reset_index()
    sentence_len_df.columns = ['sentence_len', 'counts']
    max_len = sentence_len_df['sentence_len'].iloc[-1]
    print(max_len)
    print(sentence_len_df)

    # sentence 길이 그래프 시각화
    plt.figure(figsize=(40, 20))
    sentence_len = sentence_len_df['sentence_len']
    counts = sentence_len_df['counts']
    plt.bar(sentence_len, counts, width=1, color='red')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('sentence_len', fontsize=40)
    plt.ylabel('Counts', fontsize=40)
    #plt.show()

visualize_sentence_len(train_df)

# %%
# outlier 구간 찾기

def find_outlier(train_df):
    max_len = len(train_df)
    print(max_len)

    ratio_25 = int(max_len * 0.25)
    ratio_50 = int(max_len * 0.5) 
    ratio_75 = int(max_len * 0.75) 
    ratio_97 = int(max_len * 0.97)
    ratio_98 = int(max_len * 0.98)
    ratio_99 = int(max_len * 0.99)
    print(ratio_97)
    print(ratio_98)
    print(ratio_99)

    train_df['sentence_len'] = train_df['sentence'].apply(lambda x: len(x))
    sentence_len_df = train_df['sentence_len'].sort_values()
    sentence_len_df = sentence_len_df.reset_index(drop=True)
    print(sentence_len_df)

    print(sentence_len_df[ratio_97])
    

find_outlier(train_df)

# %%

# 중복 검출
def detect_duplicated(train_df):
    sentence_dup = train_df[train_df['sentence'].duplicated(keep=False)].sort_values('subject_entity')
    print(len(sentence_dup))

    subject_entity_dup = train_df[train_df['subject_entity'].duplicated(keep=False)].sort_values('subject_entity')
    print(len(subject_entity_dup))

    object_entity_dup = train_df[train_df['object_entity'].duplicated(keep=False)].sort_values('subject_entity')
    print(len(object_entity_dup))

    train_df['subject_entity'] = tuple(train_df['subject_entity'])
    train_df['object_entity'] = tuple(train_df['object_entity'])
    dup = train_df[train_df[['sentence', 'subject_entity', 'object_entity']].duplicated(keep=False)].sort_values('subject_entity')
    print('길이', len(dup))
    print(dup['sentence'])

    print('라벨만 다른 경우')
    filtered_df = dup[dup[['sentence', 'subject_entity', 'object_entity']].duplicated(keep=False) & ~dup['label'].duplicated(keep=False)]
    print(filtered_df)


detect_duplicated(train_df)

# %%
