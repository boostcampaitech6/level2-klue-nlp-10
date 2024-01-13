import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Train/Test Split
class Spliter:
# -> tuple[pd.DataFrame, pd.DataFrame]
    def no_split(train_path:str):
        dataset = pd.read_csv(train_path)
        return dataset, dataset


    def validation_stratified_split(train_path:str, dev_ratio:float=0.1, stratify:str='new_label', random_state:int=42, shuffle:bool=True):
        """make validadtion set(considering label-entity type match)"""
        dataset = pd.read_csv(train_path)
        dataset['sub_type'] = dataset['subject_entity'].apply(lambda x :eval(x)['type'])
        dataset['obj_type'] = dataset['object_entity'].apply(lambda x :eval(x)['type'])
        dataset['type_match'] = dataset['sub_type'] + '-' + dataset['obj_type']

        dataset['new_label'] = dataset['label']
        dataset.loc[dataset['label'] == 'no_relation', 'new_label'] = "no_relation_" + dataset.loc[dataset['label'] == 'no_relation', 'type_match']

        train_idx, val_idx = train_test_split(dataset.index,
                                              test_size = dev_ratio, 
                                              stratify = dataset[stratify],
                                              random_state = random_state,
                                              shuffle = shuffle)

        train_dataset, val_dataset = dataset.iloc[train_idx, :6], dataset.iloc[val_idx, :6]
        train_dataset = train_dataset.reset_index(drop= True)
        val_dataset = val_dataset.reset_index(drop= True)
        return train_dataset, val_dataset

    
    def random_split(train_path:str, dev_ratio:float=0.1, random_state:int=42, shuffle:bool=True):
        """Random train test split"""
        dataset = pd.read_csv(train_path)
        # random split
        train_idx, val_idx = train_test_split(dataset.index, 
                                              test_size=dev_ratio, 
                                              random_state=random_state, 
                                              shuffle=shuffle)
        
        train_dataset, val_dataset = dataset.iloc[train_idx], dataset.iloc[val_idx]

        return train_dataset, val_dataset
    

    def stratified_split(train_path:str, dev_ratio:float=0.1, stratify:str='label', random_state:int=42, shuffle:bool=True):
        """Stratified train test split"""
        dataset = pd.read_csv(train_path)
        # stratified_split
        train_idx, val_idx = train_test_split(dataset.index,
                                              test_size=dev_ratio,
                                              stratify=list(dataset[stratify]),
                                              random_state=random_state,
                                              shuffle=shuffle)
        
        train_dataset, val_dataset = dataset.iloc[train_idx], dataset.iloc[val_idx]
        return train_dataset, val_dataset
    
    
    def custom_train_test_split(train_path:str):
        """직접 만든 train_test split 메서드, 메서드 명은 메서드의 특징을 알려주고, 출력 형식을 맞춰주면 좋음. (필요시 수정 가능)"""
        pass
    