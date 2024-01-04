import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Train/Test Split
class Spliter:
# -> tuple[pd.DataFrame, pd.DataFrame]
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
    