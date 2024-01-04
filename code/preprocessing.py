import pandas as pd
from typing import List

class Preprocessor:
    """dataset 내의 `sentence`를 전처리하는 class"""
    def baseline_preprocessor(dataset:pd.DataFrame):
        """baseline 코드에서 dataset의 sentence 라벨에는 전처리가 들어가 있지 않음."""
        return list(dataset['sentence'])

    def custom_preprocessor(dataset:pd.DataFrame):
        """sentence에 전처리를 적용하는 메서드."""
        pass



class Prompt:
    """dataset 내의 `subject_entity`와 `object_entity`를 활용하여 prompt를 생성하는 class"""
    def sub_sep_obj_prompt(dataset:pd.DataFrame):
        """baseline에 제공되는 prompt 부분. (현재 전처리와 같이 포함되어 있어서 논의 후 리팩토링)
            sub : '비틀즈'
            obj : '조지해리슨'
            prompt : '비틀즈'[SEP] '조지 해리슨'
            
            - 기본 코드에 공백 제거, 따옴표 제거등의 전처리가 추가적으로 필요함.

        """
        subject_entity_list, object_entity_list, prompt_list = [], [], []

        # entity 출력
        for sub, obj in zip(dataset['subject_entity'], dataset['object_entity']):
            sub = sub[1:-1].split(',')[0].split(':')[1]
            obj = obj[1:-1].split(',')[0].split(':')[1]
            subject_entity_list.append(sub) 
            object_entity_list.append(obj)
        
        # Prompt 생성
        for sub, obj in zip(subject_entity_list, object_entity_list):
            prompt = ''
            prompt = sub + '[SEP]' + obj
            prompt_list.append(prompt)

        return prompt_list


    def sub_and_obj_prompt(dataset:pd.DataFrame):
        """ (예시)
            sub : '비틀즈'
            obj : '조지해리슨'
            prompt : 비틀즈와 조지해리슨의 관계
        """
        pass

    def quiz_with_punct_marker(dataset:pd.DataFrame):
        """ (예시)
            sub : '비틀즈'
            obj : '조지해리슨'
            prompt : @ * person * 비틀즈 @ 와 @ * person * 조지해리슨 @ 의 관계를 추출하시오.
        """
        pass



def tokenized_dataset(tokenizer, prompt:List , sentence:List, max_length:int=256):
    """prompt와 sentence 입력시 tokenizer에 따라 sentence를 tokenizing 하는 메서드."""

    tokenized_sentences = tokenizer(prompt,
                                    sentence,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=max_length,
                                    add_special_tokens=True,
                                    )
    return tokenized_sentences
