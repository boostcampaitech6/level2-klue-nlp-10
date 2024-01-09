import pandas as pd
from typing import List
from ast import literal_eval
from tqdm import tqdm


class Preprocessor:
    """dataset 내의 `sentence`를 전처리하는 class"""
    
    def make_sentence(self, sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker):
        """ 주어진 문장, entity, marker를 활용하여 새로운 sentence를 생성하는 메서드"""
        # Sentence : ... [object_entity] ... [subject_entity] ...
        if sub_e_info['end_idx'] > obj_e_info['end_idx']:
            sentence = sentence[:obj_e_info['start_idx']] + obj_e_marker + sentence[obj_e_info['end_idx']+1 :sub_e_info['start_idx']] + sub_e_marker + sentence[sub_e_info['end_idx']+1 :]
        
        # Sentence : ... [subject_entity] ... [object_entity]
        else:
            sentence = sentence[:sub_e_info['start_idx']] + sub_e_marker + sentence[sub_e_info['end_idx']+1 :obj_e_info['start_idx']] + obj_e_marker + sentence[obj_e_info['end_idx']+1 :]
            
        # 뒤에 붙는 Prompt : Question 
        if add_question:
            sentence += f" [SEP] {sub_e_marker} {and_marker} {obj_e_marker}의 관계는 무엇입니까?" 
        
        return sentence

    def baseline_preprocessor(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before   : 이순신은 조선의 무신이다.
        After    : 이순신은 조선의 무신이다.
        Question : 이순신은 조선의 무신이다. [SEP] 이순신 와 무신의 관계는 무엇입니까? 
        """
        change_sentences = []

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - no marker working ...!!'):
            sub_e, obj_e = literal_eval(dataset['subject_entity'][i])['word'], literal_eval(dataset['object_entity'][i])['word']
            sentence = dataset['sentence'][i]

            if add_question:
                sentence += f" [SEP] {sub_e} {and_marker} {obj_e}의 관계는 무엇입니까?"

            change_sentences.append(sentence)    
        return change_sentences, tokenizer
    

    def entity_mask(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before   : 이순신은 조선의 무신이다.
        After    : [SUB-PER]은 조선의 [OBJ-JOB] 이다.
        Question : [SUB-PER]은 조선의 [OBJ-JOB] 이다. [SEP] [SUB-PER] 와 [OBJ-JOB]의 관계는 무엇입니까?  
        """
        new_token, change_sentences = set(), [] 

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - entity mask working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_mask, obj_e_mask = f"[SUBJ-{sub_e_info['type'].upper()}]", f"[OBJ-{obj_e_info['type'].upper()}]"
            sentence = dataset['sentence'][i]
            
            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_mask, obj_e_info, obj_e_mask, add_question, and_marker)
            change_sentences.append(sentence)

            # 신규 토큰 추가.
            new_token.add(sub_e_mask)
            new_token.add(obj_e_mask)
        
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': list(add_tokens)}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        return change_sentences, tokenizer
    

    def entity_marker(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before : 이순신은 조선의 무신이다. 
        After  : [E1] 이순신 [/E1]은 조선의 [E2] 무신 [/E2]이다.
        Question : [E1] 이순신 [/E1]은 조선의 [E2] 무신 [/E2]이다. [SEP] [E1] 이순신 [/E1] 와 [E2] 무신 [/E2]의 관계는 무엇입니까?  
        """
        new_token, change_sentences = set(['[E1]', '[/E1]', '[E2]', '[/E2]']), []

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - entity marker working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"[E1] {sub_e_info['word']} [/E1]", f"[E2] {obj_e_info['word']} [/E2]"
            sentence = dataset['sentence'][i]

            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        # token 추가
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': list(add_tokens)}
        tokenizer.add_special_tokens(special_tokens_dict)

        return change_sentences, tokenizer
    

    def entity_marker_punct(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : @ 이순신 @ 은 조선의 # 무신 # 이다.
        Qestion : @ 이순신 @ 은 조선의 # 무신 # 이다. [SEP] @ 이순신 @ 와 # 무신 #의 관계는 무엇입니까? 
        """
        new_token, change_sentences = set(['@', '#']), []

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - entity marker punct working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"@ {sub_e_info['word']} @", f"# {obj_e_info['word']} #"
            sentence = dataset['sentence'][i]

            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        # token 추가
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': list(add_tokens)}
        tokenizer.add_special_tokens(special_tokens_dict)

        return change_sentences, tokenizer
    

    def typed_entity_marker(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : <S:PERSON> 이순신 </S:PERSON>은 조선의 <O:JOB> 무신 </O:JOB>이다.
        Qestion : <S:PERSON> 이순신 </S:PERSON>은 조선의 <O:JOB> 무신 </O:JOB>이다. [SEP] <S:PERSON> 이순신 </S:PERSON> 와 <O:JOB> 무신 </O:JOB>의 관계는 무엇입니까?
        """
        new_token, change_sentences = set(), [] 

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - typed entity marker working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"<S:{sub_e_info['type'].upper()}> {sub_e_info['word']} </S:{sub_e_info['type'].upper()}>", f"<O:{obj_e_info['type'].upper()}> {obj_e_info['word']} </O:{obj_e_info['type'].upper()}>"
            sentence = dataset['sentence'][i]
            
            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

            # 신규 토큰 추가.
            new_token = new_token | set([f"<S:{sub_e_info['type'].upper()}>", f"</S:{sub_e_info['type'].upper()}>", f"<O:{obj_e_info['type'].upper()}>", f"</O:{obj_e_info['type'].upper()}>"])
            
        
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': list(add_tokens)}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        return change_sentences, tokenizer


        
    def typed_entity_marker_punct(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : @ * 사람 * 이순신 @은 조선의 # ^ 직업 ^ 무신 # 이다.
        Qestion : @ * 사람 * 이순신 @은 조선의 # ^ 직업 ^ 무신 # 이다. [SEP] @ * 사람 * 이순신 @ 와 # ^ 직업 ^ 무신 #의 관계는 무엇입니까? 
        """
        # ORG(조직), PER(인물), DAT(날짜), LOC(지명), POH(기타), NOH(기타 수량 표현) 
        mapper = {'ORG' : '조직', 'PER' : '인물', 'DAT' : '날짜', 'LOC' : '지역', 'POH' : '기타', 'NOH' : 'noh'}
        
        new_token, change_sentences = set(['@', '#', '*', '^'] + list(mapper.values())), []

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - typed entity marker punct working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"@ * {mapper[sub_e_info['type']]} * {sub_e_info['word']} @", f"# ^ {mapper[obj_e_info['type']]} ^ {obj_e_info['word']} #"
            sentence = dataset['sentence'][i]

            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        # token 추가
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': list(add_tokens)}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        return change_sentences, tokenizer


class Prompt:
    """dataset 내의 `subject_entity`와 `object_entity`를 활용하여 prompt를 생성하는 class"""

    def marker(self, marker, and_marker, sub_e, obj_e, sub_type, obj_type):
        """
        sub_e    : 비틀즈
        obj_e    : 조지해리슨
        sub_type : ORG
        obj_type : PERSON
        """
        mapper = {'ORG' : '조직', 'PER' : '인물', 'DAT' : '날짜', 'LOC' : '지역', 'POH' : '기타', 'NOH' : 'noh'}
        prompt = ""
        if marker == 'baseline_preprocessor':
            prompt = f"{sub_e} {and_marker} {obj_e}"
        
        elif marker == 'entity_mask':
            prompt = f"[SUB-{sub_type.upper()}] {and_marker} [OBJ-{obj_type.upper()}]"
        
        elif marker == 'entity_marker':
            prompt = f"[E1] {sub_e} [/E1] {and_marker} [E2] {obj_e} [/E2]"
        
        elif marker == 'entity_marker_punct':
            prompt = f"@ {sub_e} @ {and_marker} # {obj_e} #"
        
        elif marker == 'typed_entity_marker':
            prompt = f"<S:{sub_type.upper()}> {sub_e} </S:{sub_type.upper()}> {and_marker} <O:{obj_type.upper()}> {obj_e['word']} </O:{obj_type.upper()}>"

        elif marker == 'typed_entity_marker_punct':
            prompt = f"@ * {mapper[sub_type]} * {sub_e} @ {and_marker} # ^ {mapper[obj_type]} ^ {obj_e} #"
        else:
            raise Exception("Check prompt marker.. not in ['baseline_preprocessor', 'entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct']")

        return prompt



    def make_prompt(self, dataset:pd.DataFrame, kind='s_sep_o', marker='baseline_preprocessor', and_marker='[SEP]'):
        """
        sub    : 비틀즈
        obj    : 조지해리슨
        prompt : 비틀즈 [SEP] 조지 해리슨
        """
        change_prompts = []
        for i in tqdm(range(len(dataset)), desc = f'Prompt - kind : {kind}, marker : {marker}, and_marker : {and_marker} working ...!!!'):
            sub_e, obj_e = literal_eval(dataset['subject_entity'][i])['word'], literal_eval(dataset['object_entity'][i])['word']
            sub_type, obj_type = literal_eval(dataset['subject_entity'][i])['type'], literal_eval(dataset['object_entity'][i])['type']

            prompt = self.marker(marker, and_marker, sub_e, obj_e, sub_type, obj_type)
            
            if kind == 's_sep_o':
                pass
            elif kind == 's_and_o':
                prompt = prompt + '의 관계'
            elif kind == 'quiz':
                prompt = '다음 문장에서 ' + prompt + '의 관계를 추출하시오.'
            else:
                raise Exception("check promt kind.. not in ['s_sep_o', 's_and_o', 'quiz']") 
            change_prompts.append(prompt)

        return change_prompts
    



def tokenized_dataset(tokenizer, prompt:List , sentence:List, max_length:int=256, only_sentence=False):
    """prompt와 sentence 입력시 tokenizer에 따라 sentence를 tokenizing 하는 메서드."""

    if only_sentence:
        """sentence 뒤에 add_question으로 prompt가 붙는 상황에서 앞에 붙는 prompt.question을 사용하지 않을 때. 
            ex)이순신은 조선의 무신이다. [SEP] 이순신 와 무신의 관계는 무엇입니까? 
        """
        tokenized_sentences = tokenizer(sentence,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length,
                                        add_special_tokens=True,
                                        )

    else:
        tokenized_sentences = tokenizer(prompt,
                                        sentence,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length,
                                        add_special_tokens=True,
                                        )
    return tokenized_sentences
