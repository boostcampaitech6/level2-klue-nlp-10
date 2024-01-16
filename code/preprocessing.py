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
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
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
        print(f'Added New Token Cnt : {len(new_token)} List : {new_token}')
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
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
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
        tokenizer.add_special_tokens(special_tokens_dict)

        return change_sentences, tokenizer
    

    def typed_entity_marker(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : <S:PERSON> 이순신 </S:PERSON>은 조선의 <O:JOB> 무신 </O:JOB>이다.
        Qestion : <S:PERSON> 이순신 </S:PERSON>은 조선의 <O:JOB> 무신 </O:JOB>이다. [SEP] <S:PERSON> 이순신 </S:PERSON> 와 <O:JOB> 무신 </O:JOB>의 관계는 무엇입니까?
        """
        new_token, change_sentences = set(['<O:LOC>', '</O:POH>', '</S:PER>', '<O:POH>', '</O:ORG>', '<O:NOH>', '<O:ORG>', '<O:PER>', '</O:PER>', '</S:LOC>', '</O:NOH>', '<S:ORG>', '</O:LOC>', '</S:ORG>', '<S:PER>', '<O:DAT>', '</O:DAT>', '<S:LOC>']), [] 

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - typed entity marker working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"<S:{sub_e_info['type'].upper()}> {sub_e_info['word']} </S:{sub_e_info['type'].upper()}>", f"<O:{obj_e_info['type'].upper()}> {obj_e_info['word']} </O:{obj_e_info['type'].upper()}>"
            sentence = dataset['sentence'][i]
            
            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        print(f'Added New Token Cnt : {len(new_token)} List : {new_token}')
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
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
        
        new_token, change_sentences = set(['@', '$', '*', '^'] + list(mapper.values())), [] 

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
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        return change_sentences, tokenizer
    
    def typed_entity_marker_punctV2(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : @ * 사람 * 이순신 @은 조선의 # ^ 직업 ^ 무신 # 이다.
        Qestion : @ * 사람 * 이순신 @은 조선의 $ ^ 직업 ^ 무신 $ 이다. [SEP] @ * 사람 * 이순신 @ 와 $ ^ 직업 ^ 무신 $의 관계는 무엇입니까? 
        """
        # ORG(조직), PER(인물), DAT(날짜), LOC(지명), POH(기타), NOH(기타 수량 표현) 
        mapper = {'ORG' : '조직', 'PER' : '인물', 'DAT' : '날짜', 'LOC' : '지역', 'POH' : '기타', 'NOH' : '수량'}
        
        new_token, change_sentences = set(['@', '$', '*', '^']), []

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - typed entity marker punct working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"@ * {mapper[sub_e_info['type']]} * {sub_e_info['word']} @", f"$ ^ {mapper[obj_e_info['type']]} ^ {obj_e_info['word']} $"
            sentence = dataset['sentence'][i]

            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        # token 추가
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        return change_sentences, tokenizer
    

    def typed_entity_marker_punctV3(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : [E1] * 사람 * 이순신 [/E1]은 조선의 [E2] ^ 직업 ^ 무신 [/E2] 이다.
        Qestion : [E1] * 사람 * 이순신 [E1]은 조선의 [E2] ^ 직업 ^ 무신 [/E2] 이다. [SEP] [E1] * 사람 * 이순신 [/E1] 와 [E2] ^ 직업 ^ 무신 [/E2]의 관계는 무엇입니까? 
        """
        # ORG(조직), PER(인물), DAT(날짜), LOC(지명), POH(기타), NOH(기타 수량 표현) 
        mapper = {'ORG' : '조직', 'PER' : '인물', 'DAT' : '날짜', 'LOC' : '지역', 'POH' : '기타', 'NOH' : '수량'}
        
        new_token, change_sentences = set(['*', '^', '[E1]', '[/E1]', '[E2]', '[/E2]']), [] # list(mapper.values()))

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - typed entity marker punct working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"[E1] * {mapper[sub_e_info['type']]} * {sub_e_info['word']} [/E1]", f"[E2] ^ {mapper[obj_e_info['type']]} ^ {obj_e_info['word']} [/E2]"
            sentence = dataset['sentence'][i]

            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        # token 추가
        add_tokens = new_token - set(tokenizer.vocab.keys())
        print(f'Added New Token Cnt : {len(add_tokens)} List : {add_tokens}')
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
        tokenizer.add_special_tokens(special_tokens_dict)
        
        return change_sentences, tokenizer

    def typed_entity_marker_non_object_type(self, dataset:pd.DataFrame, tokenizer, add_question:bool=True, and_marker:str='와'):
        """
        Before  : 이순신은 조선의 무신이다.
        After   : <S:PERSON> 이순신 </S:PERSON>은 조선의 <O> 무신 </O>이다.
        Qestion : <S:PERSON> 이순신 </S:PERSON>은 조선의 <O> 무신 </O>이다. [SEP] <S:PERSON> 이순신 </S:PERSON> 와 <O> 무신 </O>의 관계는 무엇입니까?
        """
        new_token, change_sentences = set(['<O>', '</O>', '<S:PER>', '</S:PER>', '<S:ORG>', '</S:ORG>', '<S:LOC>', '</S:LOC>']), [] 

        for i in tqdm(range(len(dataset)), desc = 'Preprocessor - typed entity marker working ...!!'):
            sub_e_info, obj_e_info = literal_eval(dataset['subject_entity'][i]), literal_eval(dataset['object_entity'][i])
            sub_e_marker, obj_e_marker =  f"<S:{sub_e_info['type'].upper()}> {sub_e_info['word']} </S:{sub_e_info['type'].upper()}>", f"<O> {obj_e_info['word']} </O>"
            sentence = dataset['sentence'][i]
            
            # new sentence 생성
            sentence = self.make_sentence(sentence, sub_e_info, sub_e_marker, obj_e_info, obj_e_marker, add_question, and_marker)
            change_sentences.append(sentence)

        
        print(f'Added New Token Cnt : {len(new_token)} List : {new_token}')
        special_tokens_dict = {'additional_special_tokens': sorted(list(new_token))}
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
        mapper = {'ORG' : '조직', 'PER' : '인물', 'DAT' : '날짜', 'LOC' : '지역', 'POH' : '기타', 'NOH' : '수량'}
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
            prompt = f"<S:{sub_type.upper()}> {sub_e} </S:{sub_type.upper()}> {and_marker} <O:{obj_type.upper()}> {obj_e} </O:{obj_type.upper()}>"

        elif marker == 'typed_entity_marker_punct':
            prompt = f"@ * {mapper[sub_type]} * {sub_e} @ {and_marker} # ^ {mapper[obj_type]} ^ {obj_e} #"
        
        elif marker == 'typed_entity_marker_punctV2':
            prompt = f"@ * {mapper[sub_type]} * {sub_e} @ {and_marker} $ ^ {mapper[obj_type]} ^ {obj_e} $"
        
        elif marker == 'typed_entity_marker_punctV3':
            prompt = f"[E1] * {mapper[sub_type]} * {sub_e} [/E1] {and_marker} [E2] ^ {mapper[obj_type]} ^ {obj_e} [/E2]"

        elif marker == 'typed_entity_marker_non_object_type':
            prompt = f"<S:{sub_type.upper()}> {sub_e} </S:{sub_type.upper()}> {and_marker} <O> {obj_e} </O>"
        else:
            raise Exception("Check prompt marker.. not in ['baseline_preprocessor', 'entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct', 'typed_entity_marker_punctV2', 'typed_entity_marker_punctV3', 'typed_entity_marker_non_object_type']")

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


def get_entity_loc(tokenizer, tokenized_sentences, config):
    """Matching The Blanks에 사용하는 entity marker 위치 정보를 생성하는 함수"""

    # 현재 Matching the blank를 지원하는 marker인지 확인
    assert config['preprocess_method'] in ['typed_entity_marker', 'entity_marker', 'typed_entity_marker_non_object_type', 'typed_entity_marker_punctV3'], "Matching the blank possible whth ['typed_entity_marker', 'entity_marker', 'typed_entity_marker_non_object_type', 'typed_entity_marker_punctV3']"

    entitiy_marker_loc_ids = []
    marker = { 'typed_entity_marker' : ['<O:POH>', '</O:LOC>', '</O:ORG>', '<S:ORG>', '</S:LOC>', '</O:NOH>', '</O:DAT>', '<S:PER>', '<S:LOC>', '</S:ORG>', '<O:LOC>', '<O:NOH>', '<O:ORG>', '</O:POH>', '<O:PER>', '</O:PER>', '<O:DAT>', '</S:PER>'],
                'typed_entity_marker_non_object_type' : ['<O>', '</O>', '<S:PER>', '</S:PER>', '<S:ORG>', '</S:ORG>', '<S:LOC>', '</S:LOC>'],
                'typed_entity_marker_punctV3' : ['[E1]', '[E2]', '[/E1]', '[/E2]'],
                'entity_marker' : ['[E1]', '[E2]', '[/E1]', '[/E2]']}

    # start marker와 end marker 분리 - entity 위치 정보에 다른 숫자로 표기할 것임.
    start_marker_list, end_marker_list, marker_list = [], [], marker[config['preprocess_method']] 
    for m in marker_list:
        if '/' in m:
            end_marker_list.append(tokenizer.convert_tokens_to_ids(m))
        else:
            start_marker_list.append(tokenizer.convert_tokens_to_ids(m))

    # token length 추출 
    TOKEN_LENGTH = len(tokenized_sentences['input_ids'][0])


    for tokenized_sentence in tqdm(tokenized_sentences['input_ids'], desc='Add Tokenizer Matching the blanks ids ...'):
        # entity marker 위치 정보를 담을 list 생성.
        entity_marker_loc = [0]*TOKEN_LENGTH
        
        # Prompt type에 따라서, sentence 내의 marker를 선별하기 위한 IDX 추출.
        IDX = 0 if config['only_sentence'] else 1
        
        tokens = tokenized_sentence.tolist()
        # [E1], <O:POH>와 같은 entity 앞에 붙는 marker는 2로 표시
        for m in start_marker_list:
            if m in tokens:
                start_idx = [idx for idx, token in enumerate(tokens) if token == m]
                entity_marker_loc[start_idx[IDX]] = 2
        
        # [/E1], </O:POH>와 같은 entity 뒤에 붙는 marker는 3로 표시
        for m in end_marker_list:
            if m in tokens:
                end_idx = [idx for idx, token in enumerate(tokens) if token == m]
                entity_marker_loc[end_idx[IDX]] = 3

        # [CLS] 토큰 기호 : 1
        entity_marker_loc[0] = 1
        entitiy_marker_loc_ids.append(entity_marker_loc)
    
    return entitiy_marker_loc_ids