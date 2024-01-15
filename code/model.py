import torch
import torch.nn as nn 
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel

class BaseModel(nn.Module):
    def __init__(self, model_name, label_cnt, tokenizer):
        super(BaseModel, self).__init__()
        model_config = AutoConfig.from_pretrained(model_name)
        model_config.num_labels = label_cnt
        self.model =  AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

        # 추가된 토큰을 반영해서 모델 embedding size 업데이트
        self.model.resize_token_embeddings(len(tokenizer)) 

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs



class MtbModel(nn.Module):
    def __init__(self, model_name, label_cnt, tokenizer, mtb_type='entity_start'):
        super(MtbModel, self).__init__()

        # Pretrain 모델 로드 및 파라미터 설정
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.num_labels = label_cnt
        self.model = AutoModel.from_pretrained(model_name, config=self.model_config)
        self.model.resize_token_embeddings(len(tokenizer))
        
        # mul : conat 이후 hidden size가 몇 배로 증가되는지 결정
        self.mtb_type = mtb_type
        mul = 3 if self.mtb_type == 'entity_start' else 5
        
        # classifier 정의
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), # 0.1
                                        # LayerNormalization
                                        nn.Linear(self.model_config.hidden_size * mul, self.model_config.hidden_size),
                                        nn.GELU(),
                                        nn.Dropout(p=0.2), # 제거
                                        nn.Linear(self.model_config.hidden_size, self.model_config.num_labels))



    def concat_target_tensor(self, mtb_loc_ids, last_hidden_state):
        """"output의 last hidden state 중 타겟 entity vector만 추출"""
        if self.mtb_type == 'entity_start':
            # CLS, [E1], [E2]
            idx_1 = (mtb_loc_ids == 1).nonzero(as_tuple=True)
            idx_2 = (mtb_loc_ids == 2).nonzero(as_tuple=True)
            outputs_1 = last_hidden_state[idx_1[0], idx_1[1], :]
            outputs_2 = last_hidden_state[idx_2[0], idx_2[1], :]
            
            output_mtb = torch.cat((outputs_1, outputs_2.view(-1, self.model_config.hidden_size * 2)), dim=-1)


        elif self.mtb_type == 'entity_start_end':
            # CLS, [E1], [/E1], [E2], [/E2]
            idx_1 = (mtb_loc_ids == 1).nonzero(as_tuple=True)
            idx_2 = (mtb_loc_ids == 2).nonzero(as_tuple=True)
            idx_3 = (mtb_loc_ids == 3).nonzero(as_tuple=True)
            outputs_1 = last_hidden_state[idx_1[0], idx_1[1], :]
            outputs_2 = last_hidden_state[idx_2[0], idx_2[1], :]
            outputs_3 = last_hidden_state[idx_3[0], idx_3[1], :]

            output_mtb = torch.cat((outputs_1, outputs_2.view(-1, self.model_config.hidden_size * 2), outputs_3.view(-1, self.model_config.hidden_size * 2)), dim=-1)
        
        return output_mtb
    

    def forward(self, **inputs):
        outputs = self.model(input_ids = inputs['input_ids'],
                             token_type_ids = inputs['token_type_ids'],
                             attention_mask = inputs['attention_mask'])


        outputs = self.concat_target_tensor(inputs['matching_the_blanks_ids'], outputs.last_hidden_state) # (B, L, H*mul)
        outputs = self.classifier(outputs)  # (B, 30)
        return {'logits' :outputs} 