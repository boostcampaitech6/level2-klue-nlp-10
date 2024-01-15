import torch
import torch.nn as nn 
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel
from copy import deepcopy

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


        elif self.mtb_type == 'entiy_start_end':
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
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
class RecentModel(nn.Module):
    def __init__(self, model_name, label_cnt, tokenizer, restrict_num):
        super(RecentModel, self).__init__()
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.model_config.num_labels = label_cnt
        self.model = AutoModel.from_pretrained(model_name, config=self.model_config)
        self.model.resize_token_embeddings(len(tokenizer))

        ids = tokenizer.convert_tokens_to_ids(['@', '#'])
        self.sub_ids, self.obj_ids = ids[0], ids[1]
        

        self.type_label = [[0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 21, 27, 29], # 'PER, POH'
            [0, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 21, 23, 24, 26, 27, 28, 29], # 'PER, LOC'
            [0, 1, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 21, 24, 26, 27], # 'PER, PER' 
            [0, 4, 6, 8, 10, 11, 12, 13, 14, 15, 17, 23, 24, 26, 27, 29], # 'PER, ORG'
            [0, 4, 6, 10, 11, 14, 15, 17, 21, 24, 25, 26, 27], # 'PER, DAT'
            [0, 4, 6, 10, 12, 15, 21, 24, 25], # 'PER, NOH'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, POH'
            [0, 1, 2, 3, 5, 7, 19, 20], # 'ORG, LOC'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, PER'
            [0, 1, 2, 3, 5, 7, 19, 20, 28], # 'ORG, ORG'
            [0, 2, 5, 7, 18, 19, 20, 22], # 'ORG, DAT'
            [0, 1, 2, 3, 5, 7, 9, 20], # 'ORG, NOH'
            [0]] # LOC, DAT

        self.classifier = torch.nn.Sequential(
            self.model.pooler, 
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=self.model_config.hidden_size, out_features=self.model_config.num_labels , bias=True)
        )

        self.special_classifier = torch.nn.ModuleList([deepcopy(self.classifier) for _ in range(2)])
        self.weight_parameter = torch.nn.Parameter(torch.tensor([[[0.5]], [[0.5]]]))
        
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, restrict_num=None, output_attentions=False):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions)
        special_outputs = outputs.last_hidden_state
                        
        batch_size = len(input_ids)
        
        special_idx = list()
        
        for i in range(batch_size):
            
            sub_start_idx = torch.nonzero(input_ids[i] == self.sub_ids)[0][0]
            obj_start_idx = torch.nonzero(input_ids[i] == self.obj_ids)[0][0]
            
            special_idx.append([sub_start_idx, obj_start_idx])
        
        pooled_output = [torch.stack([special_outputs[i, special_idx[i][j], :] for i in range(batch_size)]) for j in range(2)]
        
        logits = torch.stack([self.special_classifier[i](pooled_output[i].unsqueeze(1)) for i in range(2)], dim=0)
        logits = torch.sum(self.weight_parameter*logits, dim=0) 
        
        loss = None
        
        if labels is not None:
            loss_sum = 0
            for i in range(batch_size):
                weights = torch.full((30, ), 2.5).float().to(device)
                weights[self.type_label[restrict_num[i]]] = 1
                loss_fun = torch.nn.CrossEntropyLoss(weight=weights, reduction='none')
                loss_sum += loss_fun(logits[i].view(-1, self.model_config.num_labels), labels[i].view(-1))
            loss = loss_sum / batch_size
            loss = loss.squeeze()
        
        if output_attentions:    
            outputs = {"loss" : loss, "logits": logits, "attentions": outputs.attentions[0]}
        else:
            outputs = {"loss" : loss, "logits": logits}
        
        return outputs 