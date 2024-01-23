<div align='center'>

# Lv.2 NLP KLUE 프로젝트 : 문장 내 개체간 관계 추출(RE)

</div>



## **개요**
> 진행 기간: 24년 1월 3일 ~ 24년 1월 18일

> 데이터셋: 
> - 학습 데이터셋 32,470개
> - 평가 데이터는 7,765개  
>
> 평가 데이터의 50%는 Public 점수 계산에 활용되어 실시간 리더보드에 표기가 되고, 남은 50%는 Private 결과 계산에 활용되었습니다.

부스트캠프AI Tech 6기의 9, 10, 11주 차 과정으로 NLP KLUE 대회입니다. 주제는 ‘문장 내 개체간 관계 추출’으로, 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 자연어처리 N21 태스크인 관계 추출(Relation Extraction, RE)을 진행했습니다. 문장과 문장 내 subject_entity와 object_entity가 주어졌을 때 두 entity의 관계를 주어진 30개의 라벨 중 하나로 분류하는 task를 수행했습니다. `no-relation` 라벨을 제외한 **micro-f1 score**와 **auprc**로 평가합니다.

## 리더보드 결과
<div align="left">
    <img src='https://img.shields.io/badge/RE_Public_LB-4️⃣-8A2BE2'></img> <img src='https://img.shields.io/badge/STS_Private_LB-3️⃣-8A2BE2'></img>
    <br>
    <img src='https://github.com/boostcampaitech6/level2-klue-nlp-10/blob/main/img/leaderboard.png?raw=true'></img>
</div>

## Fin-GPT
<div align='center'>

|권예진 [<img src="img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/Becky-Kwon)|문지원 [<img src="img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/jwmooon)|방제형 [<img src="img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/BJH9)|이경재 [<img src="img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/EbanLee))|이종원 [<img src="img/github-mark.png" width="20" style="vertical-align:middle;">](https://github.com/jongwoncode)|
|:-:|:-:|:-:|:-:|:-:|
|<img src='img/yejin.jpg' height=160 width=125></img>|<img src='img/jwmoon.jpg' height=160 width=125></img>|<img src='img/방제형.png' height=160 width=125></img>|<img src='img/KakaoTalk_20240103_170830055짜른거.jpg' height=160 width=125></img>|<img src='img/jongwon.jpg' height=160 width=125></img>|

### Learning Rate Scheduler 사용 및 Padding 추가 구현
일정한 LR 사용 시 학습 초기에는 모델의 수렴 속도가 늦춰지고 학습 후기에는 최적해 근처에서 진동하는 문제가 있습니다. 이를 보완하기 위해서 LR Scheduler를 이용하여 학습 초기에는 큰 값의 LR을 부여하여 수렴을 빠르게 하고, 이를 점차 감소시키며 좀 더 정밀하게 최적해를 찾게 했습니다. 이를 구현하기 위해 transformer 라이브러리의 ‘get_linear_schedule_with_ warmup’을 사용했습니다.  
또한 collate_fn을 통해 배치 단위로 padding을 추가했습니다. 일부 사전 학습된 모델의 경우 입력 길이에 차이가 있으면 에러가 발생했습니다. 이를 해결하고자 transformers의 ‘DataCollatorWithPadding’을 사용하였고 덕분에 보다 보편적인 실험 환경을 갖출 수 있었습니다.

## Model 선택 및 앙상블
- 이번 RE 테스크는 KLUE 데이터셋을 활용하여 진행했기에 klue 데이터로 fine-tunning된 모델에서 좋은 성능을 보였습니다. 특히, klue/RoBERTa-large가 다른 모델에 비해 높은 성능을 보였습니다.
- 추가로 저희 팀에서는 klue/RoBERTa-base, klue/RoBERTa-large, klue/bert-base, monologg/KoELECTRA-base-v3-discriminator, vaiv/kobigbird-roberta-large, team-lucid/deberta-v3-base-korean, wooy0ng/korquad1-klue-roberta-large, kakaobank/kf-deberta-base 모델들을 실험했습니다.
- 


## 앙상블

<div align='center'>
|  | 모델 | 데이터 | 비율 |
|:---:|:---|:---|:---:|
| **A** | snunlp/KR-ELECTRA-discriminator | train+dev (shuffle 8:2) | 0.3 |
| **B** | snunlp/KR-ELECTRA-discriminator (stacking) | train + swap data | 0.2 |
| **C** | beomi/KcELECTRA-base | train + swap data | 0.2 |
| **D** | team-lucid/deberta-v3-xlarge-korean | train data | 0.3 |
| **E** | xlm-roberta-large (PERSON 토큰 추가) | train data | 0.1 |

</div>

## 최종결과

> **총 제출 횟수: 56**

<div align='center'>

| 순위 | 분류 | 점수(Pearson Correlation) |
|:---:| --- |:---:|
| 🥇 | Public Score (대회 진행) | 0.9374 |
| 🥇 | Private Score (최종) | 0.9428 |

</div>

