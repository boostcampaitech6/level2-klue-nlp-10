import pandas as pd
from googletrans import Translator
from tqdm.auto import tqdm

train_path = './dataset/train/train.csv'
train_df = pd.read_csv(train_path)
translator = Translator()

def google_ko2en2ko(ko_text, translator):
    ko_en_result = translator.translate(ko_text, dest = 'en').text
    en_ko_result = translator.translate(ko_en_result, dest = 'ko').text
    return en_ko_result

sen_list = train_df['sentence'] 

for idx, sentence in enumerate(tqdm(sen_list)):
    result =  google_ko2en2ko(sentence, translator)
    train_df.loc[idx,'sentence'] = result

train_df.to_csv('./dataset/train/trans_train.csv', index=False)



# test_text = "나는지금코드를보고있는중이다."
# print(google_ko2en2ko(test_text, translator))

# for idx, sentence in enumerate(tqdm(sen1_list)):
#     result =  google_ko2en2ko(sentence, translator)
#     test.loc[idx,'sentence_1'] = result

# for idx, sentence in enumerate(tqdm(sen2_list)):
#     result =  google_ko2en2ko(sentence, translator)
#     test.loc[idx,'sentence_2'] = result