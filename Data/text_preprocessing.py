import re
import pandas as pd

df_raw = pd.read_csv("전처리 하려는 데이터의 저장경로.csv")  ## 전처리 하려는 텍스트가 들어있는 csv
# df_raw = pd.read_json("전처리 하려는 데이터의 저장경로.json")    ## 또는 json 파일

# 전처리 함수
def preprocess_soft(text):
    text = re.sub(r'http\S+', '[URL]', text)
    text = re.sub(r'@\w+', '@[USER]', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

processed = []
for t in df_raw['text']:
    processed.append(preprocess_soft(t))

df_raw['text'] = processed
df_raw.to_csv("전처리 한 데이터를 저장할 이름.csv")
# df_raw.to_json("전처리 한 데이터를 저장할 이름.json")
