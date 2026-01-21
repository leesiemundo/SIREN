# LLM annotation 이후 수작업으로 수정한 라벨을 원래 데이터셋에 적용하는 코드

import pandas as pd

df_set = pd.read_json('/content/학습용 최종 데이터셋.json')
df_revised = pd.read_csv('/content/라벨 오류 수동 수정.csv')

# 1. df_revised에서 'text'를 인덱스로, 'revised'를 값으로 하는 매핑 Series 생성
revised_map = df_revised.set_index('Text')['revised']

# 2. df_set에서 df_revised에 있는 'text' 값을 포함하는 행을 식별
 # = .isin()을 사용하여 'text' 칼럼 값이 revised_map의 인덱스(key)에 있는지 확인
matching_texts_mask = df_set['text'].isin(revised_map.index)

# 3. .loc[]을 사용하여 조건에 맞는 행('matching_texts_mask'가 True인 행)만 선택하고,'label_cssrs_stage' 칼럼을 업데이트
#  -> df_set['text']에 .map(revised_map)을 적용하여 일치하는 'text'에 대한 'revised' 값을 가져와 업데이트
df_set.loc[matching_texts_mask, 'label_cssrs_stage'] = df_set.loc[matching_texts_mask, 'text'].map(revised_map)

# 업데이트된 df_set의 상위 몇 개 행 출력 (확인용)
print("\n업데이트된 df_set의 상위 5개 행:")
print(df_set.head())
