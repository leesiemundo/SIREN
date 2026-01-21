import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from collections import Counter
import warnings

# Matplotlib 경고 무시
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------------------
# 0. 폰트 설치 및 설정 (Colab/Linux 환경용)
# ---------------------------------------------------------------------------------------

try:
    # 나눔 폰트 설치 명령어 실행 (Colab 환경에서 유용)
    !sudo apt-get install -y fonts-nanum
    !sudo fc-cache -fv
    # 폰트 경로 다시 로드
    import matplotlib.font_manager as fm
    fm._rebuild()
    
    # 설치된 나눔고딕 폰트 경로 확인 및 설정
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if font_path not in [f.fname for f in fm.fontManager.ttfonts]:
        # 시스템에 폰트가 설치되어 있어도 matplotlib에 등록되지 않은 경우를 대비
        fm.fontManager.addfont(font_path)

    plt.rcParams['font.family'] = 'NanumGothic'
    print("-> Nanum Gothic 폰트를 설치하고 설정했습니다. 런타임을 재시작해야 완전히 적용됩니다.")
    
except Exception as e:
    # 윈도우나 맥 환경에서는 기본 폰트로 설정 시도
    if 'Windows' in plt.get_backend():
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif 'Darwin' in plt.get_backend():
        plt.rcParams['font.family'] = 'AppleGothic'
    else:
        plt.rcParams['font.family'] = 'sans-serif'
    print(f"-> 폰트 설치 오류 발생: {e}. 기본 폰트 설정을 시도했습니다.")

# ---------------------------------------------------------------------------------------
# 1. 데이터 로딩 및 준비
# ---------------------------------------------------------------------------------------

file_name = "학습용 최종 데이터셋.json"
try:
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다. 파일 이름을 확인하거나 {file_name} 파일을 업로드해주세요.")
    exit()

df = pd.DataFrame(data)
df['label_cssrs_stage'] = df['label_cssrs_stage'].astype(int)
df['text_length'] = df['text'].apply(len)

# ---------------------------------------------------------------------------------------
# 2. 텍스트 길이 분포 분석 (생략 없이 다시 실행)
# ---------------------------------------------------------------------------------------

print("\n"+"="*60)
print("### 2. 텍스트 길이 분포 분석")
print("="*60)

# A. 전체 텍스트 길이 통계 (출력)
overall_stats = df['text_length'].agg(['min', 'max', 'mean', 'median']).round(2)
print("\n#### A. 전체 텍스트 길이 통계 (문자 수)")
print(overall_stats.to_markdown(numalign="left", stralign="left"))

# Visualize overall distribution (Histogram and Boxplot)
plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(df['text_length'], kde=True, bins=30)
plt.title('전체 텍스트 길이 분포 (히스토그램)', fontsize=12)
plt.xlabel('텍스트 길이 (문자 수)')
plt.ylabel('빈도')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=df['text_length'])
plt.title('전체 텍스트 길이 분포 (상자 그림)', fontsize=12)
plt.ylabel('텍스트 길이 (문자 수)')

plt.tight_layout()
plt.savefig('overall_text_length_distribution.png')
plt.close()
print("-> overall_text_length_distribution.png 파일이 저장되었습니다.")


# B. Stage-wise Text Length Comparison (출력)
stage_stats = df.groupby('label_cssrs_stage')['text_length'].agg(['count', 'mean', 'median', 'min', 'max']).round(2)
stage_stats.index = [f'Stage {i}' for i in stage_stats.index]
print("\n#### B. Stage별 텍스트 길이 통계")
print(stage_stats.to_markdown(numalign="left", stralign="left"))

# Visualize stage-wise comparison
plt.figure(figsize=(8, 6))
sns.boxplot(x='label_cssrs_stage', y='text_length', data=df)
plt.title('C-SSRS Stage별 텍스트 길이 분포 비교', fontsize=14)
plt.xlabel('C-SSRS 위험 단계')
plt.ylabel('텍스트 길이 (문자 수)')
plt.xticks(ticks=df['label_cssrs_stage'].unique(), labels=[f'Stage {i}' for i in sorted(df['label_cssrs_stage'].unique())])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('stage_text_length_comparison.png')
plt.close()
print("-> stage_text_length_comparison.png 파일이 저장되었습니다.")


# ---------------------------------------------------------------------------------------
# 3. 언어적 특징 분석 (키워드 및 빈도)
# ---------------------------------------------------------------------------------------

print("\n\n"+"="*60)
print("### 3. 언어적 특징 분석 (키워드 및 빈도)")
print("="*60)

# 3-1. 불용어 및 토큰화 정의 (EDA 결과에 사용된 불용어 리스트)
korean_stopwords = set([
    '이다', '있다', '하다', '되다', '이다', '것', '게', '제', '함', '나', '내', '너', '네', '고', '면', '는', '은',
    '이', '가', '을', '를', '도', '만', '으로', '와', '과', '에서', '에게', '한테', '좀', '막', '뭐', '다', '개', '걍',
    '진짜', '너무', '정말', '시발', '씨발', '진심', '아니', '존나', '있음', '함', '거', '듯', '임', '왜', '저', '수',
    '이런', '그럼', '이젠', '같음', '같아', '같다', '말', '또', '안', '때', '에', '임', '됨', '해', '해줘', '줄', '분', '님',
    '날', '건', '버', '못', '걍', '와', '과', '으로', '한테', '에서', '에게', '임', '됨', '때', '왜케', '시발련', '개새끼',
    '좆같', 'ㅅㅂ', 'ㅆㅂ', '개', '일이', '보세', '말씀', '하려', '해볼', '먹는', '먹은', '마시고', '먹고', '아님',
    '않는', '안함', '대체', '대해', '어디', '어떡해', '아니면', '보자', '있도록', '하는건', '할까', '있을', '없을',
    '해준', '거도', '가서', '가고', '말이', '학교', '오늘', '내일', '해서', '있는데', '없다', '있고', '하는', '싶어', 
    '죽고', '살고', '싶은', '싶지'
])

def tokenize_and_clean(text, stopwords):
    # 한글만 남기고 공백으로 치환
    text = re.sub(r'[^가-힣\s]', ' ', text)
    tokens = text.lower().split()
    cleaned_tokens = []
    for token in tokens:
        # 불용어 목록에 없고, 길이가 1보다 크고 10보다 작은 단어만 선택
        if token not in stopwords and len(token) > 1 and len(token) < 10:
            cleaned_tokens.append(token)
    return cleaned_tokens

# 3-2. 전체 데이터셋 상위 빈출 단어
all_tokens = [token for text in df['text'] for token in tokenize_and_clean(text, korean_stopwords)]
overall_word_counts = Counter(all_tokens)

print("\n#### A. 전체 데이터셋 상위 빈출 키워드 (Top 20)")
top_20_overall = pd.Series(overall_word_counts).sort_values(ascending=False).head(20)
print(top_20_overall.to_markdown(numalign="left", stralign="left"))


# 3-3. Stage별 핵심 키워드 비교 (Top 10)
stage_keywords = {}
for stage in [1, 2, 3]:
    stage_text = df[df['label_cssrs_stage'] == stage]['text']
    stage_tokens = [token for text in stage_text for token in tokenize_and_clean(text, korean_stopwords)]
    stage_word_counts = Counter(stage_tokens)
    stage_keywords[stage] = pd.Series(stage_word_counts).sort_values(ascending=False).head(10)

print("\n#### B. Stage별 상위 빈출 키워드 (Top 10)")
for stage, keywords in stage_keywords.items():
    print(f"**Stage {stage} Top 10 Keywords:**")
    print(keywords.to_markdown(numalign="left", stralign="left"))
    print("\n")


# 3-4. 은어/약물 관련 키워드 빈도 분석
slang_keywords = {
    '자살': '자살', '두부장사': '두부장사', 'ㄷㅂㅈㅅ': 'ㄷㅂㅈㅅ', '한강다이브': '한강다이브', '투신': '투신',
    '자해': '자해', '오디': '오디', 'od': 'od', '쿨드림': '쿨드림', '메지콘': '메지콘', '번개탄': '번개탄',
    '살자': '살자', 'ㅈㅅ': 'ㅈㅅ'
}

# 전체 빈도 계산 (출력)
slang_overall_counts = Counter()
for text in df['text'].astype(str):
    for keyword_key, keyword_str in slang_keywords.items():
        slang_overall_counts[keyword_key] += text.lower().count(keyword_str)

print("#### C. 주요 위험 은어/약물 키워드 전체 빈도")
slang_df_overall = pd.Series(slang_overall_counts).sort_values(ascending=False)
print(slang_df_overall.to_markdown(numalign="left", stralign="left"))


# Stage별 빈도 계산 (출력)
slang_stage_counts = {}
for stage in sorted(df['label_cssrs_stage'].unique()):
    stage_text = df[df['label_cssrs_stage'] == stage]['text'].astype(str)
    stage_counts = {}
    for keyword_key, keyword_str in slang_keywords.items():
        stage_counts[keyword_key] = stage_text.apply(lambda x: x.lower().count(keyword_str)).sum()
    slang_stage_counts[stage] = pd.Series(stage_counts)

slang_stage_df = pd.DataFrame(slang_stage_counts)
slang_stage_df.columns = [f'Stage {i}' for i in slang_stage_df.columns]
slang_stage_df = slang_stage_df[slang_stage_df.sum(axis=1) > 0] # 총합 0인 키워드 제거

print("\n#### D. 주요 위험 은어/약물 키워드 Stage별 빈도")
print(slang_stage_df.sort_values(by='Stage 3', ascending=False).to_markdown(numalign="left", stralign="left"))


# ---------------------------------------------------------------------------------------
# 4. 레이블링 이유(Reasoning) 분석
# ---------------------------------------------------------------------------------------

print("\n\n"+"="*60)
print("### 4. 레이블링 이유(Reasoning) 분석")
print("="*60)

# 4-1. Stage 0 제외 사유 분석
stage_0_rules = ['Rule C', 'Rule I', 'has_sarcasm', '허구', '타인', '제3자', '비유', '내러티브']
df_stage_0 = df[df['label_cssrs_stage'] == 0].copy()
total_stage_0 = len(df_stage_0)
stage_0_counts = Counter()

for text in df_stage_0['reasoning'].astype(str).str.lower():
    for rule in stage_0_rules:
        if rule.lower() in text:
            stage_0_counts[rule] += 1
            
# 관련 규칙 통합
consolidated_stage_0_counts = Counter()
consolidated_stage_0_counts['허구/내러티브 (Rule I)'] = stage_0_counts['rule i'] + stage_0_counts.get('허구', 0) + stage_0_counts.get('내러티브', 0)
consolidated_stage_0_counts['타인/제3자 관련 (Rule C)'] = stage_0_counts['rule c'] + stage_0_counts.get('타인', 0) + stage_0_counts.get('제3자', 0)
consolidated_stage_0_counts['희화화/비유 (Sarcasm)'] = stage_0_counts['has_sarcasm'] + stage_0_counts.get('비유', 0)

print(f"\n#### A. Stage 0 (비위험) 제외 사유 빈도 (총 {total_stage_0}개)")
stage_0_reasons_df = pd.Series(consolidated_stage_0_counts).sort_values(ascending=False)
print(stage_0_reasons_df.to_markdown(numalign="left", stralign="left"))
print("*참고: 하나의 텍스트에 여러 제외 사유가 중복으로 언급될 수 있습니다.")


# 4-2. Stage 1, 2, 3 C-SSRS 핵심 특징 분석
cssrs_feature_matrix = {}

for stage in [1, 2, 3]:
    df_stage = df[df['label_cssrs_stage'] == stage].copy()
    
    stage_features = {}
    
    if stage == 1:
        # Stage 1: thoughts/wishes
        stage_features = {
            'has_thoughts_of_suicide': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_thoughts_of_suicide' in x or '자살 사고' in x).sum(),
            'has_wish_of_death': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_wish_of_death' in x or '죽음의 소망' in x).sum(),
            'has_method (Stage 1)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_method' in x).sum()
        }
    elif stage == 2:
        # Stage 2: method/NSSI 
        stage_features = {
            'has_method (OD/NSSI)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_method' in x or '자해' in x or 'od' in x or '쿨드림' in x or '투신' in x).sum(),
            'has_intention (Stage 2)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_intention' in x).sum(),
            'Rule E (위험군 설명)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'rule e' in x or '트친소' in x or '위험군 설명' in x).sum(),
        }
    elif stage == 3:
        # Stage 3: intent/plan/action/inducement/imminent
        stage_features = {
            'has_intention (실행 의지)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_intention' in x or '실행 의지' in x or '할거야' in x).sum(),
            'has_plan (구체적 계획)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_plan' in x or '계획' in x or '날짜' in x).sum(),
            'has_action (시도/준비)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_action' in x or '시도 경험' in x or '행동을 완료' in x).sum(),
            'has_inducement (동반/권유)': df_stage['reasoning'].astype(str).str.lower().apply(lambda x: 'has_inducement' in x or '동반 자살' in x or '권유' in x).sum(),
            'is_imminent_rescue (긴급)': df_stage['is_imminent_rescue'].sum()
        }
    
    cssrs_feature_matrix[stage] = pd.Series(stage_features)

cssrs_df = pd.DataFrame(cssrs_feature_matrix).T
cssrs_df.index = [f'Stage {i}' for i in cssrs_df.index]
cssrs_df = cssrs_df.fillna(0).astype(int)

# 컬럼 정렬 및 정리 (결과 출력과 일치하도록)
all_cols = sorted(cssrs_df.columns)
cssrs_df = cssrs_df.reindex(columns=all_cols, fill_value=0)

print(f"\n#### B. Stage별 주요 C-SSRS 위험 특징 발생 건수 (근거 분석)")
print(cssrs_df.to_markdown(numalign="left", stralign="left"))
