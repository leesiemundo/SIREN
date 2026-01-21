import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoTokenizer, RobertaModel, BertModel

# =========================================================
# 1. 설정 및 클래스 정의
# =========================================================
class Config:
    """모델 설정을 위한 환경 변수"""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 128
    # 학습 때 사용한 모델명과 정확히 일치해야 합니다.
    MODEL_R_NAME = "klue/roberta-large"
    MODEL_K_NAME = "beomi/kcbert-large"
    ENSEMBLE_WEIGHTS = (0.6, 0.4)
    EPSILON = 1e-10
    SAVE_PATH = "./" # 모델 파일이 있는 경로

class SoftEnsembleModel(nn.Module):
    """
    저장된 가중치(pth 파일)를 불러오기 위한 모델 클래스
    """
    def __init__(self, roberta_name, kcbert_name):
        super(SoftEnsembleModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.kcbert = BertModel.from_pretrained(kcbert_name)
        self.classifier_r = nn.Linear(self.roberta.config.hidden_size, 2)
        self.classifier_k = nn.Linear(self.kcbert.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.w_r, self.w_k = Config.ENSEMBLE_WEIGHTS

    def forward(self, input_ids_r, mask_r, input_ids_k, mask_k):
        out_r = self.roberta(input_ids=input_ids_r, attention_mask=mask_r)
        logits_r = self.classifier_r(out_r.pooler_output)
        probs_r = self.softmax(logits_r)

        out_k = self.kcbert(input_ids=input_ids_k, attention_mask=mask_k)
        logits_k = self.classifier_k(out_k.pooler_output)
        probs_k = self.softmax(logits_k)

        ensemble_probs = self.w_r * probs_r + self.w_k * probs_k
        return torch.log(ensemble_probs + Config.EPSILON)

# =========================================================
# 2. 모델 로드 및 예측 함수
# =========================================================

def load_models():
    """저장된 3개의 모델 파일을 메모리로 로드하기"""
    # 파일 존재 여부 확인
    files = ["model_stage1.pth", "model_stage2.pth", "model_stage3.pth"]
    for f in files:
        if not os.path.exists(os.path.join(Config.SAVE_PATH, f)):
            print(f"❌ 오류: '{f}' 파일을 찾을 수 없습니다.")
            print("👉 왼쪽 파일 메뉴에 모델 파일 3개를 업로드했는지 확인해주세요.")
            return None

    print("🔄 모델 구조를 생성하고 가중치를 로드합니다... (시간이 조금 걸릴 수 있습니다)")

    # 모델 뼈대 생성
    m1 = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)
    m2 = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)
    m3 = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)

    # 가중치 로드
    try:
        m1.load_state_dict(torch.load(f"{Config.SAVE_PATH}model_stage1.pth", map_location=Config.DEVICE))
        m2.load_state_dict(torch.load(f"{Config.SAVE_PATH}model_stage2.pth", map_location=Config.DEVICE))
        m3.load_state_dict(torch.load(f"{Config.SAVE_PATH}model_stage3.pth", map_location=Config.DEVICE))

        m1.eval(); m2.eval(); m3.eval() # 평가 모드로 전환
        print("✅ 모델 로드 완료!")
        return (m1, m2, m3)
    except Exception as e:
        print(f"❌ 모델 로드 중 에러 발생: {e}")
        return None

def predict_sentence(text, models, tokenizers, thresholds):
    """단일 문장을 입력받아 최종 등급을 예측"""
    m1, m2, m3 = models
    tokenizer_r, tokenizer_k = tokenizers
    th1, th2, th3 = thresholds

    # 입력 데이터 전처리
    inputs_r = tokenizer_r(text, return_tensors='pt', max_length=Config.MAX_LEN, padding='max_length', truncation=True)
    inputs_k = tokenizer_k(text, return_tensors='pt', max_length=Config.MAX_LEN, padding='max_length', truncation=True)

    ids_r, mask_r = inputs_r['input_ids'].to(Config.DEVICE), inputs_r['attention_mask'].to(Config.DEVICE)
    ids_k, mask_k = inputs_k['input_ids'].to(Config.DEVICE), inputs_k['attention_mask'].to(Config.DEVICE)

    logs = {}

    with torch.no_grad():
        # Stage 1 (0 vs 1,2,3)
        prob_s1 = torch.exp(m1(ids_r, mask_r, ids_k, mask_k))[0][1].item()
        logs['S1_Risk'] = f"{prob_s1:.1%}"
        if prob_s1 < th1: return 0, logs

        # Stage 2 (1 vs 2,3)
        prob_s2 = torch.exp(m2(ids_r, mask_r, ids_k, mask_k))[0][1].item()
        logs['S2_HighRisk'] = f"{prob_s2:.1%}"
        if prob_s2 < th2: return 1, logs

        # Stage 3 (2 vs 3)
        prob_s3 = torch.exp(m3(ids_r, mask_r, ids_k, mask_k))[0][1].item()
        logs['S3_Severe'] = f"{prob_s3:.1%}"
        if prob_s3 < th3: return 2, logs
        else: return 3, logs

# =========================================================
# 3. 메인 실행 파트
# =========================================================
import sys

# 1. 등급별 설명
LEVEL_DESC = {
    0: "🟢 정상 (Level 0) - 위험 징후가 낮습니다.",
    1: "🟡 관심 (Level 1) - 죽음에 대한 소망이나 자살 사고가 드러나는 것으로 추정됩니다.",
    2: "🟠 주의 (Level 2) - 구체적인 자살 사고가 의심됩니다.",
    3: "🔴 심각 (Level 3) - 자살 계획이나 시도가 우려되는 고위험 상태입니다."
}

# 2. 모델 및 토크나이저 로드 확인
# 변수가 없거나 None인 경우 새로 로드합니다.
if 'models' not in locals() or models is None:
    models = load_models()
    if models is None:
        print("❌ 모델 로드 실패. 파일 업로드 여부를 확인하세요.")
        # sys.exit() 대신 루프 진입 방지
    else:
        # 토크나이저 변수명을 전역에서 사용 가능하도록 확실히 선언
        tokenizer_r = AutoTokenizer.from_pretrained(Config.MODEL_R_NAME)
        tokenizer_k = AutoTokenizer.from_pretrained(Config.MODEL_K_NAME)
        print("✅ 토크나이저 로드 완료.")

# 3. 임계값 설정
THRESHOLDS = (0.4, 0.7, 0.45)

# 모델이 정상적으로 로드된 경우에만 루프 시작
if 'models' in locals() and models is not None:
    print("\n" + "="*60)
    print("🤖 심리 상태 분석기 (AI Model)가 준비되었습니다.")
    print("💬 분석하고 싶은 문장을 입력하고 Enter를 누르세요.")
    print("❌ 종료하려면 'q' 또는 'quit'을 입력하세요.")
    print("="*60)

    while True:
        try:
            user_input = input("\n📝 입력: ")

            if user_input.lower() in ['q', 'quit', 'exit', '종료']:
                print("\n👋 프로그램을 종료합니다.")
                break

            if not user_input.strip():
                print("⚠️ 내용을 입력해주세요.")
                continue

            # 로드된 tokenizer_r, tokenizer_k 변수를 직접 전달
            pred_level, logs = predict_sentence(user_input, models, (tokenizer_r, tokenizer_k), THRESHOLDS)

            print(f"\n👉 분석 결과: {LEVEL_DESC[pred_level]}")
            print(f"📊 상세 확률:")
            print(f"   - 1단계(위험군 진입 확률): {logs.get('S1_Risk', '0%')}")

            if 'S2_HighRisk' in logs:
                print(f"   - 2단계(고위험군 진입 확률): {logs['S2_HighRisk']}")
            if 'S3_Severe' in logs:
                print(f"   - 3단계(심각단계 진입 확률): {logs['S3_Severe']}")

            print("-" * 40)

        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
else:
    print("⚠️ 모델이 준비되지 않아 프로그램을 실행할 수 없습니다.")
