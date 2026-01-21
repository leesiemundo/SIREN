import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaModel, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
import gc
import sys

# =========================================================
# 1. 설정 및 상수 관리
# =========================================================
class Config:
    """전역 설정 및 하이퍼파라미터를 관리하는 클래스"""
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    MAX_LEN = 128
    MODEL_R_NAME = "klue/roberta-large"
    MODEL_K_NAME = "beomi/kcbert-large"
    ENSEMBLE_WEIGHTS = (0.6, 0.4)
    EPSILON = 1e-10
    SAVE_PATH = "./"  # 모델 저장 위치

def set_seed(seed):
    """모든 난수 시드를 고정하여 재현성을 확보"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(Config.SEED)

# =========================================================
# 2. 데이터셋 클래스 정의
# =========================================================
class DualBertDataset(Dataset):
    """
    RoBERTa와 KcBERT 두 모델에 동시에 입력 데이터를 제공하기 위한 Dataset 클래스
    """
    def __init__(self, df, tokenizer_r, tokenizer_k, max_len=Config.MAX_LEN):
        self.texts = df['text'].tolist()
        if 'label' in df.columns:
            self.labels = df['label'].tolist()
        else:
            self.labels = df['label'].tolist()

        self.tokenizer_r = tokenizer_r
        self.tokenizer_k = tokenizer_k
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding_r = self.tokenizer_r(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        encoding_k = self.tokenizer_k(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids_r': encoding_r['input_ids'].squeeze(0),
            'attention_mask_r': encoding_r['attention_mask'].squeeze(0),
            'input_ids_k': encoding_k['input_ids'].squeeze(0),
            'attention_mask_k': encoding_k['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =========================================================
# 3. Soft Ensemble 모델 정의
# =========================================================
class SoftEnsembleModel(nn.Module):
    """
    두 개의 BERT 모델(RoBERTa, KcBERT)의 예측 확률을 가중 평균하여 앙상블하는 모델
    (수정: 명시적인 모델 클래스 로딩으로 변경)
    """
    def __init__(self, roberta_name, kcbert_name):
        super(SoftEnsembleModel, self).__init__()

        # 1. RoBERTa 모델 로드 (klue/roberta-large)
        self.roberta = RobertaModel.from_pretrained(roberta_name)

        # 2. KcBERT 모델 로드 (beomi/kcbert-large)
        self.kcbert = BertModel.from_pretrained(kcbert_name)

        # Binary Classification Heads
        self.classifier_r = nn.Linear(self.roberta.config.hidden_size, 2)
        self.classifier_k = nn.Linear(self.kcbert.config.hidden_size, 2)

        self.softmax = nn.Softmax(dim=1)
        self.w_r, self.w_k = Config.ENSEMBLE_WEIGHTS

    def forward(self, input_ids_r, mask_r, input_ids_k, mask_k):
        # RoBERTa Forward
        out_r = self.roberta(input_ids=input_ids_r, attention_mask=mask_r)
        logits_r = self.classifier_r(out_r.pooler_output)
        probs_r = self.softmax(logits_r)

        # KcBERT Forward
        out_k = self.kcbert(input_ids=input_ids_k, attention_mask=mask_k)
        logits_k = self.classifier_k(out_k.pooler_output)
        probs_k = self.softmax(logits_k)

        # Soft Ensemble (가중치 사용)
        ensemble_probs = self.w_r * probs_r + self.w_k * probs_k

        # Log Probabilities 반환
        return torch.log(ensemble_probs + Config.EPSILON)

# =========================================================
# 4. Focal loss 정의
# =========================================================
class FocalLoss(nn.Module):
    """데이터 불균형 해결을 위한 Focal Loss 구현"""
    def __init__(self, alpha=1, gamma=2, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_pt = inputs.gather(1, targets.view(-1, 1)).view(-1)
        pt = torch.exp(log_pt)

        loss = -self.alpha * ((1 - pt) ** self.gamma) * log_pt

        if self.weight is not None:
            batch_weights = self.weight.gather(0, targets.view(-1))
            loss = loss * batch_weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# =========================================================
# 5. 학습 함수 (Config 사용 통일)
# =========================================================
def train_stage_model(train_df, val_df, stage_name, custom_criterion=None):
    """단일 스테이지(1/2/3단계) 모델 학습을 실행하는 함수"""
    print(f"\n--- Training {stage_name} ---")
    print(f"Data count - Train: {len(train_df)}, Val: {len(val_df)}")

    # Config에서 모델 이름 가져오기
    tokenizer_r = AutoTokenizer.from_pretrained(Config.MODEL_R_NAME)
    tokenizer_k = AutoTokenizer.from_pretrained(Config.MODEL_K_NAME)

    train_dataset = DualBertDataset(train_df, tokenizer_r, tokenizer_k)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # Config에서 모델 이름, DEVICE, LEARNING_RATE 사용
    model = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 손실 함수 설정
    if custom_criterion:
        criterion = custom_criterion
        print(f"Applying Custom Loss Function: {type(criterion).__name__}")
    else:
        criterion = nn.NLLLoss()

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            ids_r = batch['input_ids_r'].to(Config.DEVICE)
            mask_r = batch['attention_mask_r'].to(Config.DEVICE)
            ids_k = batch['input_ids_k'].to(Config.DEVICE)
            mask_k = batch['attention_mask_k'].to(Config.DEVICE)
            labels = batch['labels'].to(Config.DEVICE)

            log_probs = model(ids_r, mask_r, ids_k, mask_k)
            loss = criterion(log_probs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f}")

    return model

# =========================================================
# 6. 예측 및 평가 함수
# =========================================================
def load_models_for_inference():
    """저장된 체크포인트에서 3단계 모델을 로드하기"""

    m1 = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)
    m2 = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)
    m3 = SoftEnsembleModel(Config.MODEL_R_NAME, Config.MODEL_K_NAME).to(Config.DEVICE)

    try:
        m1.load_state_dict(torch.load(f"{Config.SAVE_PATH}model_stage1.pth", map_location=Config.DEVICE))
        m2.load_state_dict(torch.load(f"{Config.SAVE_PATH}model_stage2.pth", map_location=Config.DEVICE))
        m3.load_state_dict(torch.load(f"{Config.SAVE_PATH}model_stage3.pth", map_location=Config.DEVICE))
        print("\n✅ 3단계 모델 로드 완료.")
    except Exception as e:
        print(f"\n❌ 모델 로드 오류: {e}")
        print("   💡 저장된 파일 이름, 경로, 또는 학습이 정상 완료되었는지 확인하세요.")
        return None, None, None

    m1.eval()
    m2.eval()
    m3.eval()
    return m1, m2, m3

def predict_three_stage_final(models, data_loader, thresholds):
    """3단계 계층적 예측을 수행하고 최종 레이블(0, 1, 2, 3)을 반환"""

    m1, m2, m3 = models
    th1, th2, th3 = thresholds
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            batch_size = batch['input_ids_r'].size(0)

            ids_r = batch['input_ids_r'].to(Config.DEVICE)
            mask_r = batch['attention_mask_r'].to(Config.DEVICE)
            ids_k = batch['input_ids_k'].to(Config.DEVICE)
            mask_k = batch['attention_mask_k'].to(Config.DEVICE)

            preds = np.zeros(batch_size, dtype=int)

            # 1. Stage 1 (0 vs 1/2/3) 예측
            log_probs_s1 = m1(ids_r, mask_r, ids_k, mask_k)
            probs_s1 = torch.exp(log_probs_s1)
            group_123_mask = (probs_s1[:, 1] >= th1)
            group_123_indices = torch.where(group_123_mask)[0].cpu().numpy()

            if len(group_123_indices) > 0:
                # 2. Stage 2 (1 vs 2/3) 예측
                sub_ids_r = ids_r[group_123_mask]
                sub_mask_r = mask_r[group_123_mask]
                sub_ids_k = ids_k[group_123_mask]
                sub_mask_k = mask_k[group_123_mask]

                log_probs_s2 = m2(sub_ids_r, sub_mask_r, sub_ids_k, sub_mask_k)
                probs_s2 = torch.exp(log_probs_s2)

                group_23_mask_in_s1 = (probs_s2[:, 1] >= th2)

                # S2에서 Class 0 (최종 1)로 예측된 샘플 업데이트
                preds[group_123_indices[~group_23_mask_in_s1.cpu().numpy()]] = 1

                group_23_indices_in_s1 = group_123_indices[group_23_mask_in_s1.cpu().numpy()]

                if len(group_23_indices_in_s1) > 0:
                    # 3. Stage 3 (2 vs 3) 예측
                    sub_ids_r = sub_ids_r[group_23_mask_in_s1.cpu()]
                    sub_mask_r = sub_mask_r[group_23_mask_in_s1.cpu()]
                    sub_ids_k = sub_ids_k[group_23_mask_in_s1.cpu()]
                    sub_mask_k = sub_mask_k[group_23_mask_in_s1.cpu()]

                    log_probs_s3 = m3(sub_ids_r, sub_mask_r, sub_ids_k, sub_mask_k)
                    probs_s3 = torch.exp(log_probs_s3)

                    pred_s3_final_mask = (probs_s3[:, 1] >= th3)

                    # S3에서 Class 0 (최종 2) 업데이트
                    preds[group_23_indices_in_s1[~pred_s3_final_mask.cpu().numpy()]] = 2

                    # S3에서 Class 1 (최종 3) 업데이트
                    preds[group_23_indices_in_s1[pred_s3_final_mask.cpu().numpy()]] = 3

            all_preds.extend(preds.tolist())

    return all_preds

def evaluate_and_print(y_true, y_pred, thresholds):
    """평가 결과 출력 코드"""

    print("\n" + "="*70)
    print(f"✨ 3단계 계층적 분류 최종 결과 (Thresholds: {thresholds})")
    print("="*70)

    print("📋 [Classification Report]")
    print(classification_report(y_true, y_pred, target_names=["0", "1", "2", "3"], digits=4, zero_division=0))
    print("-" * 70)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm,
                         index=['Actual 0', 'Actual 1', 'Actual 2', 'Actual 3'],
                         columns=['Pred 0', 'Pred 1', 'Pred 2', 'Pred 3'])
    print("📐 [Confusion Matrix]")
    print(cm_df.to_markdown(index=True))
    print("="*70)

# =========================================================
# 7. 메인 실행 로직 (학습 및 평가 통합)
# =========================================================
def prepare_stage_data(df_full, exclude_list, label_map):
    """특정 Stage를 위한 데이터를 준비하고, 이진 분류 레이블을 할당"""
    if exclude_list:
        df_stage = df_full[~df_full['label'].isin(exclude_list)].copy()
    else:
        df_stage = df_full.copy()

    df_stage['label'] = df_stage['label'].map(label_map)

    t_stage, v_stage = train_test_split(
        df_stage, test_size=0.1, stratify=df_stage['label'], random_state=Config.SEED
    )
    return t_stage, v_stage

def execute_training_stage(train_df, val_df, stage_name, stage_num, focal_gamma, weight_tensor_list):
    """단일 Stage의 학습을 실행하고 모델을 저장"""
    print(f"\n{'='*50}\nExecuting {stage_name}\n{'='*50}")

    weights = torch.tensor(weight_tensor_list, dtype=torch.float).to(Config.DEVICE)
    focal_loss_fn = FocalLoss(gamma=focal_gamma, alpha=1.0, weight=weights)

    model = train_stage_model(
        train_df, val_df,
        f"{stage_name} with Focal Loss (Gamma={focal_gamma})",
        custom_criterion=focal_loss_fn
    )

    save_path = f"{Config.SAVE_PATH}model_stage{stage_num}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    del model; gc.collect(); torch.cuda.empty_cache()

    return save_path

def run_training_pipeline(data_path):
    """데이터 로드부터 3단계 학습까지 전체 파이프라인을 실행"""

    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일 경로 오류: {data_path}를 찾을 수 없습니다.")
        print("   💡 파일 경로를 수정하거나, Colab 환경에 파일을 업로드했는지 확인해주세요.")
        return None

    print("--- 0. 데이터 로드 및 분할 ---")
    df = pd.read_json(data_path)

    train_full, test_full = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=Config.SEED
    )

    t_s1, v_s1 = prepare_stage_data(train_full, [], {0: 0, 1: 1, 2: 1, 3: 1})
    t_s2, v_s2 = prepare_stage_data(train_full, [0], {1: 0, 2: 1, 3: 1})
    t_s3, v_s3 = prepare_stage_data(train_full, [0, 1], {2: 0, 3: 1})

    # === 모델 학습 실행 ===
    execute_training_stage(t_s1, v_s1, "Stage 1 (0 vs 1-3)", 1, focal_gamma=5.0, weight_tensor_list=[1.5, 3.0])
    execute_training_stage(t_s2, v_s2, "Stage 2 (1 vs 2-3)", 2, focal_gamma=5.0, weight_tensor_list=[1.5, 3.0])
    execute_training_stage(t_s3, v_s3, "Stage 3 (2 vs 3)", 3, focal_gamma=1.0, weight_tensor_list=[1.0, 3.0])

    print("\n🎉 모든 Stage 학습 및 모델 저장이 완료되었습니다. 🎉")
    return test_full

def run_evaluation_pipeline(test_data_df, thresholds):
    """평가 파이프라인 실행"""
    print("\n\n" + "#"*70)
    print("### 최종 평가 파이프라인 시작 ###")
    print("#"*70)

    models = load_models_for_inference()
    if models is None or any(m is None for m in models):
        return

    tokenizer_r = AutoTokenizer.from_pretrained(Config.MODEL_R_NAME)
    tokenizer_k = AutoTokenizer.from_pretrained(Config.MODEL_K_NAME)

    test_dataset = DualBertDataset(test_data_df, tokenizer_r, tokenizer_k)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    y_true = test_data_df['label'].astype(int).tolist()

    print("--- 2. 테스트 데이터로 예측 실행 ---")
    y_pred = predict_three_stage_final(models, test_loader, thresholds)

    evaluate_and_print(y_true, y_pred, thresholds)

    del models, test_loader, test_dataset
    gc.collect()
    torch.cuda.empty_cache()

# =========================================================
# 9. 통합 실행
# =========================================================

DATA_FILE_PATH = "1217 최종 데이터셋.json"  # 학습을 위한 데이터셋 넣기
test_data_df = run_training_pipeline(DATA_FILE_PATH)

# --- 2. 평가 실행 ---
if test_data_df is not None:
    EVALUATION_THRESHOLDS = (0.4, 0.7, 0.45)
    run_evaluation_pipeline(test_data_df, EVALUATION_THRESHOLDS)
