# Dataset 파일 정리 가이드

> `gr00t/data/dataset*.py` 파일들의 역할, 상속 관계, 사용처를 정리한 문서.

---

## 한눈에 보기

```
LeRobotSingleDataset  (dataset.py — NVIDIA 원본)
│
├── ActionOnlyDataset                   (dataset_action_only.py)
│   ├── ActionOnlyDatasetV3             (dataset_action_only_v3.py)
│   └── ActionStateDataset              (dataset_action_state_pretransform.py)
│       └── ActionStateDatasetV3        (dataset_action_state_pretransform_v3.py)
│
└── LeRobotSingleDatasetActlatFM        (dataset_actlat_fm.py)
    (= LeRobotSingleDatasetWithSplit)   (dataset_val.py — 거의 동일)

PreTransformedActionOnlyDataset         (dataset_action_only_pretransform.py)
└── PreTransformedActionOnlyDatasetV3   (dataset_action_only_pretransform_v3.py)

PreTransformedActionStateDataset        (dataset_action_state_pretransform.py)
└── PreTransformedActionStateDatasetV3  (dataset_action_state_pretransform_v3.py)
```

---

## 파일별 상세 설명

### 1. `dataset_action_only.py`
- **클래스**: `ActionOnlyDataset`
- **역할**: 비디오/언어 없이 **action parquet만** 로드. LeRobotSingleDataset에서 action modality만 추출.
- **특징**: transform(정규화·concat) 자체 구성, 에피소드 단위 train/val split 내장, 학습이 훨씬 빠름.
- **사용처**: 직접 사용 X. `ActionStateDataset`, `PreTransformedActionOnlyDataset`의 **부모 클래스**로 사용.
  - `eval_tokenizer_recon.py`의 `ActionOnlyCollator`가 이 파일에서 import됨.
  - `train_action_latent_tokenizer_v3.py`도 `ActionOnlyCollator`를 여기서 import.

---

### 2. `dataset_action_only_v3.py`
- **클래스**: `ActionOnlyDatasetV3`
- **역할**: `ActionOnlyDataset` + **persistent fixed-val split** 기능 추가.
- **V3 핵심 차이**: split을 랜덤 생성 대신 JSON 파일에서 로드/저장 → 여러 실험이 동일한 val 에피소드를 공유.
- **사용처**:
  - `eval_tokenizer_recon.py`에서 직접 import하여 evaluation 시 사용.
  - `dataset_action_only_pretransform_v3.py`의 내부 source로 사용.

---

### 3. `dataset_action_only_pretransform.py`
- **클래스**: `PreTransformedActionOnlyDataset`
- **역할**: `ActionOnlyDataset`을 **초기화 시 전체 데이터를 메모리에 캐싱**. `__getitem__`은 단순 텐서 인덱싱만 수행.
- **효과**: parquet I/O 및 transform 오버헤드 제거 → 학습 속도 대폭 향상.
- **출력**: `{"action": Tensor[T, D]}`
- **사용처**:
  - `train_action_latent_tokenizer.py` (v1) — 직접 import
  - `train_action_latent_tokenizer_v2.py` — `num_hand_tokens == 0` 분기에서 사용

---

### 4. `dataset_action_only_pretransform_v3.py`
- **클래스**: `PreTransformedActionOnlyDatasetV3`
- **역할**: `PreTransformedActionOnlyDataset`과 동일한 캐싱 방식 + 내부적으로 `ActionOnlyDatasetV3` 사용 → fixed-val split 지원.
- **사용처**:
  - **`train_action_latent_tokenizer_v3.py`** — `num_hand_tokens == 0` 분기 (action-only, state 예측 없을 때)
  - 즉 **두 sbatch 스크립트(`recon_ln_bn64_fs.sh`, `mask_recon_q99_l1.sh`)의 Stage 1**에서 사용됨.

---

### 5. `dataset_action_state_pretransform.py`
- **클래스**: `ActionStateDataset`, `PreTransformedActionStateDataset`, `ActionStateCollator`
- **역할**: action에 더해 **state(손 관절 등)도 함께 로드**하여 캐싱. 추가로:
  - `hand_state`: 현재 hand state
  - `future_hand_states`: 미래 시점 hand state
  - `fast_tokens`: FAST tokenizer로 action discrete 토큰화 결과 (선택적)
- **사용처**:
  - `train_action_latent_tokenizer_v2.py` — `num_hand_tokens > 0` 분기 (hand state 예측 있을 때)
  - `dataset_action_state_pretransform_v3.py`의 **부모 클래스**

---

### 6. `dataset_action_state_pretransform_v3.py`
- **클래스**: `ActionStateDatasetV3`, `PreTransformedActionStateDatasetV3` (+ `ActionStateCollator` re-export)
- **역할**: `PreTransformedActionStateDataset`과 동일 + fixed-val split 지원.
- **사용처**:
  - **`train_action_latent_tokenizer_v3.py`** — `num_hand_tokens > 0` 분기 (hand state 예측 있을 때)
  - 현재 두 sbatch 스크립트는 `--num-hand-tokens 0`이므로 **실제로는 호출되지 않음**.
  - 향후 hand state 예측 실험 시 사용.

---

### 7. `dataset_actlat_fm.py`
- **클래스**: `LeRobotSingleDatasetActlatFM`, `ActlatFMDataCollator`
- **역할**: Stage 2 VLA 학습용. **비디오 + 언어 + action 전체**를 로드하는 full 데이터셋.
  - train/val split 내장 (에피소드 단위)
  - single-timestep observation 기반 (1~3카메라 지원, FLARE 듀얼타임스텝 없음)
- **사용처**:
  - **`scripts/gr00t_finetune_actlat_fm.py`** — **두 sbatch 스크립트의 Stage 2** 모두에서 사용.

---

### 8. `dataset_val.py`
- **클래스**: `LeRobotSingleDatasetWithSplit`
- **역할**: `LeRobotSingleDataset` + train/val split. `dataset_actlat_fm.py`의 `LeRobotSingleDatasetActlatFM`과 거의 동일한 역할.
- **차이**: `dataset_actlat_fm.py`보다 먼저 만들어진 버전. collator가 없음.
- **사용처**:
  - `scripts/gr00t_finetune_with_val.py`
  - `scripts/gr00t_probing_qformer.py`
  - **현재 두 sbatch 스크립트에서는 사용 안 함**.

---

### 9. `dataset_action_state_pretransform_copy.py`
- **역할**: `dataset_action_state_pretransform.py`의 **구버전 백업**. `action_target_rotations` 파라미터가 없는 이전 버전.
- **사용처**: **어디서도 import되지 않음** → 삭제 후보.

---

## 두 sbatch 스크립트 기준 실제 사용 파일

| Stage | 스크립트 | Dataset 파일 | 조건 |
|-------|----------|-------------|------|
| Stage 1 (Tokenizer) | `train_action_latent_tokenizer_v3.py` | `dataset_action_only_pretransform_v3.py` | `num_hand_tokens == 0` → **두 스크립트 모두** |
| Stage 1 (Tokenizer) | `train_action_latent_tokenizer_v3.py` | `dataset_action_state_pretransform_v3.py` | `num_hand_tokens > 0` → 현재 두 스크립트에선 **미사용** |
| Stage 2 (VLA) | `gr00t_finetune_actlat_fm.py` | `dataset_actlat_fm.py` | **두 스크립트 모두** |

---

## 파일별 "쓰이는가?" 요약

| 파일 | 현재 두 sbatch에서 | 다른 스크립트에서 |
|------|:-----------------:|:-----------------:|
| `dataset_action_only.py` | △ (간접: 부모 클래스) | △ (collator만 직접 사용) |
| `dataset_action_only_v3.py` | △ (간접: 부모 클래스) | ✓ `eval_tokenizer_recon.py` |
| `dataset_action_only_pretransform.py` | ✗ | ✓ `train_v1`, `train_v2` |
| **`dataset_action_only_pretransform_v3.py`** | **✓ Stage 1** | |
| `dataset_action_state_pretransform.py` | ✗ (현재) | ✓ `train_v2` |
| `dataset_action_state_pretransform_v3.py` | ✗ (현재) | (hand예측 실험 시) |
| **`dataset_actlat_fm.py`** | **✓ Stage 2** | |
| `dataset_val.py` | ✗ | ✓ `finetune_with_val`, `probing` |
| `dataset_action_state_pretransform_copy.py` | ✗ | ✗ → **삭제 후보** |

---

## 핵심 설계 패턴

```
1. action-only vs action+state
   - action-only:  pretransform → (V3: fixed-val 추가)
   - action+state: pretransform → (V3: fixed-val 추가)
   두 계열 모두 "초기화 시 전체 캐싱" 패턴 공유.

2. v2 → v3 변경사항
   유일한 차이: train/val split이 seed 기반 랜덤 → JSON 파일 기반 persistent
   (여러 실험이 같은 val 에피소드를 공유하기 위함)

3. Stage 1 vs Stage 2 dataset 설계
   Stage 1 (tokenizer): action/state만 읽는 경량 dataset (비디오 없음)
   Stage 2 (VLA):       비디오 + 언어 + action 전체 로드하는 full dataset
```
