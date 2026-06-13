# V4 (visual-dynamics) latent의 VLA 학습 난이도 문제와 개선 방향

- 날짜: 2026-06-11
- 대상 실험: `sbatch_scripts/robocasa_gr1_100/v4/recon_dino_bn64_l1_mse_naiveln.sh`
  (동일 증상이 `recon_vggt_bn64_l1_mse_naiveln.sh`에도 적용)
- 관련 코드: `gr00t/model/action_latent_tokenizer_v4.py`,
  `gr00t/model/action_latent_tokenizer_wrapper.py`,
  `gr00t/model/gr00t_n1_actlat_fm.py`,
  `gr00t/experiment/trainer_actlat_fm.py`,
  `gr00t/model/rla_modules.py`

---

## 1. 현재 문제 (관측된 증상)

V4 토크나이저는 action autoencoder(V3)에 RLA식 visual-dynamics를 결합한 구조다.
latent은 다음과 같이 정의된다:

```
latent(time_tok) = f(action,  x1_feat - x0_feat)
   x0 = 청크 시작 프레임(obs index 0)
   x1 = 청크 끝   프레임(obs index action_horizon-1, 약 15 step 뒤)
   x0_feat / x1_feat = 동결된 DINO/VGGT patch feature [B, Lp, C]
```

- **토크나이저 학습(Stage 1)**: `eval/recon_l1`이 vision 없이 action만 enc/dec하던
  이전(V2/V3) 실험과 비슷하게 잘 나옴 → 토크나이저 자체는 정상 수렴.
- **VLA 학습(Stage 2)**: `eval/latent_l1`이 이전보다 훨씬 큼.
  - 이전(action-only): 0.25 이하, 또는 0.5 이하까지 하강.
  - V4(vision 사용): 0.9 부근에서 정체.
  - `action_l1`은 이전과 비슷.
- **시뮬 성능**: vision 정보를 넣어도 의미 있는 향상이 없음.

## 2. 코드 정합성 검증 결과 (버그 아님)

latent이 VLA 학습에 들어가는 전 경로를 Stage1과 대조 → **버그 없음, 정합적으로 사용 중.**

- **프레임 정렬**: Stage1 `ActionFramesDatasetV4`(video modality `delta_indices=[0,H-1]`)와
  Stage2 `LeRobotSingleDatasetActlatFMV4._load_frame_pair`(`step_indices=[0,H-1]+base`)가
  동일한 프레임 쌍을 읽는다. 둘 다 `[0, traj_len-1]` clip, 둘 다
  `timestamp[step_indices]` → `get_frames_by_timestamps`.
  (`dataset.py:758-782` vs `dataset_actlat_fm_v4.py:62-83`)
- **horizon/size 일치**: `frame_action_horizon=len(action_indices)`(=16),
  `frame_image_size=224`. (`gr00t_finetune_actlat_fm.py:289-291`)
- **collator**: `_collate_actlat_fm`이 `frame_x0/x1`을 `[B,3,224,224]` uint8로 보존.
  (누락 시 v4 wrapper가 `ValueError`로 즉시 죽으므로, 학습이 도는 한 존재함)
- **feature 추출 일치**: wrapper `_resolve_dino_feats`와 trainer `_extract_feats`가
  동일(`/255`, VGGT `[B,Lp,C]`, DINO grid-flatten).
  (`wrapper:904-936` vs `train_..._v4.py:114-125`)
- **latent 사용/디코딩**: `encode`가 `dino_diff=x1_feat-x0_feat`를 action latent과
  fusion → `time_tok [B,16,64]`. `get_latent_target("all")`은 Ng=Nh=0이라 그대로 반환.
  action head는 `action_dim=64, action_horizon=16`으로 재구성. eval `latent_l1`은
  같은 wrapper로 만든 target과 비교하고 decode도 동일 경로 → self-consistent.
  (`gr00t_n1_actlat_fm.py:170-181`, `trainer_actlat_fm.py:104-133`)

→ **latent_l1이 높은 것은 코드 오류가 아니라 latent의 "구조" 문제.**

## 3. 핵심 진단: latent의 *구조*가 VLA가 회귀하기 나쁘다

> 반론 정리: "action도 VLA의 입력이 아니라 예측 대상이고, future를 생성하는 논문도
> 많다." → 맞다. expert demo + 결정론적 sim에서는 x1도 (obs, action)의 함수라
> 원리적으로 action만큼 예측 가능하다. 따라서 "future라서 예측 불가능"은 틀린 설명.
> 진짜 문제는 **visual dynamics를 VLA가 학습하기 쉬운 구조로 latent에 넣지 못한 것.**

action latent이 잘 예측되는 이유는 "입력이라서"가 아니라 **저차원·매끄럽고·스케일이
정규화된, action의 결정론적 인코딩**이기 때문이다. V4의 visual-dynamics latent은 다음
세 가지가 동시에 깨져 있다:

1. **목적함수가 예측가능성과 무관하게 latent을 빚는다.**
   latent은 recon+dino loss로만 모양이 결정된다. dino loss("x0_feat에서 x1_feat 복원")는
   **action과 무관한 고엔트로피 perceptual detail**(텍스처/렌더링 디테일/패치 노이즈)까지
   latent에 담도록 보상한다. 이 성분은 task/action의 함수가 아닌 거의 순수 노이즈라,
   VLA가 못 맞추고 맞출 필요도 없는 방향을 타깃에 만든다 → latent_l1 바닥이 올라감.

2. **스케일/정규화가 free다.**
   fusion `out_layer = Linear(norm_out(h))`, `norm_output_tokens=False`
   (`rla_modules.py:223-224, 284`). latent의 per-dim 분산이 학습에 맡겨져 임의적.
   v3 baseline이 output LayerNorm으로 단위 스케일이었다면 **0.9 vs 0.25의 상당 부분이
   단순 스케일 차이**일 수 있음 → 절대 latent_l1은 토크나이저 간 비교 불가 지표.

3. **action과 vision이 같은 토큰에 엉켜 있다.**
   `time_tok`이 dino_diff에 attention해 나오므로 매 timestep 토큰이 action+visual 혼합.
   쉬운 부분(action)과 어려운 부분(visual detail)이 한 벡터에 섞여 같은 회귀 난이도로
   끌려 올라감. (`TimeWiseEncoderV4.forward`, `action_latent_tokenizer_v4.py:418-431`)

## 4. 개선 방향 (대체 방법, 우선순위 순)

### (0) 스케일부터 공정 비교 — 진단 먼저
- v4/v3 latent의 per-dim **std**를 측정 → `latent_l1 / std` (또는 R²)로 다시 본다.
- 여기서 갭이 거의 사라지면 **순수 정규화 문제**.
  - 해결: fusion 출력에 `norm_output_tokens=True`(L2 normalize) 또는 latent에 final
    LayerNorm / dataset-whitening(per-dim mean·std 고정) 적용.
- 비용: 측정은 추론만. 적용은 토크나이저 소폭 수정 + 재학습.

### (1) lambda_dino sweep — 인과 확인
- `lambda_dino → 0`으로 갈수록 `latent_l1`이 v3 수준으로 떨어지는지 확인.
- 떨어지면 "dino loss가 만든 방향 = 어려운 부분"이라는 직접 증거.
- 비용: 토크나이저 재학습 여러 개(또는 기존 `recon_dino_bn64`(dino) vs 순수 recon 비교).

### (2) action-irrelevant 성분을 bottleneck에서 제거
- information bottleneck(VIB / latent variance·KL penalty) 또는 더 좁은 `token_dim`으로,
  **action으로 설명되지 않는 visual 잔차**를 latent에서 억제.
- 핵심: latent에 "x1 복원에 필요한 전부"가 아니라 "**action과 상관된 visual dynamics만**"
  남긴다.
- 비용: 토크나이저 구조/loss 변경 + 재학습.

### (3) 쉬운 부분과 어려운 부분을 분리
- per-timestep action 토큰은 깨끗하게(action-only와 동일) 유지.
- visual dynamics는 **별도의 소수 토큰**(global/dynamics token)으로 분리.
- VLA가 action 토큰은 낮은 l1로 맞추고, dynamics 토큰은 따로 다루게 → 엉킴 제거.
- 기존 `num_global_tokens` 경로 재사용 가능 (wrapper/디코더 이미 지원).
- 비용: 구조 변경 + 재학습. VLA target 토큰 수/타깃 모드 조정 필요.

### (4) 예측가능성 자체를 정규화
- 토크나이저 학습에 "**x0(현재 프레임)만으로 latent을 회귀하는 작은 predictor**"를 붙여,
  encoder가 visual dynamics를 **현재 관측에서 추론 가능한 방향**에 담도록 유도.
- VLA가 실제로 맞춰야 할 양(현재 obs로부터의 예측가능성)을 직접 최적화.
- 비용: aux predictor + loss 추가, 재학습.

### (5) latent dim 축소 (64 → 32)
- 본질적으로 **(2) information bottleneck의 무딘 버전**. 용량을 조이면 encoder가 우선순위를
  강제당해 recon-우세 시 action 보존·시각 디테일 우선 폐기 → 어려운 성분 감소 가능.
- **주의**: `latent_l1 = F.l1_loss`는 원소당 평균이라 dim 축소로 절대값이 자동으로 안 줄어듦.
  이득은 *내용*(어려운 방향이 빠지는지)에서만 옴. dino loss가 시각 디테일을 강제하면 더
  압축만 될 뿐.
- **반드시 `recon_l1` 동반 관찰**: 32가 action 복원을 깨면 latent_l1↔recon_l1 맞바꾸기일 뿐.
- 비용: 가장 쌈. `--token-dim 32`로 A/B. (담당: 사용자가 직접 진행)

### (6) VAE 방식 latent mapping
구조는 LDM의 KL-autoencoder와 **동일**: encoder→(μ, logvar)→reparameterize→KL.
적용 시 **(0) 스케일 정규화**를 확실히 잡고, 부분적으로 **(2) bottleneck** 효과.

**최소 변경 설계** (out_layer 그대로 두는 방식 — 비-VAE 경로 바이트 동일 유지):
- `TimeWiseEncoderV4`: 기존 64-dim 출력을 **μ**로 쓰고 `logvar_head = nn.Linear(token_dim,
  token_dim)`(zero-init)만 추가. `forward(..., sample)`에서
  `z = μ + ε·exp(0.5·logvar)`(학습) / `z = μ`(추론), free-bits KL 계산.
- `ActionLatentTokenizerV4`: 공개 `encode`는 항상 μ(`sample=False`), `forward`(학습)만
  `sample=True`+`lambda_kl*kl` 추가, `register_buffer("_is_vae")` 마커.
- wrapper `_build_timewise_v4_tokenizer`: `_is_vae` 감지 시 `logvar_head` 생성(strict load
  매칭). `token_dim`은 out_layer shape 그대로 64, 추론은 μ 반환 → **downstream 무변경**.
- train 스크립트: `--use-vae --lambda-kl --free-bits`(+선택 `--kl-anneal-steps`), `loss_kl` 로깅.
- **VLA 쪽(`gr00t_n1_actlat_fm.py`/`trainer_actlat_fm`/`get_latent_target`) 무변경.**
  `SimpleTokenTransformer`(rla_modules, verbatim 복사본)·디코더 무변경.

**왜 타깃은 μ인가**: encoder가 분포 `q(z|x)=N(μ,σ²)`를 내놓음. 학습은 reparam을 위해 z
샘플이 필요하지만, VLA 타깃으로 샘플 z를 쓰면 같은 입력에 매번 `μ+랜덤노이즈`가 되어
**예측 불가능한 라벨 노이즈**가 latent_l1 바닥을 인위적으로 올림(σ≈0.3이면 완벽 예측해도
`≈0.8σ≈0.24` 오차 깔림). μ는 분포 중심(denoised)이라 예측 가능한 안정 타깃.

**LDM 비교 — 두 갈래 (어느 KL regime이냐). 용어 주의: "클래식"이 두 뜻**:
- **Stable Diffusion이 실제로 하는 것**(="클래식 latent diffusion") = **(A) 약한 KL**.
- **교과서 VAE**(Kingma-Welling, β=1, `N(0,I)` prior) = **(B)**. SD는 (B)를 *일부러 안 씀*.

|  | (A) SD/LDM식 약한 KL ★실제 클래식 | (B) 교과서 β=1 강한 VAE |
|---|---|---|
| 목적 | 스케일 정규화 (0) | 정보 bottleneck (2) |
| KL weight | 아주 작음(~1e-6) | 큼(β≈1) |
| σ | ≈0 | 큼 |
| VLA 타깃 | z 샘플 OK(σ≈0이라 μ와 동일) | **μ 필수** |
| collapse | 일부러 피함 | 위험 → free-bits/anneal 필요 |
| 성격 | 사실상 정규화된 AE | 진짜 prior-matching VAE |
| recon 영향 | 거의 없음(고fidelity 유지) | 악화 위험 |

- **SD = (A)**: KL~1e-6 + `z.sample()*0.18215`(고정 scaling factor). **실제 스케일 정규화 일은
  KL이 아니라 scaling factor가 한다** — 작은 KL은 거의 장식. 정보를 *보존*하므로 nuisance를
  빼는 게 아님. σ≈0이라 "μ vs z" 구분은 무의미(내 'μ 필수' 주의점은 (B)에서만 유효).
- **SD를 레퍼런스로 하면 (A)로 가라.** β=1 강한 VAE((B))는 SD가 피하는 방향이고 복원이 깨지기 쉬움.
- ★**가장 싼 길(재학습 0)**: SD의 scaling factor는 동결 VAE 출력에 상수를 곱하는 것뿐.
  우리도 **토크나이저 재학습 없이** latent std를 데이터셋에서 한 번 측정해 `get_latent_target`에서
  나누고 `decode_latent`에서 곱하면 됨(=0.18215의 비학습 버전). (0) 스케일 가설을 즉시
  검증/해결. 스케일 문제면 이걸로 끝 — VAE도 dim 축소도 불필요.

**핵심 한계(재확인)**: dino loss가 시각 디테일에 "비용을 지불"하므로 **VAE 단독으로는
action-무관 시각 엔트로피를 못 버린다.** VAE의 몫은 (0) 스케일 + 약한 bottleneck까지.
그 이상은 (3) 토큰 분리나 dino 목적함수 축소와 병행해야 함.

**측정 함정**: VAE의 KL이 latent을 단위 스케일로 정규화해 latent_l1이 *그냥* 낮아져 보임.
반드시 **scale-normalized(R² 또는 latent_l1/std)**로 비교.

**구현 완료 (2026-06-11, SD식 (A))** — `use_vae` 플래그(기본 False). False면 새 파라미터/버퍼/
난수 호출 전무 → 기존과 바이트 동일(스모크로 검증). True면 fusion 출력=μ, `logvar_head`
추가, encode가 reparam 샘플 z 반환(VLA 타깃=z). KL은 `lambda_kl`(기본 1e-6, SD regime),
`kl_free_bits`(기본 0). out_layer/decoder/latent dim 무변경.
- 변경 파일: `action_latent_tokenizer_v4.py`(`TimeWiseEncoderV4` logvar_head+reparam+KL,
  `ActionLatentTokenizerV4` lambda_kl+`_is_vae` 마커+KL loss), `action_latent_tokenizer_wrapper.py`
  (`_is_vae` 감지→matching encoder 재구성), `train_action_latent_tokenizer_v4.py`
  (`--use-vae/--lambda-kl/--kl-free-bits` 인자, loss_kl 로깅, ckpt config).
  **VLA 쪽 무변경.**
- 실행 스크립트: `sbatch_scripts/robocasa_gr1_100/v4/recon_dino_bn64_l1_mse_naiveln_vae.sh`
  (기존 dino bn64 naiveln과 동일 + `--use-vae --lambda-kl 1e-6`).
- 검증: off-path state_dict 키 동일, vae-path 추가 키만 `_is_vae`/`encoder.logvar_head.*`,
  encode 확률성(z1≠z2), KL grad 흐름, wrapper strict 로드 round-trip 모두 통과.
- 주의: SD식이라 KL 자체는 스케일을 거의 안 잡음(SD는 scaling factor가 그 일을 함).
  스케일 정규화가 필요하면 (0)의 latent 표준화를 병행하거나 lambda_kl을 키워 튜닝.

## 5. 권장 진행 순서

1. **(0)+(1) 진단**부터: v3 baseline 체크포인트와 v4를 같은 val 배치에 태워
   `latent std`, `latent_l1/std`, `lambda_dino` 민감도를 한 번에 측정.
   → "스케일 문제냐 / 구조 문제냐" 확정.
2. 스케일 문제로 판명 → (0)의 정규화로 마무리.
3. 구조 문제로 판명 → (2)/(3)/(4) 중 선택.
   - 가장 깔끔한 1순위 후보: **(3) 토큰 분리** + **(0) 정규화** 병행.
   - 정보량 제어가 핵심이면 **(2) IB**.
   - VLA 예측가능성 직접 최적화가 목표면 **(4) predictor 정규화**.

## 6. 참고: 이미 있는 진단 도구
- `scripts/diag_vggt_frame_diff.py`: x0/x1 feature의 차이 크기·cosine·identity baseline
  ·random-pair baseline·degenerate(픽셀 동일) 비율을 측정. visual diff가 의미 신호를
  갖는지(혹은 거의 0인지) 확인용. (0)/(1) 진단과 상보적.
