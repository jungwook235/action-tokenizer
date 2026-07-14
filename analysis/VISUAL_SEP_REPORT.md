# v3(action-only) vs v4(DINO-fused) 토크나이저 — Visual Separation 분석 리포트

**질문:** v4 토크나이저는 "액션 값은 같지만 시각적 맥락(장면/물체)이 다른" 청크를 서로 다른 latent으로 분리하는가? action-only인 v3는 (정의상) 그러지 못한다. 이걸 정량·시각적으로 확인한다.

**비교 대상 (모두 checkpoint-100000):**
- **v3** = `..._v3_recon_ln_bn16` — 액션만 입력, latent K=16
- **v4** = `..._v4_recon_dino_bn64_l1_mse_naiveln_vae` — 액션 + DINOv2 시각 특징 융합, VAE, latent K=64

**두 데이터셋에서 동일 분석 실행:**
- **dexjoco** (dual-arm 5 task, N=2322) — 각 task의 액션 공간이 거의 disjoint
- **gr1_1000** (24 PnP task, N=4008) — 집기-놓기 primitive를 공유 → "같은 액션·다른 장면"이 실제로 존재

---

## TL;DR — 결론 먼저

| 지표 | dexjoco (액션 disjoint) | **gr1_1000 (primitive 공유)** | 가설 지지? |
|---|---|---|---|
| **① R²(action→latent)** v3 / v4 | 0.996 / 0.983 (Δ0.01) | 0.984 / **0.899** (Δ0.085) | gr1 ✓ |
| **④ NN action↔latent** v3 / v4 | 0.939 / 0.956 (반대) | **0.828 / 0.762** | gr1 ✓ |
| **④ NN visual↔latent** v3 / v4 | 0.803 / 0.831 | 0.539 / **0.615** | gr1 ✓ |
| **④ within-action task-ARI** v3 / v4 | 0.623 / 0.409 (반대) | **0.272 / 0.356** | gr1 ✓ |
| **② near-dup 그룹 수** | **0** (표본 없음) | **6** (각 13~17 task) | gr1 ✓ |
| **②′ frame-swap** (액션 고정, visual만 교체) | v4 6% / v3 0% | v4 **20%** / v3 0% | gr1 ✓✓ |

**한 줄 결론:** 가설은 참이다. 단 **"같은 액션이 다른 시각 맥락에서 실제로 재등장하는 데이터"에서만 관측된다.** gr1처럼 primitive를 공유하는 태스크 묶음에서는 v4가 액션이 같아도 시각 맥락에 따라 latent을 뚜렷이 분리한다(액션 스케일의 ~20%). dexjoco는 태스크 액션이 겹치지 않아 애초에 검증 표본이 없다 — 가설이 틀린 게 아니라 데이터가 부적합.

---

## 0. 공통 준비 — 공유 샘플 인코딩

- **파일:** `analysis/vsep_collect.py` → 출력 `analysis/output/visual_sep{,_gr1}/cache.npz`
- **핵심:** 동일한 val 액션 청크·프레임을 v3·v4 **둘 다로** 인코딩해서, 이후 모든 분석이 완전히 같은 표본을 비교하도록 함. v4 latent은 VAE 샘플이 아니라 **결정적 posterior mean(μ)**을 저장(재현성).
- 저장 내용: `A`(정규화 액션 [N,T,D]), `Z3`(v3 latent), `Z4`(v4 μ), `Vcontext`(mean-pool DINO f0‖f1, [N,2C]), `Vdyn`(f1−f0), `task`(데이터셋 인덱스), 그리고 프레임 재조회용 인덱스.

```python
# vsep_collect.py — v4는 액션과 DINO 특징을 함께 인코딩(μ), v3는 액션만
mu, _sig, _lv, _z = _encode_mu_and_sample(enc, a, f0, f1)   # v4 (deterministic mu)
...
g, t, h = wrap.encode(b["action"].to(device).to(dtype))     # v3 (frames 미사용)
```

> 타당성: 두 토크나이저를 **완전히 동일한 입력**으로 비교하는 게 이 분석 전체의 공정성 근거다. dexjoco는 val이 task당 1 에피소드(총 5개), gr1은 task당 3 에피소드(총 72개)라 gr1 쪽 표본 다양성이 더 크다.

---

## ① 잔차 분산 분해 (Residual Variance Decomposition)

**무엇을 보나 (직관):** latent을 액션만으로 예측(ridge, 5-fold CV)했을 때의 설명력 `R²(action→latent)`을 본다. 값이 1에 가까우면 "latent = 액션 코드". v4가 시각 정보를 담았다면 액션만으로는 100% 설명이 안 돼야 한다(값이 낮아짐).

**코드/파일:** `analysis/vsep_stats.py` → `analysis_residual()` / `cv_r2()` → `residual_r2.txt`

```python
def cv_r2(X, Y, seed=0, alphas=(1,10,100,1000,1e4)):
    Xs = _pca(StandardScaler().fit_transform(X), 256, seed)
    Ys = StandardScaler().fit_transform(Y.reshape(Y.shape[0], -1))
    pred = cross_val_predict(RidgeCV(list(alphas)), Xs, Ys, cv=KFold(5, shuffle=True, random_state=seed))
    return r2_score(Ys, pred, multioutput="variance_weighted"), (Ys - pred)
```

**결과 (5-fold CV R², variance-weighted):**

| | R²(act→z) | R²(vis→z) | R²(a+v→z) | R²(vis→resAct) |
|---|---|---|---|---|
| **dexjoco** v3 | 0.9963 | 0.9727 | 0.9925 | 0.2758 |
| **dexjoco** v4 | 0.9833 | 0.9320 | 0.9700 | 0.2970 |
| **gr1** v3 | 0.9845 | 0.6932 | 0.9731 | 0.4127 |
| **gr1** v4 | **0.8990** | 0.6713 | **0.9059** | 0.3679 |

**해석:**
- dexjoco: v3(0.996)·v4(0.983) 둘 다 사실상 액션 코드 — 차이 0.01로 미미.
- gr1: v3 0.984 vs **v4 0.899** — v4 latent의 약 **10%가 액션으로 설명 안 됨**. 또한 `R²(a+v→z)`가 v4에서 액션 단독(0.899)보다 **증가(0.906)** → 시각이 액션 너머로 정보를 더함(v3는 오히려 감소=시각 무의미).

> **타당성/한계 (반드시 읽을 것):** `R²(vis→resAct)`(잔차를 시각으로 예측)는 원래 핵심 지표로 설계했지만 **confounded**다 — 액션과 시각이 서로 상관되면(에피소드 적음/액션 disjoint), 시각이 "액션의 비선형 성분"을 대리 예측해서 v3에서도 0.28~0.41로 높게 나온다. 그래서 이 열은 신뢰하지 말고, **`R²(act→z)` 열(v3 vs v4 격차)**과 아래 ②′ frame-swap을 함께 봐야 한다. gr1의 `R²(act→z)` 격차(0.085)가 이 방법에서 가장 믿을 만한 신호.

---

## ③ 액션 거리 vs latent 거리 (Δaction → 0에서의 latent floor)

**무엇을 보나 (직관):** 무작위 청크 쌍에 대해 (상대) latent 거리 vs 액션 거리를 그린다. v3는 원점을 지나는 직선(Δlatent ∝ Δaction)이어야 하고, v4는 **액션 거리가 0에 가까워져도 latent 거리가 0으로 안 떨어지는 "floor"**를 남겨야 한다 — 그 floor가 시각 신호의 크기.

**코드/파일:** `analysis/vsep_stats.py` → `analysis_dist()` → `dist_vs_dist.png/.txt`

```python
# 각 latent 거리를 자기 median으로 정규화(토크나이저 간 비교가능), 액션거리 최소 5%에서 floor 계산
lo = dar <= np.quantile(dar, 0.05)
floor3, floor4 = d3r[lo].mean(), d4r[lo].mean()
```

**결과 (Δaction 최소 5% 구간의 상대 latent 거리 floor):**

| | v3 floor | v4 floor | v4/v3 |
|---|---|---|---|
| **dexjoco** | 0.1259 | 0.2958 | 2.35× |
| **gr1** | 0.4957 | 0.6582 | 1.33× |

**dexjoco**
![dexjoco dist](output/visual_sep/dist_vs_dist.png)

**gr1_1000**
![gr1 dist](output/visual_sep_gr1/dist_vs_dist.png)

**해석:** 두 경우 모두 v4 floor > v3 floor. 하지만 배율만 보면 dexjoco(2.35×)가 더 커 보이는 함정이 있다 —

> **타당성/한계:** dexjoco에서 배율이 큰 건 **near-dup 액션 쌍이 실제로는 거의 없어서**(②에서 0개 확인) floor가 통계적으로 얇고, K=64 고차원 latent의 스케일 분포 특성(concave 포화)이 섞인 결과다. gr1은 floor 자체가 높은데(v3 0.50, v4 0.66) 이는 **진짜 near-dup 쌍이 많아서** 의미가 있다. 즉 이 지표는 **단독으로는 오해 소지**가 있고, ②(그룹 존재 여부)와 반드시 같이 봐야 한다.

---

## ④ NN 겹침 + latent t-SNE + 액션 통제 task-ARI

**무엇을 보나 (직관):** 세 가지.
1. **kNN 겹침** — 각 latent의 최근접이웃이 "액션 그래프"·"시각 그래프"와 얼마나 겹치나. 가설대로면 v4는 액션 그래프에서 **멀어지고**(action↔v4 < action↔v3) 시각 그래프로 **가까워짐**(visual↔v4 > visual↔v3).
2. **task-ARI** — latent을 KMeans로 군집화해 원래 task를 얼마나 복원하나.
3. **within-action task-ARI (핵심)** — **같은 액션 군집 안에서** latent이 여전히 task를 가르나. 액션을 통제한 뒤 남는 분리력 = 시각 분리력.

**코드/파일:** `analysis/vsep_stats.py` → `analysis_nn()` → `nn_overlap.txt`, `latent_tsne.png`

```python
def task_ari_within_action(Xz):     # 각 액션 군집 내부에서 task 복원 ARI의 가중평균
    for c in np.unique(ca):
        m = ca == c
        if m.sum() < 20 or len(np.unique(task[m])) < 2: continue
        lab = KMeans(len(np.unique(task[m])), ...).fit_predict(Xz[m])
        vals.append(adjusted_rand_score(task[m], lab)); wts.append(m.sum())
    return np.average(vals, weights=wts)
```

**결과 — kNN 겹침 (k=10):**

| | action↔v3 | action↔v4 | visual↔v3 | visual↔v4 |
|---|---|---|---|---|
| **dexjoco** | 0.939 | 0.956 ↑(반대) | 0.803 | 0.831 |
| **gr1** | 0.828 | **0.762** ↓✓ | 0.539 | **0.615** ↑✓ |

**결과 — task-ARI:**

| | global v3 | global v4 | **within-action v3** | **within-action v4** |
|---|---|---|---|---|
| **dexjoco** | 0.595 | 0.413 | 0.623 | 0.409 (반대) |
| **gr1** | 0.106 | 0.126 | **0.272** | **0.356** ✓ |

**dexjoco latent t-SNE**
![dexjoco tsne](output/visual_sep/latent_tsne.png)

**gr1_1000 latent t-SNE**
![gr1 tsne](output/visual_sep_gr1/latent_tsne.png)

**해석:**
- dexjoco: NN·within-action ARI 모두 **가설과 반대**(v3가 task 분리를 더 잘함). t-SNE에서 v3·v4 둘 다 5개 task를 깨끗한 곡선으로 분리(각 곡선 = task당 단일 에피소드의 시간 궤적).
- gr1: NN 두 방향 모두 **가설대로**, within-action ARI도 v4(0.356) > v3(0.272). 액션을 통제해도 v4가 task/시각을 더 잘 가름.

> **타당성/한계:** within-action task-ARI가 이 방법의 알짜다. 단 **dexjoco처럼 task 액션이 disjoint하면 액션만으로 이미 task가 갈려서 깨끗한 액션 코드(v3)가 오히려 높게 나온다** — 이 지표는 "같은 액션이 여러 task에 재등장하는" 데이터에서만 가설을 검증한다. 또 v3(K16)와 v4(K64)의 **차원이 달라** 절대 NN/ARI 값 자체보다는 **v3↔v4 상대 방향**과 **within vs global의 변화**를 봐야 한다.

---

## ② Near-duplicate 액션 그룹 + 실제 프레임 (스토리 그림)

**무엇을 보나 (직관):** 액션 공간에서 아주 작은 반경 안에 모이면서(=거의 같은 액션) **≥2개 task**에 걸친 청크 그룹을 찾는다. 그런 그룹 안에서 v4 latent이 얼마나 퍼지는지(v3 대비)를 재고, **실제 비디오 프레임**으로 "동작은 같은데 장면이 다름"을 보여준다.

**코드/파일:** `analysis/vsep_frames.py` → `find_groups()`, `group_spread()`, `pooled_corr()` → `neardup_groups.png`, `neardup.txt`

```python
# 액션거리 1퍼센타일을 near-dup 반경으로, ≥2 task에 걸친 그룹만 채택
radius = np.percentile(action_pairdist, radius_pct)   # p1.0
groups = find_groups(Xa, task, radius, min_size=4, ...)   # ≥2 tasks
# 그룹 내 v3/v4 latent 산포 = 그룹내 평균 쌍거리 / 전역 median
```

**결과:**
- **dexjoco: 0개 그룹.** action pairdist p1=2.46 vs median=38.3 → 5개 task의 액션이 거의 안 겹쳐 "같은 액션·다른 task" 쌍이 존재하지 않음. (그림 없음 — 표본 부재 자체가 결과)
- **gr1: 6개 그룹** (각 78~254 멤버, **13~17개 task**에 걸침). 그룹내 산포 v4 ~0.64 vs v3 ~0.48 (1.3~1.5×). 그룹내 latent-dist ↔ visual-dist 상관: v3 r=0.368, v4 r=0.397.

**gr1_1000 — 각 행이 하나의 near-dup 그룹 (거의 동일한 손동작, 서로 다른 장면/물체; 테두리색 = task)**
![gr1 neardup](output/visual_sep_gr1/neardup_groups.png)

**해석:** gr1 그림에서 한 행 안의 손 포즈(=액션)는 거의 같은데 배경(카펫/흰 테이블/대리석)·물체가 전혀 다르다. 정확히 "same action, different visual". v4는 이를 1.3~1.5× 더 퍼뜨림.

> **타당성/한계:** 이게 가장 직관적인 스토리 그림이다. 다만 그룹내 상관 지표(v3 0.368 vs v4 0.397)는 격차가 작은데, 이는 near-dup 그룹 안에도 **미세한 액션 차이가 남아 있고 그게 시각 차이와 상관**되기 때문(v3도 그 미세 액션차를 인코딩). 따라서 "산포 배율(1.3~1.5×)"과 "프레임 그림"이 주 증거, 상관계수는 보조. dexjoco에서 그룹이 0개인 것은 **이 데이터로는 관찰 검증 불가**라는 중요한 음성 결과.

---

## ②′ Frame-swap Intervention (결정적 메커니즘 테스트)

**무엇을 보나 (직관):** 관찰 분석이 안 되는 데이터(dexjoco)에서도 쓸 수 있는 **개입 실험**. 액션 청크를 **byte 단위로 고정**하고, 관측 프레임(DINO 특징)만 다른 샘플 것으로 바꿔 넣어 latent이 얼마나 움직이는지 측정한다.
- v3는 프레임을 안 보므로 **정확히 0** (음성 대조군, 정의상).
- v4는 프레임에 반응 → 이동량 = 시각이 latent에 주는 순수 기여. 액션 차이가 전혀 개입 안 되므로 confound가 없다.

**코드/파일:** `analysis/vsep_swap.py` → `swap_encode_v4()`, `spreads()` → `frame_swap.png`, `frame_swap.txt`

```python
# 앵커 액션 a_i를 고정하고, 도너 l의 프레임 특징으로 인코딩 → latent 이동
for l, d in enumerate(donors):
    mu,*_ = _encode_mu_and_sample(enc, a_i_batch, f0[d].expand(...), f1[d].expand(...))
# visual spread(액션 고정) / action scale(앵커 간 자연 산포) 의 비율을 보고
```

**결과 (액션 고정, 프레임만 교체; latent 이동량 ÷ 액션 스케일):**

| | v3 (median) | v4 (median) | corr(이동량, visual거리) |
|---|---|---|---|
| **dexjoco** | 0.0000 | 0.0606 (~6%) | 0.114 |
| **gr1** | 0.0000 | **0.2015 (~20%)** | 0.154 |

**dexjoco frame-swap**
![dexjoco swap](output/visual_sep/frame_swap.png)

**gr1_1000 frame-swap**
![gr1 swap](output/visual_sep_gr1/frame_swap.png)

**해석:** 액션을 완전히 고정해도 시각만 바꾸면 v4 latent이 dexjoco에서 액션 스케일의 ~6%, **gr1에서 ~20%(3.3배)** 이동. v3는 두 경우 다 정확히 0. 이동량은 시각 거리와 양의 상관(오른쪽 산점도의 상승 하한선).

> **타당성/한계:** 이게 **가장 깨끗한 메커니즘 증거**다 — 액션 confound가 원천 차단되고, v3=0이라는 완벽한 음성 대조군이 있다. 유일한 주의점: 실제로 안 맞는 (액션, 프레임) 조합을 넣는 것이라 인코더 입장에선 약간 OOD다. 하지만 이는 "인코더가 시각 입력에 얼마나 민감한가(dz/d visual)"라는 학습된 함수의 성질을 직접 재는 것이므로 메커니즘 해석엔 타당하다. gr1의 20%가 dexjoco의 6%보다 큰 것은 gr1이 시각 맥락에 더 의존하도록 학습됐음을 뜻함.

---

## 종합 판단 가이드 (어느 지표를 믿을까)

| 방법 | 신뢰도 | 비고 |
|---|---|---|
| ②′ frame-swap | ★★★ | confound 없음, v3=0 대조군. **1순위 증거** |
| ① R²(act→z) 격차 | ★★☆ | 깨끗함. 단 R²(vis→resAct) 열은 confounded라 무시 |
| ② near-dup 그룹 존재+프레임 | ★★★ | 스토리 그림. 산포 배율 신뢰, 상관계수는 보조 |
| ④ within-action task-ARI | ★★☆ | 좋은 아이디어지만 disjoint-action 데이터에선 역전됨 |
| ③ dist floor | ★☆☆ | 단독 오해 소지(차원/스케일 효과). ②와 함께만 |

**최종:** frame-swap(★★★) + near-dup 프레임 그림(★★★) + R²(act→z) 격차(★★☆)가 서로 일관되게 **gr1에서 가설을 지지**하고 dexjoco에서는 "표본 부재"로 관측 불가임을 보여준다. 발표에 쓴다면 **`neardup_groups.png`(스토리) + `frame_swap.png`(메커니즘) + TL;DR 표** 조합을 추천.

**재현:** `analysis/vsep_run.sh` (dexjoco). gr1은 동일 스크립트에 v3/v4 ckpt와 `--data-config fourier_gr1_arms_waist`, `--dataset-path .../gr1_unified.*`만 바꿔 실행. 결과는 `analysis/output/visual_sep/`(dexjoco), `analysis/output/visual_sep_gr1/`(gr1).
