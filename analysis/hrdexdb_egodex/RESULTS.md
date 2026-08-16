# HRDexDB → EgoDex 토크나이저 latent 분석 기록

대상 토크나이저: `joint_soupv1_v4_recon_dino_bn64_l1_mse_naiveln_vae_embtok/checkpoint-400000`,
`embodiment_id = egodex_naivekey` (57-dim encoder, `_class_token_id__egodex_naivekey = 1`).
VAE는 `vae_sample_override=False` → 사후평균 μ. latent = `[16, 64]`.
전부 CPU에서 실행 (chunk당 약 0.9초, DINOv2-large 2장 forward가 지배적).

---

## 1. HRDexDB MANO ↔ EgoDex action 대응 검증

`check_mano_egodex_correspondence.py`

토크나이저 입력은 `human_egodex_camera_hand_unit`의 57-dim:
`camera_pos(3) camera_rot6(6) | rightHand_pos(3) rightHand_rot6(6) right{Thumb,Index,Middle,Ring,Little}Tip_pos(15) | left…(24)`,
`meta/stats.json` min/max로 [-1,1] 정규화, 16-step chunk, 30 fps.

| EgoDex dim | HRDexDB 소스 | 검증 결과 |
|---|---|---|
| `rightHand_pos` | `hand/mano_params/*.json` → `joints[0]` | 둘 다 미터, 손목 원점 차이 ≲ 6 mm |
| `rightHand_rot` | `global_orient` (3×3) | MANO 손목 프레임 ≈ ARKit hand 프레임, **잔차 회전 9.0°** (4개 finger MCP Kabsch, RMS 5.9 mm) |
| `right*Tip_pos` ×5 | `joints[[4,8,12,16,20]]` | 순서(thumb→little) 완전 일치, 정렬 후 손끝 차이 19–32 mm |
| `camera_pos/rot` | `cam_param/ego_calib.json` 헤드캠 per-frame `[R\|t]` | 카메라 규약 동일 (OpenCV, world-from-cam) |
| `left*` (24 dim) | **없음** (HRDexDB human은 오른손 MANO 1개) | — |

부수적으로 확인한 사실:
- MANO joint 순서 = manopth/OpenPose 순서 (wrist, thumb, index, middle, ring, pinky × 4).
  체인 길이(중지 최장 0.167 m, 소지 최단 0.132 m) + MANO tip vertex(745/317/444/556/673 → joint 4/8/12/16/20, 오차 1.5–5 mm) + 영상 재투영으로 확인.
- **오른손**만 존재 (ego view에서 팔이 우하단 진입, 샘플 22 에피소드 전부 동일).
- EgoDex의 rot6d = **R의 처음 두 열** (이 가정에서만 손목 국소좌표계의 knuckle 위치가 std ≤ 1 mm로 상수).
- EgoDex `camera_rot` = world-from-cam, OpenCV 축 (x 오른쪽, y 아래, z 광축).
- 월드 프레임: EgoDex는 y-up·바닥 원점 / HRDexDB는 **up = −z**, 원점이 머리 높이 부근.
- human 441 에피소드 / 94 objects / 113,680 프레임 / 30 fps. `ego_calib.json`은 **413/441**만 보유.

강체 정렬 후 EgoDex min-max 범위 커버리지: camera_pos·rightHand_pos·rightHand_rot·5 tips = 1.000,
**camera_rot = 0.816** (HRDexDB 헤드캠 pitch 46–71° vs EgoDex 16–37° → 진짜 도메인 차이).

---

## 2. human ↔ robot 페어 latent 거리 분석

`pair_latent_distance.py` (추출) → `pair_latent_report.py` (거리/그림/표)

### 2.1 설정

- 페어 소스: 로봇 `grasp_result.json`의 `human_paired_episode`, `grasp_success: true`만.
  사용 가능 페어 239개 / 60 objects. 이 실행은 **30 objects × 2 pairs**.
- chunk: `object_6d_pose_v2.npz`로 물체가 2 cm 이상 움직인 첫 프레임 = lift onset.
  `approach = [onset−16, onset)`, `lift = [onset, onset+16)`. → **236 chunk (robot 116 / human 120)**.
  (240에서 4개 탈락: onset < 16. 그 결과 human (obj,pair) 조합 60 vs robot 58 — 완전 대칭 아님.)
- **대칭 매핑** (양쪽 동일 인코더):

| 57-dim 블록 | human | robot |
|---|---|---|
| `rightHand_pos/rot` | MANO 손목 (+9° `R_fix`) | arm EE 플랜지 pose (`raw/arm/action` 4×4 → `C2R` → 영상 프레임 시각 리샘플) |
| 손끝 15 | **0** | **0** (Inspire 핸드 FK 불가 → 대칭 유지) |
| 왼손 24 | 0 | 0 |
| camera 9 + DINO 프레임 | 공통 정적 카메라 `22641005` | 동일 카메라 id |

- 월드 프레임: 세션마다 카메라 리그가 물리적으로 옮겨졌으므로(공통 20대 카메라 중심 rigid fit RMS 38 cm 실패)
  **오브젝트 기준** 정렬 — 원점 = 물체 정지 위치, up = −z, yaw = 카메라→물체 방향, 물체를 `(0, 0.80, −0.45)`에 배치.
- **플랜지→핸드 마운트 캘리브레이션** (`fit_mount`): 전역 human 평균 손 프레임과 전역 robot 평균 EE 프레임을
  맞추는 상수 1개. 이번 실행 값 = **회전 102.0°, 오프셋 120 mm** `[-0.104, 0.041, -0.045]`.
  물체·페어 라벨을 쓰지 않으므로 same/diff 구조를 만들 수 없음.
- 두 변종을 같은 DINO 특징으로 인코딩: `wrist` (camera dim도 0) / `wristcam` (camera dim 유지).
- 거리 = 1024차원(=16×64) flatten 후 cosine distance. target은 항상 **phase 일치**.

### 2.2 그룹 정의 (anchor = 로봇 chunk 116개)

| 그룹 | 정의 | anchor당 target |
|---|---|---|
| H-same | 같은 물체 + 같은 pair의 human ep | 1 |
| R-same | 같은 물체 + 다른 pair의 robot ep | 1 |
| R-diff | 다른 물체의 robot ep | ~58 (또는 pair 일치 29 / 무작위 1) |
| H-diff | 다른 물체의 human ep | ~58 (동일) |

HRDexDB에는 **물체를 넘나드는 대응 관계가 없다.** 다만 분석 풀의 human chunk는 전부 페어링된
에피소드이므로 H-diff도 "다른 물체의 페어링된 human ep"이다.

### 2.3 거리 (평균 / 중앙값) — `out/distance_tables.md`가 재생성 소스

**latent (wrist only), cosine**

| H-diff 정의 | H-same | H-diff | R-same | R-diff | anchor별 (H-same − H-diff) |
|---|---|---|---|---|---|
| 다른 물체 전부 | 0.0456 / 0.0384 | 0.0877 / 0.0732 | 0.0628 / 0.0398 | 0.0951 / 0.0802 | −0.0422 |
| pair-index 일치 | 0.0456 / 0.0384 | 0.0876 / 0.0729 | 0.0628 / 0.0398 | 0.0948 / 0.0796 | −0.0420 |
| 무작위 1개 | 0.0456 / 0.0384 | 0.0948 / 0.0840 | 0.0628 / 0.0398 | 0.1006 / 0.0809 | −0.0493 |

**raw action (wrist only), cosine**

| H-diff 정의 | H-same | H-diff | R-same | R-diff | anchor별 |
|---|---|---|---|---|---|
| 다른 물체 전부 | 0.0757 / 0.0510 | 0.2241 / 0.1695 | 0.1667 / 0.0929 | 0.2618 / 0.1915 | −0.1483 |
| pair-index 일치 | 0.0757 / 0.0510 | 0.2239 / 0.1718 | 0.1667 / 0.0929 | 0.2601 / 0.1915 | −0.1481 |
| 무작위 1개 | 0.0757 / 0.0510 | 0.2081 / 0.1366 | 0.1667 / 0.0929 | 0.2403 / 0.1938 | −0.1323 |

**실제 손목 궤적 거리 (cm)** — 해석 기준선

| H-diff 정의 | H-same | H-diff | R-same | R-diff | anchor별 |
|---|---|---|---|---|---|
| 다른 물체 전부 | 7.35 / 6.05 | 12.96 / 12.13 | 9.64 / 8.71 | 14.10 / 13.47 | −5.62 |

**latent (wrist + camera), cosine** (참고): H-same 0.1067 / H-diff 0.1205 / R-same 0.0231 / R-diff 0.0386.

### 2.4 비율 지표

| | 같은 행동이 더 가까움 (cross-emb) | (within-emb) | top-1 retrieval (human 60개 중) |
|---|---|---|---|
| latent (wrist only) | **87.1%** | 76.8% | **16.4%** (chance 1.7%) |
| latent (wrist+camera) | 81.9% | 83.0% | 7.8% |
| raw action (wrist only) | 90.5% | 77.7% | 17.2% |

"같은 행동이 더 가까움" = anchor별로 `d(H-same)` 1개와 `d(H-diff)` 평균을 비교해 전자가 작은 anchor 비율.

H-diff 정의를 바꿔도 latent의 cross-emb 승률은 86–87%로 안정. 단 **latent vs raw 우열은 프로토콜 의존**:
여러 target 평균과 비교 시 raw 우세(90.5% vs 87.1%), 무작위 1개와 1:1 비교 시 latent 우세(86.2% vs 80.2%).

### 2.5 결론

1. **같은 행동 pair가 다른 행동보다 약 2배 가깝다** (latent 0.046 vs 0.088). H-diff 정의에 무관하게 성립.
2. **그룹 순서가 실제 물리 거리 순서와 일치**:
   물리 `H-same 7.4cm < R-same 9.6cm < H-diff 13.0cm < R-diff 14.1cm`,
   latent `0.046 < 0.063 < 0.088 < 0.095`. → latent이 새로 만든 구조가 아니라 입력 구조를 보존한 것.
3. **대비 폭은 raw가 더 크다** (H-diff/H-same: 물리 1.76배, latent 1.92배, raw 2.96배).
   → 이 설정에서 latent이 행동 구분을 더 선명하게 만들지는 못한다.
4. **camera 9 dim을 넣으면 나빠진다** (87.1% → 81.9%, top-1 16.4% → 7.8%). 세션별 캘리브레이션이
   "촬영 세션 지문"으로 작동해 embodiment 덩어리를 되살림 (centroid 거리 0.092, 1-NN 같은 embodiment 100%).
   → `wrist only`가 주 결과.
5. wrist-only 변종에서는 embodiment 분리가 거의 사라짐 (centroid 거리 0.0054, 1-NN 같은 embodiment 97.5%).

### 2.6 도중에 폐기한 주장 (기록용)

- **마운트 보정 전 결과 전부 무효.** 보정 전에는 cross-emb 승률 44.8%(동전 던지기 이하), top-1 4.3%였고
  "latent이 embodiment만 encode한다"고 결론냈으나, 이는 human 손목 프레임과 robot 플랜지 프레임이
  **102° 어긋난** 것(실제 행동에 의한 회전 변화폭 26–32°의 3배 이상)이 만든 아티팩트였다.
- **"H-same < R-same" 주장 철회.** 평균은 그렇게 보이지만 R-same 분포의 두꺼운 꼬리(p90 0.153 vs 0.084)
  때문이고, 중앙값은 0.0384 vs 0.0398로 무승부, anchor별로는 R-same이 더 가까운 경우가 55.4%로 다수.
  물리 거리에서는 H-same(6.05 cm) < R-same(8.71 cm)이 확실하므로, **latent은 이 둘을 구별하지 못한다**가 맞다.
- **raw 베이스라인 조건 오류 수정.** 처음 보고한 raw(89.7% / top-1 14.7%)는 camera dim이 살아있어
  wrist-only latent과 조건이 달랐다. 조건을 맞춘 값은 90.5% / 17.2%.

### 2.7 한계

- 로봇 손끝 15 dim이 비어 action 쪽엔 손목 궤적(9 dim)만 있다. 결론 3의 "latent ≈ raw"는 이 저차원성 탓일
  가능성이 크다. → **Inspire 핸드 FK로 손끝 채우기가 개선 1순위** (`/data/junghun/rrc-release/rrc/robots/assets/inspire_rh56dfq`,
  실제 하드웨어는 dftp; 액추에이터 0–1000 → 관절각 캘리브레이션 가정 필요).
- "같은 행동" = 같은 물체. apple/orange/lemon처럼 실제로 같은 grasp인 경우가 "다른 행동"으로 세어지므로
  87%는 과소평가일 수 있다.
- 마운트 보정(102.0°, 120 mm)은 데이터 추정치. Inspire 마운트 도면 값으로 대체 검증이 필요하다.
- 236 chunk / 30 objects / 2 phase. 토크나이저는 로봇 팔과 이 정적 카메라를 학습에서 본 적 없는
  **zero-shot OOD probe**.

---

## 3. 파일

| 파일 | 내용 |
|---|---|
| `check_mano_egodex_correspondence.py` | 1절 검증 + `hrdex_to_egodex57()` 매핑 |
| `extract_latent_cpu_smoketest.py` | 에피소드 1개 CPU latent 추출 smoke test |
| `pair_latent_distance.py` | 페어 수집 → 마운트 캘리브 → 57-dim → DINO → latent, `out/pair_latents.npz` |
| `pair_latent_report.py` | 4그룹 거리·비율 지표, `out/pair_latent_distance.png`, `out/distance_tables.md` |
| `out/pair_latents.npz` | latent(2변종) / raw / 라벨(obj, pair, kind, phase, ep) |

재현: `conda activate gr00t-actlat` → `python analysis/hrdexdb_egodex/pair_latent_distance.py 30 2`
(CPU 약 6분) → `python analysis/hrdexdb_egodex/pair_latent_report.py`.
