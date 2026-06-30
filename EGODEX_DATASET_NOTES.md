# EgoDex (part1~4) 데이터셋 정리

분석 대상 경로: `/workspace/project/datasets/egodex/{part1,part2,part3,part4}`
분석일: 2026-06-29
(분석 범위: `part1`~`part4` 내부만. `_resized.mp4`, `_inpainted.mp4`, `_inpainted.mp4.done`는 무시. 핵심은 `N.hdf5` + `N.mp4`이며, `N_mano.hdf5`(로봇 리타게팅 산출물)도 포함하여 정리)

---

## 1. 한눈에 보기 (요약)

- **데이터 성격**: Egocentric(1인칭) RGB 비디오 + 프레임 동기화된 **전신 + 양손 3D 스켈레톤(SE3)** + 자연어 task 설명으로 이루어진 human manipulation 데이터.
- **총 규모**: 86개 task, **239,265 에피소드** (part1~4).
- **비디오**: 1920×1080, **30 fps** (모든 part 일관).
- **에피소드 길이**: median 300프레임(10초), mean 488프레임(16초), 범위 16~4776프레임(0.5초~159초).
- **에피소드 1개 = `N.hdf5`(라벨) + `N.mp4`(비디오)** 한 쌍. 비디오 프레임 수와 hdf5 시퀀스 길이 T가 **1:1 일치**.

---

## 2. 전체 규모

| Part  | Task 수 | 에피소드 수 |
|-------|--------:|-----------:|
| part1 | 26      | 46,234     |
| part2 | 13      | 95,125     |
| part3 | 24      | 53,777     |
| part4 | 23      | 44,129     |
| **합계** | **86** | **239,265** |

- part2는 task 수는 적지만 `basic_pick_place`(27,419개) 등 대형 task 때문에 에피소드 수가 가장 많음.
- task별 편차가 매우 큼: 최소 `fry_bread`(7개) ~ 최대 `basic_pick_place`(27,419개).

---

## 3. Task별 에피소드 개수

### part1 (26 tasks, 46,234 episodes)

| Task | Episodes |
|------|---------:|
| add_remove_lid | 2,565 |
| arrange_topple_dominoes | 646 |
| assemble_disassemble_legos | 3,653 |
| assemble_disassemble_soft_legos | 704 |
| assemble_disassemble_structures | 614 |
| assemble_disassemble_tiles | 601 |
| assemble_jenga | 669 |
| boil_serve_egg | 487 |
| braid_unbraid | 1,050 |
| build_unstack_lego | 1,206 |
| charge_uncharge_airpods | 5,687 |
| charge_uncharge_device | 6,217 |
| clean_cups | 2,968 |
| clean_surface | 4,323 |
| clean_tableware | 2,758 |
| clip_unclip_papers | 403 |
| color | 1,998 |
| crumple_flatten_paper | 457 |
| deal_gather_cards | 1,502 |
| declutter_desk | 1,140 |
| dry_hands | 3,869 |
| fidget_magnetic_spinner_rings | 1,075 |
| flip_coin | 179 |
| flip_pages | 1,436 |
| fry_bread | 7 |
| fry_egg | 20 |

### part2 (13 tasks, 95,125 episodes)

| Task | Episodes |
|------|---------:|
| assemble_disassemble_furniture_bench_chair | 8,856 |
| assemble_disassemble_furniture_bench_desk | 5,795 |
| assemble_disassemble_furniture_bench_drawer | 7,505 |
| assemble_disassemble_furniture_bench_lamp | 3,484 |
| assemble_disassemble_furniture_bench_square_table | 8,489 |
| assemble_disassemble_furniture_bench_stool | 8,611 |
| basic_fold | 7,536 |
| basic_pick_place | 27,419 |
| fold_stack_unstack_unfold_cloths | 1,618 |
| fold_unfold_paper_basic | 1,686 |
| fold_unfold_paper_origami | 925 |
| insert_remove_furniture_bench_cabinet | 8,309 |
| insert_remove_furniture_bench_round_table | 4,892 |

### part3 (24 tasks, 53,777 episodes)

| Task | Episodes |
|------|---------:|
| gather_roll_dice | 1,530 |
| insert_dump_blocks | 982 |
| insert_remove_airpods | 4,177 |
| insert_remove_bagging | 1,068 |
| insert_remove_bookshelf | 3,117 |
| insert_remove_cups_from_rack | 1,773 |
| insert_remove_drawer | 1,591 |
| insert_remove_plug_socket | 4,836 |
| insert_remove_shirt_in_tube | 1,914 |
| insert_remove_tennis_ball | 1,351 |
| insert_remove_usb | 5,363 |
| insert_remove_utensils | 3,679 |
| knead_slime | 1,858 |
| load_dispense_ice | 713 |
| lock_unlock_key | 2,004 |
| make_sandwich | 1,692 |
| measure_objects | 842 |
| open_close_insert_remove_box | 1,213 |
| open_close_insert_remove_case | 2,367 |
| open_close_insert_remove_tupperware | 4,023 |
| paint_clean_brush | 1,894 |
| peel_place_sticker | 992 |
| pick_up_and_put_down_case_or_bag | 1,235 |
| play_piano | 3,563 |

### part4 (23 tasks, 44,129 episodes)

| Task | Episodes |
|------|---------:|
| pick_place_food | 4,782 |
| play_mancala | 1,981 |
| play_reset_connect_four | 2,091 |
| point_and_click_remote | 2,144 |
| pour | 3,205 |
| push_pop_toy | 2,801 |
| put_away_set_up_board_game | 559 |
| put_in_take_out_glasses | 2,688 |
| put_toothpaste_on_toothbrush | 1,928 |
| rake_smooth_zen_garden | 1,955 |
| roll_ball | 1,286 |
| scoop_dump_ice | 6,360 |
| screw_unscrew_allen_fixture | 1,096 |
| screw_unscrew_bottle_cap | 2,868 |
| screw_unscrew_fingers_fixture | 1,600 |
| set_up_clean_up_chessboard | 227 |
| setup_cleanup_table | 537 |
| sleeve_unsleeve_cards | 349 |
| slot_batteries | 1,172 |
| sort_beads | 1,156 |
| staple_paper | 522 |
| stock_unstock_fridge | 2,263 |
| sweep_dustpan | 559 |

---

## 4. 에피소드 파일 구성

각 task 폴더(`partX/<task_name>/`) 안에는 에피소드 인덱스 `N`마다 다음 파일들이 있음:

| 파일 | 사용 | 설명 |
|------|------|------|
| `N.hdf5`         | ✅ 핵심 | 원본 라벨 (카메라/전신+양손 스켈레톤 transforms, confidence, 메타데이터) |
| `N.mp4`          | ✅ 핵심 | 원본 egocentric RGB 비디오 (1920×1080, 30fps) |
| `N_mano.hdf5`    | ✅ 사용 | MANO 정제 손 keypoint + **로봇 리타게팅 state(GR1/Allex)** |
| `N_resized.mp4`  | ⛔ 무시 | 리사이즈 비디오 |
| `N_inpainted.mp4`| ⛔ 무시 | 손/팔 인페인팅 비디오 |
| `N_inpainted.mp4.done` | ⛔ 무시 | 인페인팅 완료 플래그 |

> **에피소드 개수 집계 기준**: `^\d+\.hdf5$` 패턴 파일 수 (`_mano.hdf5` 제외).

---

## 5. 비디오 (`N.mp4`)

- 해상도 **1920×1080**, **30 fps** — 모든 part에서 일관됨.
- 프레임 수 = hdf5의 시퀀스 길이 T 와 **1:1 일치** (예: 288프레임 ↔ 9.6초).

샘플 검증:
```
part1/braid_unbraid/0.mp4    -> 1920x1080, 30/1, 1627 frames
part2/basic_pick_place/0.mp4 -> 1920x1080, 30/1,   88 frames
part3/play_piano/0.mp4       -> 1920x1080, 30/1,  300 frames
part4/pour/0.mp4             -> 1920x1080, 30/1,   71 frames
```

---

## 6. 에피소드 길이(프레임 수) 분포

687개 에피소드 샘플(task당 8개) 기준:

| 지표 | 프레임 | 시간(@30fps) |
|------|-------:|-------------:|
| min    | 16   | 0.5s  |
| p5     | 72   | 2.4s  |
| p25    | 169  | 5.6s  |
| median | 300  | 10.0s |
| mean   | 488  | 16.3s |
| p75    | 520  | 17.4s |
| p95    | 1611 | 53.7s |
| max    | 4776 | 159.2s |

→ 대부분 수 초~10여 초, 꼬리가 길어 1~2분짜리 장기 에피소드도 존재.

---

## 7. `N.hdf5` 내부 구조

T = 프레임 수 (= 비디오 프레임 수, 예시 파일은 T=288).

### 7-1. Root attributes (메타데이터)

| attr | 예시 값 | 설명 |
|------|---------|------|
| `task` | `Add Remove Lid` | task 이름 |
| `llm_description` | `Add lids onto four cups placed on a wooden table...` | 자연어 instruction |
| `llm_description2` | `Remove lids from four cups...` | reversible task의 두 번째 방향 instruction |
| `which_llm_description` | `1` | 해당 에피소드가 두 description 중 어느 방향인지 |
| `llm_objects` | `['cup' 'lid']` | 등장 객체 |
| `llm_verbs` | `['add' 'remove']` | 동작 동사 |
| `llm_type` | `reversible` | task 유형 (reversible 등) |
| `object` | `object:cup, object:lid, number:4` | 객체/개수 정보 |
| `environment` | `table:wood, position:sitting, background:red` | 환경 정보 |
| `session_name` | `2025-02-20_14-09-09.mov` | 원본 세션(녹화) 식별자 |
| `annotated` | `True` | 어노테이션 여부 |
| `annotator_version` | `0.2` | 어노테이터 버전 |
| `extra` | `C5` | 기타 |

### 7-2. Datasets

```
camera/
  intrinsic                (3, 3)        float32   # 카메라 내부 파라미터
transforms/                              # 69개 키 = camera 1 + pose joint 68
  camera                   (T, 4, 4)     float32   # 카메라(=머리) SE(3) pose
  hip                      (T, 4, 4)     float32
  spine1 ~ spine7          (T, 4, 4)     float32   # 척추 7관절
  neck1 ~ neck4            (T, 4, 4)     float32   # 목 4관절
  {left,right}Shoulder     (T, 4, 4)     float32
  {left,right}Arm          (T, 4, 4)     float32
  {left,right}Forearm      (T, 4, 4)     float32
  {left,right}Hand         (T, 4, 4)     float32   # 손목
  {left,right}<Finger>...  (T, 4, 4)     float32   # 양손 손가락 풀 스켈레톤
confidences/                             # transforms의 각 joint(camera 제외)에 대응
  <joint_name>             (T,)          float32   # 프레임별 confidence
```

**`transforms/` 구성 = camera(1) + pose joint(68), 각각 `(T, 4, 4)` SE(3) 변환행렬**

손가락 joint 명명 규칙 (양손 × 5손가락):
`{side}{Finger}Metacarpal → Knuckle → IntermediateBase → IntermediateTip → Tip`
- side: `left` / `right`
- Finger: `Thumb`, `Index`, `Middle`, `Ring`, `Little`

`confidences/`는 `transforms/`의 각 joint(`camera` 제외)에 대응하는 `(T,)` 신뢰도 값.

---

## 8. `N_mano.hdf5` 구조 — MANO 정제 손 + 로봇 리타게팅

`N.hdf5`가 **원본 트래킹(전신+양손 SE3 스켈레톤)** 이라면, `N_mano.hdf5`는 그것을
**MANO 손 모델로 정제 + 로봇 액션 공간으로 리타게팅(retargeting)** 한 산출물.
(별도 파이프라인 `egodex_mano_optim` 산출물로 추정. root attrs 비어 있음. T = 프레임 수.)

```
camera/
  intrinsic              (3, 3)        float32   # 내부 파라미터 (fx=fy=736.6, cx=960, cy=540)
  poses                  (T, 4, 4)     float64   # 프레임별 카메라(머리) extrinsic. 시간에 따라 미세 변동
left_hand/  right_hand/
  joints_21              (T, 21, 3)    float64   # MANO 표준 21 keypoint 3D, **wrist 기준 상대좌표**
  wrist_se3              (T, 4, 4)     float64   # 손목 자체의 SE(3) global pose
gr1_state                (T, 44)       float32   # Fourier GR1 휴머노이드 리타게팅 joint state (관절각 rad)
allex_state              (T, 48)       float32   # Allex 로봇 리타게팅 joint state (관절각 rad)
```

### 8-1. 손 (MANO)
- `joints_21` `(T,21,3)`: **wrist(joint 0)가 항상 원점 `[0,0,0]`** → 손목 기준 상대좌표(root-relative).
  값 범위가 미터가 아닌 정규화된 손 모델 단위로 보임(관측 범위 약 -3.7 ~ 6). 21 = 손목 1 + 5손가락 × 4관절.
- `wrist_se3` `(T,4,4)`: 손목의 global SE(3) pose. translation 범위 ~0.3 ~ 0.6 (미터로 추정).
- → **손의 global pose는 `wrist_se3`, 손가락 articulation은 `joints_21`** 로 분리 저장.

### 8-2. 로봇 리타게팅 state ⭐
human 비디오를 **로봇이 바로 실행 가능한 action(관절 시퀀스)으로 변환**해둔 부분.
action_tokenizer의 `gr1` embodiment과 직접 연결될 수 있는 가장 유용한 데이터.
- `gr1_state` `(T,44)`: Fourier **GR1** 휴머노이드용. 관측 관절각 범위 약 -2.3 ~ 1.6 rad. 일부 dim은 0 고정(미사용 관절).
- `allex_state` `(T,48)`: **Allex** 로봇용 48-DoF 관절각.
- ⚠️ 정확한 dim별 관절 매핑(layout)은 hdf5에 명시돼 있지 않음 → 리타게팅 파이프라인(`egodex_mano_optim`) 코드 확인 필요.

---

## 부록: 재현용 명령

에피소드 개수 집계:
```python
import os, re
pat = re.compile(r'^\d+\.hdf5$')
for part in ['part1','part2','part3','part4']:
    for t in sorted(os.listdir(part)):
        d = os.path.join(part, t)
        if not os.path.isdir(d): continue
        n = sum(1 for f in os.listdir(d) if pat.match(f))
        print(part, t, n)
```

hdf5 구조 확인:
```python
import h5py
with h5py.File('part1/add_remove_lid/0.hdf5','r') as f:
    print(dict(f.attrs))
    f.visititems(lambda n,o: print(n, getattr(o,'shape',''), getattr(o,'dtype','')))
```

비디오 정보:
```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 part1/add_remove_lid/0.mp4
```
