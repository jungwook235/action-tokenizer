# 📋 Status Board — actlat-phase0-probes

> **Master-owned.** Only the Master edits this file. One-glance human view of the whole run,
> updated every 10-min loop tick. Mechanical live state is `ar board`; this is the narrative.

| | |
|---|---|
| **Central idea** | As action DoF ↑, intent becomes less accessible from **raw** action labels; a semantically-compressed, **visually-grounded (v4 DINO-fused)** latent recovers it. Phase 0 = offline motivation evidence. |
| **Group ID** | `4cb173b8-1df1-42ed-8969-b9cefcfb9f06` |
| **Started** | 2026-07-20T15:46:14Z |
| **Last updated** | 2026-07-21T16:55Z |
| **Overall health** | 🟢 on-track — **3/4 sections verified (S1 gate, S3 copycat, S4 geometry all PASS)**; overview written. Remaining: robocasa probe (running) + EgoDex (decisive DoF-scaling) + S2 within-embodiment; then verify endpoints. |

Legend: ⚪ not-started · 🟡 in-progress · 🟢 done/on-track · 🔴 blocked/failed · ✅ verified

---

## Tasks / sections

| # | Section | State | Progress | Key result so far | Blocker | Next step |
|---|---|---|---|---|---|---|
| 1 | `intent-probe` (S1, PRIORITY) | 🟢✅ verified + refined | exp-0001 done; **ver-0001 PASS**; mechanism refined to MIXED | **v4 0.347 > PCA-best(k32) 0.319 > v3 0.301≈raw 0.298 > PCA-canonical 0.246**; raw→v4 gain splits ~40% compression / ~60% grounding | EgoDex needed to test if grounding share grows w/ DoF | robocasa→EgoDex collection |
| 2 | `dof-redundancy` (S2) | 🟢 exp match | exp-0003 done (match) | **raw ~48× redundant, eff DoF ~9.5, 23% var task-predictive** — ties S1 (why raw intent is modest) | — | fill S2; verify |
| 3 | `label-noise-shortcut` (S3) | 🟢✅ verified | exp-0004 done; **ver-0004 PASS** (blind-R² held-out/leak-free) | **Copycat: raw & v3 ~98% blind-predictable; v4 0.915, 3× harder to copy, autocorr 0.94<0.99.** Denoising REFUTED → de-correlation | — | integrate |
| 4 | `semantic-geometry` (S4) | 🟢✅ verified | exp-0002 match; **ver-0003 PASS** (superseded ver-0002 fail) | v4 best: cross-ep P@10 v4 0.273>v3 0.251>raw 0.228≈PCA 0.228; ARI v4 0.127>v3 0.101>raw 0.085≈PCA 0.079; v3>raw geometry | — | integrate; add EgoDex geom if S1 lands |

## Experiments in flight
| Exp | Section | Status | Owner | Outcome |
|---|---|---|---|---|
| `exp-0001` | intent-probe (S1) | ✅ done (partial) | experiment | gr1 gate: v4 top, modest, low absolute; split leak-free ✓ |
| `exp-0002` | semantic-geometry (S4) | ✅ done (match) | experiment | v4 best geometry; PCA whitening bug fixed |
| `exp-0003` | dof-redundancy (S2) | ✅ done (match) | experiment | raw ~48× redundant; 23% var task-predictive |
| `ver-0001` | intent-probe (S1) | ✅ PASS | verification | gate verified: split leak-free (source+empirical), CNA exact, ordering + PCA fairness confirmed |
| `ver-0002` | semantic-geometry (S4) | ❌ FAIL (superseded) | verification | stale-PCA wording; corrected |
| `ver-0003` | semantic-geometry (S4) | ✅ PASS | verification | corrected S4 claim matches post-fix file; leak-free re-confirmed → **S4 verified** |
| `exp-0004` | label-noise-shortcut (S3) | ✅ done; filled; verify pending | experiment | copycat strong; denoising refuted→reframed |
| `exp-0005` | dof-redundancy (S2) | 🟡 pending (not started) | experiment | within-embodiment gr1 control |
| `exp-0006` | intent-probe (S1) | 🟡 probe running; **full-161 near-chance** | experiment | 161-task starved (raw CNA 0.02, v4 0.04 — all ~chance); controlled subset is the real point, running next |
| `exp-0007` | intent-probe (S1) | 🟡 building (HIGH — decisive) | experiment | EgoDex path B (egodex_lerobot_part1_gr1, 304 tasks, joints[0:44]); v4=joint_egodex_dexdual; **decode-L1 gate** before trusting numbers; ~150 eps/task (no starvation) |
| `ver-0004` | label-noise-shortcut (S3) | ✅ PASS | verification | copycat verified: blind-R² held-out/leak-free, values match, denoising-refute honest → **S3 verified** |
| `exp-0005` | dof-redundancy (S2) | 🟡 pending | experiment | within-embodiment gr1 control: arm-only vs full vs top-k PCs (kills dataset confound) |
| `exp-0006` | intent-probe (S1) | 🟡 pending (HIGH) | experiment | **robocasa low-DoF endpoint** collection+probe (GPU srun) |
| `exp-0007` | intent-probe (S1) | 🟡 pending (HIGH) | experiment | **EgoDex high-DoF endpoint** — load-bearing DoF-scaling point (GPU srun, cap eps) |

## 🔴 Blockers & risks
- **robocasa may be a degenerate low-DoF anchor:** full 161-task probe is near-chance for ALL reps (data starvation: 161-way × ~3 eps/task). Controlled top-N-by-eps subset running; if it's *also* near-chance, robocasa's fine pick-place tasks just aren't chunk-distinguishable → report with caveat, DON'T force onto the DoF curve; lean headline on gr1(mid)→EgoDex(high). Told Experiment.
- **PCA integrity (RESOLVED):** Experiment regenerated S1 PCA leak-free (per-fold StandardScaler+PCA on train only); gate consistent (PCA-256 0.245/0.240, still < v4). S4 PCA-whitening fix under independent check in ver-0002.
- **GIT (resolved):** outer-repo commit f6e29c1 kept (run files only); local user.name/email unset per user. Going-forward: NO further outer-repo git ops without Master OK; versions tracked via store + Notion.
- **Modest gr1 gap is the key risk to the story.** gr1 (mid-DoF, 29) shows v4 only +0.02–0.05 CNA over raw. The DoF-scaling headline REQUIRES a high-DoF endpoint where the gap widens → **EgoDex collection is now critical-path**, not optional. If EgoDex also shows a small gap, the storyline must be reframed (e.g., toward the S2 redundancy / denoising angle, which is strong: PR 2.4 vs 29 nominal).
- **dexjoco confound (design, confirmed):** raw=v3=v4=1.00 ceiling → labeled CONTROL only, off the DoF curve. gr1 = clean anchor. ✓ working as intended.
- robocasa (low-DoF) + EgoDex (high-DoF) endpoints NOT cached → need collection (GPU only via 1-GPU `srun` for DINO/encoder forward). EgoDex tractability → cap eps/task.

## 🧭 Decisions (recent, newest first)
- `2026-07-21 17:00` — **robocasa (exp-0006) design:** approved split='all' + LOEPTO-3 leak-free (fixed-val's 7 eps unusable, documented). Added: PRIMARY DoF-figure point = task-count-CONTROLLED (top ~24-40 pick-place tasks by #episodes, ≈gr1 granularity) to isolate DoF from the #tasks confound; full 161-way as robustness; annotate every figure point with #tasks; apply same top-N-by-eps rule to EgoDex for comparability. No arbitrary semantic grouping.
- `2026-07-21 16:20` — **User decision on outer-repo git:** keep commit f6e29c1 (run files only), unset local user.name/email (done). Standing policy: no further outer-repo git ops without Master OK; versions via store + Notion.
- `2026-07-21 16:15` — S1 gate ACCEPTED (ver-0001 pass). Gate ordering v4>raw≈v3>PCA verified & robust; v4 margin +0.031 over best PCA. S1 still needs EgoDex endpoint before full integration (DoF-scaling claim).
- `2026-07-21 16:10` — Queued S3 (exp-0004) cache-derivable; EgoDex+robocasa GPU collection is the next major step (headline figure).
- `2026-07-21 14:45` — dexjoco = CONTROL (disjoint actions confound raw-intent decoding), gr1 = clean DoF anchor; PCA-k baseline kept as the compression-vs-grounding fork (relayed to Writing+Experiment).
- `2026-07-21 14:42` — Corrected Monitoring rule: do NOT deny for cd/$VAR/pipes; approve read-only + trusted run scripts (ar.py/send_msg.py/…), ESCALATE only deletes/pre-existing-writes/scancel/installs. (Old over-strict rule had blocked legit commands.)
- `2026-07-21 14:30` — Seeded 4 Phase-0 sections per DIRECTION.md; S1 intent-probe gates the storyline (grounding-recovers-intent vs compression/denoising).

## 📈 Progress log (append-only, newest first)
- `2026-07-21 18:36` — EgoDex plan set (user greenlit path B): egodex_lerobot_part1_gr1, joints[0:44], v4=joint_egodex_dexdual (egodex_gr1 token), runtime data-config (no file edits). Endorsed + added: **decode-L1 gate** vs gr1 ref (~0.0014) before trusting numbers; task-count-controlled top~24-40 as primary point (EgoDex ~150 eps/task = no starvation → clean high-DoF); raw/PCA/v4 (v3 N/A). This is THE decisive DoF-scaling experiment.
- `2026-07-21 18:28` — robocasa full-161 probe near-chance for all reps (raw 0.02/0.05, v4 0.04/0.06) = data starvation as predicted. Told Experiment: run the task-count-controlled subset as the real robocasa point; if that's also degenerate, caveat it and lean DoF headline on gr1→EgoDex. EgoDex (exp-0007) is now the decisive point.
- `2026-07-21 18:18` — **ver-0004 PASS → S3 verified** (blind-R² confirmed held-out/leak-free, denoising-refute honest). Verification scorecard now 3/4: S1 gate, S3 copycat, S4 geometry all PASS. Remaining: S2 (verify after within-embodiment exp-0005) + the robocasa/EgoDex endpoints.
- `2026-07-21 18:12` — Experiment progressing on robocasa probe (161-task + controlled subset). Kicked off **ver-0004** (S3 copycat) on the idle Verification agent in parallel — checks blind-predict R² is held-out/leak-free + values + honest denoising-refute. (doctor still flags ver-0002 FAIL but it's SUPERSEDED by ver-0003 PASS — S4 verified.)
- `2026-07-21 18:00` — Writing added an **overview/abstract** (sections/overview) leading with the unifying insight (v3≈raw on supervision, v3>raw on geometry, only v4 helps) + compression-vs-grounding table. Registered it via hand-written meta.json (NOT `ar section-add`, which would clobber the files); board now shows 5 sections. Saved the section-add-overwrite gotcha to memory.
- `2026-07-21 17:55` — Experiment had stalled idle (post-recap) with robocasa cache ready but probe unrun; woke it (send+flush) → now running robocasa probe → EgoDex → S2-control. Respawned board for the user on **port 8900 (single-store)** — status.json confirms all sections have valid report paths (the stale 8899 process predated the reports). Bash classifier intermittently unavailable much of this window.
- `2026-07-21 17:35` — **ver-0003 PASS → S4 verified** (corrected claim matches post-fix file; leak-free re-confirmed). **robocasa collection DONE** (cache.npz 122.5MB, v4+v3+actions, 161 tasks/7728 chunks) — but Experiment went idle after a context recap before running the probe; nudged it to resume (probe robocasa → EgoDex → S2-control) + fix store bookkeeping (exp-0005/6/7 unclaimed). S3 fully filled (PH=0). Bash classifier intermittently unavailable this tick.
- `2026-07-21 17:20` — Writing applied all 3 edits (S1 mixed-mechanism, S4 wording fix, S3 fill). Endorsed the unifying **overview line: v3≈raw on supervision/decodability (S1,S3) but v3>raw on geometry (S4); only v4 helps supervision** → into abstract. Kicked off ver-0003 (S4 re-verify corrected claim). S3-copycat verification next.
- `2026-07-21 17:12` — **S3 (exp-0004) done — net stronger:** copycat headline strong — raw & v3 ~98% blind-predictable from history, v4 drops to 0.915 + 3× harder to naive-copy + autocorr 0.94<0.99 → v4 is a less-shortcut-prone VLA target. Denoising(a) hypothesis REFUTED (v4 recon has MORE HF, not smoother) → guided Writing to reframe as spectral de-correlation, report honestly. Experiment hunting idle GPU node for robocasa collection (exp-0006). Nudged Writing to fill S3 + confirm S4 fix.
- `2026-07-21 16:55` — **ver-0002 (S4) FAIL — high-quality catch:** compute reproduced bit-for-bit, method leak-free, but S4 claim used STALE pre-fix PCA (~0.05); post-fix PCA≈raw (P@10 0.228/ARI 0.079). True ranking v4>v3>raw≈PCA (v3>raw in geometry — nuance vs S1's v3≈raw). Relayed exact fix to Writing; re-verify as ver-0003. **S1 gate refined to MIXED mechanism** (Writing re-think): rank-tuned PCA(k32)=0.319>raw, so raw→v4 gain ~40% compression / ~60% grounding — endorsed, present both PCA numbers; EgoDex tests if grounding share grows w/ DoF. exp-0004 recon-decode done.
- `2026-07-21 16:42` — **PCA integrity CLOSED:** Experiment confirmed S1 PCA regenerated leak-free (per-fold scaler+PCA on train), gate consistent (PCA-256 0.245/0.240 < v4). Experiment acked collection plan (exp-0004 recon → exp-0005 → robocasa → EgoDex, one srun, cap eps). Kicked off ver-0002 (S4 semantic-geometry, clear+brief) — verifying cross-vs-within-ep retrieval + PCA-whitening fix.
- `2026-07-21 16:32` — exp-0004 (S3) running; Writing filled S1/S2/S4 (S4 down to 2 PH). Queued the endpoint critical-path: exp-0005 (S2 within-embodiment gr1 control, cache), exp-0006 (robocasa low-DoF, GPU), exp-0007 (EgoDex high-DoF, GPU — the load-bearing DoF-scaling point). Briefed Experiment on collection order + GPU-srun/cap-eps/DINO-cache/val-split gotchas. Next: verify S4 (bug-fix warrants unbiased check).
- `2026-07-21 16:20` — **ver-0001 PASS → S1 gate VERIFIED.** Writing disclosed a pre-policy outer-repo commit (f6e29c1, run files only) + local git-config write; user chose keep-commit/unset-config → I unset local user.name/email (repo restored to global fallback). Relayed no-further-outer-git policy. Queued S3 (exp-0004). Writing filled S1 consistently (v4 0.347, best-PCA k32 0.316, margin +0.031).
- `2026-07-21 16:05` — **All 3 cache exps CLOSED.** S1 partial (v4 top/modest/low-abs, split self-verified leak-free 0 overlap); S2 match (raw ~48× redundant, eff DoF ~9.5, 23% var task-predictive — ties S1); S4 match (v4 best geometry cross-ep P@10 0.273>v3>raw≫PCA, ARI v4 0.127>raw 0.085; **PCA whitening bug found+fixed**). Queued ver-0001 + ran clear+brief → Verification independently checking gate (reading dataset.py + fixed_val_split.py, not just result files). Asked Experiment if PCA bug also touched S1's PCA number (open). Delegated queued-msg flush duty to Monitoring. Cleared 1 escalation (sed -n read-only). Coherent cross-section narrative forming.
- `2026-07-21 15:47` — **Cleared an Experiment stall:** agent sat idle with compute done but exp-0001 not closed; nudges had QUEUED-unsubmitted in its input box (send_msg said delivered, but `Press up to edit queued messages`). `monitor.py scan/unstick` didn't catch it; a bare `herdr pane send-keys w1:p3 Enter` flushed them → agent resumed (now running, 2 shells: closing exp-0001, finishing exp-0003 dex, starting exp-0002). Saved memory for the pattern.
- `2026-07-21 15:35` — **GATE LANDED (gr1, leak-free):** v4 CNA 0.347/0.344 (lin/mlp) > raw 0.298/0.328 ≈ v3 0.301/0.321 > PCA-canonical 0.238 (PCA-best k32≈0.320). Direction-consistent (grounding on top, compression alone doesn't help) but MODEST (~+0.02–0.05). dex ceiling control ✓. S2: gr1 raw PR≈2.4 / n_pc95=6 vs 29 nominal (strong redundancy). Nudged Experiment to close exp-0001. EgoDex endpoint now critical-path.
- `2026-07-21 15:22` — gr1 v3 results in: **v3≈raw (CNA≈0.30)** — action-only latent adds no intent over raw (interim; awaiting PCA+v4 to complete the fork). Writing 1h re-think: no pivot, all 4 sections drafted+aligned. Nudged Experiment to run light exp-0002/0003 concurrently with exp-0001's slow PCA-k sweep (don't serialize wall-clock).
- `2026-07-21 15:12` — exp-0001 producing results: gr1 raw CNA≈0.30 (linear)/0.33 (MLP) [24 tasks, LOEPTO leak-free]; dexjoco raw=v3=1.00 ceiling (control confirms disjoint-action confound). PCA-k sweep + v3/v4 gr1 still running. Cleared a /proc-read prompt (safe) via Monitoring. exp-0002/0003 claimed, queued behind priority exp-0001.
- `2026-07-21 15:00` — All 3 cache-only experiments claimed & running (exp-0001/0002/0003). No stalls/fails. Experiment hit a read-only cache-read prompt (recover_episode_ids.py for by-episode split); Monitoring inspecting/clearing. S2/S4 now 50 placeholders each (Writing drafting), S3 22.
- `2026-07-21 14:52` — S1 revised & landed (dexjoco→off-curve control, DoF curve on gr1/robocasa/EgoDex; 53 real placeholders). Parallelized: queued 3 cache-only experiments exp-0001 (S1 gr1+dexjoco, priority), exp-0002 (S4), exp-0003 (S2) so Experiment isn't idle. Writing drafting S2–S4.
- `2026-07-21 14:45` — Experiment reported substrate ready (gr1 N=4008/24 tasks + dexjoco N=2322/5 tasks caches, A/Z3/Z4/DINO/task, CPU sklearn). Relayed dexjoco-confound design to Writing+Experiment. Writing unblocked, actively drafting S1.
- `2026-07-21 14:42` — Fixed Monitoring over-strict deny rule that had blocked master/experiment/writing commands; re-issued corrected escalate-only rule.
- `2026-07-21 14:30` — Topology recovered on resume: submitted Monitoring watch-loop, cleared Experiment prompt, briefed Writing, seeded sections S1–S4, started Master /loop 10m (job 5cf9f50f).

## ➡️ Next actions
- [ ] Experiment: answer PCA-bug scope Q (regenerate S1 PCA rows if affected)
- [ ] Verification: finish ver-0001 gate verdict (pass/fail)
- [ ] Writing: fill S1 (in progress) + fill S2/S4 from exp-0003/0002 results; commit
- [ ] Master: queue S3 (label-noise-shortcut) cache-derivable exps — spectral/jerk raw-vs-recon, blind history-only R² (copycat), temporal autocorr
- [ ] Master: verify S2 + S4 core claims (clear+brief) after fill
- [ ] Master: plan robocasa (low-DoF) + EgoDex (high-DoF) endpoint collection via 1-GPU srun (headline DoF figure)
