# S3 · Raw actions are a copycat-prone supervision signal; the grounded latent resists it

## TL;DR
- **Main result:** raw action chunks are **~98% predictable from their own recent history alone** (blind ridge R² **0.980**) — a textbook copycat / causal-confusion shortcut; the action-only v3 latent does **not** help (R² 0.982), only the DINO-grounded **v4 reduces it to 0.915** and makes naive imitation **3× harder** (copy-NMSE 0.104 vs raw 0.035).
- **Why it matters:** a target you can copy from history teaches a policy to extrapolate motion instead of reading the scene — so v4 is a strictly better *supervision target*, independent of the DoF/intent argument.
- **Honest sub-finding:** the tokenizer does **not** denoise (reconstruction is near-lossless, L1 0.0014, no smoothing) — instead the v4 *latent* is spectrally whiter (low-freq power 0.387 vs raw 0.600), i.e. it **de-correlates** rather than smooths.

---

Motivated by the copycat problem (Wen et al., "Fighting Copycat Agents", NeurIPS 2020) and causal
confusion (de Haan et al., NeurIPS 2019): highly self-predictable action targets let policies
shortcut. gr1 cache, CPU; dex shown as reference. See **S1 §Shared setup**.

## (b) Copycat / blind predictability — the headline

Predict each step from **action history alone** (ridge, 2-step history, no observation). High R²
or low copy-error = easy to shortcut.

| gr1 target | blind ridge R² (↓ better) | naive-copy NMSE (↑ better) | AR(1) R² |
|---|---|---|---|
| raw action | 0.980 | 0.035 | 0.977 |
| v3 latent | 0.982 | 0.020 | 0.981 |
| **v4 latent** | **0.915** | **0.104** | **0.886** |

**Reading:** raw actions are ~98% blindly predictable — a policy can regress the next action from
the past two and ignore the scene. The learned **action-only v3 does not fix this** (R² 0.982,
and it is *more* trivially copyable, NMSE 0.020 < raw 0.035) — compression alone doesn't break the
shortcut. Only the **grounded v4** departs: blind R² drops to 0.915 and naive copying is **3×
harder** (NMSE 0.104). The reduction in ridge R² is modest (raw is still highly structured in
time), but v4 is unambiguously the *least* copyable target — so a head trained against v4 cannot
lean as hard on motion extrapolation. (dex reference: raw R² 0.993 → v4 0.986; same direction,
smaller gap.)

<details>
<summary><b>Reproducibility — blind predictability (exp-0004)</b></summary>

- **Script:** `analysis/label_noise_signals.py` (ridge history→step, 2-step history, α=1; naive-copy NMSE; AR fits), gr1 `cache.npz`, 56112 transitions.
- **Seed:** 0 · branch `master`, commit `f6e29c1` (dirty) · Raw: `.autoresearch/results/exp-0004/cacheonly_gr1.json` (`signals.*.blind_predict/copycat`).
- ✅ Core copycat claim independently verified — **ver-0004 PASS**.
</details>

## (c) Temporal autocorrelation

| gr1 target | lag-1 | lag-4 |
|---|---|---|
| raw action | 0.989 | 0.963 |
| v3 latent | 0.990 | 0.965 |
| v4 latent | 0.941 | 0.896 |

v4 is the only representation with visibly lower temporal autocorrelation — consistent with (b):
less step-to-step redundancy means less to copy. v3 tracks raw exactly.

<details>
<summary><b>Reproducibility — autocorrelation (exp-0004)</b></summary>

- Same script; lag-1…8 per signal in `.autoresearch/results/exp-0004/cacheonly_gr1.json` (`signals.*.autocorr`).
</details>

## (a) Not denoising — de-correlation *(honest reframing of the pre-registered sub-hypothesis)*

We pre-registered a *denoising* claim (tokenizer low-passes jitter). **The data refutes it:** the
tokenizer reconstructs the raw action near-losslessly and is **not** a smoother.

| gr1 | HF-energy fraction | jerk | decode L1 vs GT |
|---|---|---|---|
| raw action | 0.180 | 0.0183 | — |
| v4 reconstruction | 0.280 (×1.56) | 0.0183 (×1.00) | 0.0014 |

Reconstruction HF is *higher*, not lower, and jerk is unchanged — no smoothing. But the effect is
real on a different axis: the v4 **latent** sequence has a **whiter spectrum** — low-frequency
power fraction **0.387 vs raw 0.600**, spectral flatness **0.787 vs 0.499**. So v4 removes the
trivially-predictable low-frequency *drift* and redistributes content across frequencies — it
**de-correlates** the temporal signal (which is exactly why it is less copyable in (b)), rather
than denoising it. We report this honestly: high-fidelity representation + temporal de-correlation,
not smoothing.

<details>
<summary><b>Reproducibility — spectral / reconstruction (exp-0004)</b></summary>

- **Script:** `analysis/label_noise_recon.py` (decode μ → HF fraction, jerk, L1 vs GT) + latent spectral stats in `cacheonly_gr1.json` (`signals.*.spectral`). Raw: `.autoresearch/results/exp-0004/recon_gr1.json`.
</details>

## Takeaway
The recurring theme across S1–S4 sharpens here: **raw ≈ v3 on every *supervision* axis** — both
are ~98% self-predictable, highly autocorrelated, low-frequency-dominated, and weakly
intent-decodable (S1) — so the learned *action-only* compression buys nothing for supervision
quality (it only reorganizes geometry, S4). The **grounded v4** is the single representation that
departs: least copyable, least autocorrelated, spectrally whitest, best semantic geometry (S4). A
VLA/diffusion head trained against v4 is therefore harder to shortcut and forced to use
observations — the supervised-learning pillar of the universal-tokenizer motivation.
