# Plan: BCI proxies that generalise across users and sessions

## Question

Which EEG features of a single trial distinguish the three PROTEUS tasks
(motor imagery, mental subtraction, word generation) **for every user**, not
just on average? The search covers ~50 000 classic BCI and non-BCI features.
A proxy counts as general when it ranks among a user's best features for most
users.

## Data

The official Track 02 windows, the same as bci-erds (`official.npz`):
2655 windows, 41 channels, 4 s (cue +1 s to +5 s), 120 Hz, 0.1–75 Hz.
The official split and random state are **not** used here.

**User = subject × session.** That gives 14 users: P04 and P10 have 3 sessions
each, the other 8 subjects 1. Each user has about 190 trials (≈63 per class)
and two interfaces (Graz, BrainHero).

## Protocol

1. **Best features per user** (model-free, that user's trials only): 3-class
   ANOVA F, and per class Cohen's d of class k vs rest. Each user ranks every
   feature.
2. **Generality** = the number of users (out of 14) in whose top k a feature
   sits (k = 50, 100, 500). For the per-class version the d must also have the
   majority sign. Output: `generality.csv`.
3. **Does it transfer?** Leave-one-user-out XGBoost, 14 folds, three models:
   - `all`: every feature, with an F-test filter (top 2000) on the training users;
   - `general`: the 100 most general features, counted on the training users only;
   - `pooled`: the 100 best by pooled F. This tests "general" against "strong on
     average".
   Scored with the official metric (balanced accuracy per session × interface,
   averaged) on the held-out user. The SHAP of `all` on the held-out user gives
   a model-based top-k count as a second view.
4. `--strict-subject`: hold out each user's other sessions too. This is the
   cross-subject version. With sessions as users, the P04-s1 fold otherwise
   still trains on P04-s2/s3. Report both.

## Feature families (`proxies/features/`), ~50 000 deployable columns

| group | family | what | columns |
|---|---|---|---|
| BCI | spectral | log / relative power ×7 bands, ratios, edge, centroid, alpha peak, 1/f slopes; channel + region; L–R asymmetry | 1 228 |
| BCI | erd_inwin | ERD/S inside the window: power of seconds 2–4 vs the first second, log-power trend, ×7 bands | 1 400 |
| BCI | bursts | band-envelope mean / CV / skew, burst rate / duration / fraction | 1 722 |
| BCI | cov | log-Euclidean region covariance ×7 bands | 315 |
| BCI | cov_ch | log-Euclidean 41-channel covariance ×7 bands (Riemannian BCI input) | 6 027 |
| time series | temporal | log var, line length, Hjorth, skew, kurtosis, zero-crossings, ptp | 328 |
| time series | temporal_band | the same (Hjorth, var, skew, kurtosis) per band | 1 435 |
| time series | wavelet | db4 DWT: energy, relative energy, std, entropy per level; wavelet entropy | 1 025 |
| time series | complexity | permutation (3 delays + 4 bands), spectral, SVD, sample entropy; LZC; Higuchi / Katz / Petrosian FD; DFA | 615 |
| time series | complexity_ms | multiscale sample entropy and order-4 permutation entropy (scales 2–5), approximate entropy | 369 |
| time series | catch22 | the 22 canonical hctsa features × (broadband + 7 bands) × 41 channels | 7 216 |
| interaction | conn | PLV, wPLI, AEC, AEC-orth ×7 bands: region pairs, node strength, weighted clustering, global | 3 388 |
| interaction | conn_pairs | the same 4 measures for all 820 channel pairs | 22 960 |
| interaction | wsmi | weighted symbolic MI (King 2013 / Sitt 2014), τ = 1, 2, 4 | 234 |
| interaction | hoi | Gaussian-copula O-information of all region triplets; TC, DTC, O, S ×7 bands, envelopes and signals | 1 232 |
| interaction | cfc | PAC and amplitude–amplitude coupling | 492 |
| baseline | classic_ref | ERD/S against the pre-cue second (*privileged*) | 1 000 |

`classic_ref` uses the pre-cue second, which is outside the window, so a
submission can't compute it. It is the reference to beat and is excluded from
`all`. `erd_inwin` is its deployable counterpart.

## Next steps

- [ ] Smoke test every family on 64 windows (`LIMIT=64 bash run/submit_extract.sh`)
- [ ] Full extraction, then `bash run/submit_eval.sh`
- [ ] Notebook: generality histogram per family; top-20 general features per class
      with their per-user d; LOO all vs general vs pooled per user
- [ ] Redundancy: cluster the top features (|r| > 0.9) before naming "the" proxy
- [ ] Possible additions: specparam, microstates, full hctsa, fold-fitted
      CSP / tangent space, channel-level HOI on a subset
- [ ] Session normalisation (z-score within session): transductive, so report it
      only next to the raw result
