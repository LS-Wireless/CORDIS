# Fig. (appendix) — SNR / power sweep

Optional appendix figure: a sanity check that both metrics scale sensibly with
the transmit power budget, from an `snr_sweep` campaign.

- **(top) min-SINR vs SNR** and **(bottom) sum-SCNR vs SNR.** Both rise
  monotonically with SNR = P_max/σ_n², and the algorithm ordering
  (Centralized ≥ CORDIS-ADMM ≳ CORDIS-Split) is preserved across the range —
  the reviewer-reassurance figure.
- A dashed line marks the deployed operating point **SNR\*** (= `channel.snr_db`
  in the favorable config, 137 dB).

Filtered to the proposed pair vs the centralized ceiling
(`Centralized`, `CORDIS-ADMM`, `CORDIS-Split`), styled via `style_for` /
`ALGORITHM_STYLE`.

## ⚠️ `channel.snr_db` is a transmit-side P_max/σ_n² ≈ 137 dB

In this framework `channel.snr_db` is the *transmit-side* power-to-noise ratio;
the favorable config sets it to **137 dB**. After ~117 dB path loss at the
default 650 m cell, that is the ~20 dB *received* SNR quoted in the system
model. So the sweep grid must **bracket 137 dB** — the registry's generic
default grid `[-10 … 25]` would put every point far below the operating point
and read as all-infeasible. Pass an explicit grid around 137 (see below).

## Files

| File | Purpose |
|---|---|
| `build_fig_snr.py` | Headless builder **and** importable module (loader + helpers + `build_figure`). Writes `../../figures/fig_snr.pdf`. |
| `fig_snr.ipynb` | Interactive front-end. Loader/summary helpers imported from the module; **`build_figure` inlined as an editable cell** for style overrides. |
| `README.md` | This file. |

No runner — point the builder at an `snr_sweep` campaign (or let it
auto-discover the newest, preferring `array_<jobid>_aggregated/`, never
`latest`).

## How to build

```bash
# Newest aggregated run; SNR* auto-detected from metadata (= channel.snr_db):
python3 paper/figure_src/fig_snr/build_fig_snr.py

# On a node without pdflatex:
python3 paper/figure_src/fig_snr/build_fig_snr.py --no-tex

# Pin the operating point / run:
python3 paper/figure_src/fig_snr/build_fig_snr.py \
    --snr-star 137 \
    --result-dir results/exp_snr_sweep/array_12345_aggregated
```

The builder prints each algorithm's min-SINR and SCNR at the lowest, operating,
and highest SNR — enough to confirm monotonic scaling for the caption.

## Recommended sweep grid and campaign

Bracket the 137 dB operating point, denser near it:

```
--snr-values-db "128,132,135,137,140,143"
```

This spans a power-starved regime (~128 dB), the operating point (137 dB,
marked), and a power-rich plateau (~143 dB). Keep the favorable config
otherwise, `spec_set=cordis_vs_centralized` (the trio), and the drop count high.
See the chat notes for the full campaign command.

## Research-integrity note

As an appendix sanity figure this is typically reported at the favorable config
only; if a reviewer asks, the **stress scenario**
(`configs/scenarios/stress.json`) override can be run the same way and pointed
at with `--result-dir`.

