# Ball possession — More than a Hack 2027, Phase 1

Labels every frame of FC Barcelona vs Athletic Club (LaLiga 2024-25, matchday 2) as `H`
(Barcelona), `A` (Athletic) or `D` (dead ball), from the tactical tracking and Smart Tagging only.
No hand labels and no ground truth are used to produce or tune the labels.

## Method in short

1. **Rules** (`1_initial_labels.ipynb`) label the frames the tracking makes clear: a player clearly
   controls the ball → `H`/`A`; six dead-ball rules (ball out, ball lost at a line, ball still,
   set-piece tags, players walking, kickoff formation after a goal) → `D`; between two clear moments
   of the same team, that team keeps the ball. About 82% of graded frames get a label.
2. **LightGBM on player movement only** (`2_lightgbm.ipynb`): 60 team-shape features (width,
   line height, forward speed, marking, ...), no ball. Validated on held-out 2-minute blocks.
3. **Final labels** (`3_final_labels.ipynb`): the model fills the unlabelled frames with `H` or `A`
   only (`D` comes from the rules), then Viterbi smoothing over H/A/D with a cost per change of
   label and a minimum stoppage length of 1 s. Writes and validates the submission CSV.

`explore_tracking.ipynb` and `rules_analysis.ipynb` are the exploratory analysis behind the
thresholds; they are not needed to produce the CSV.

## Reproduce

Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the competition data in `data/` (see `data/README.md`), or set `MTAH_DATA_DIR`:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Only these three files are read. Then:

```bash
jupyter nbconvert --to notebook --execute --inplace 3_final_labels.ipynb
python evaluation/validate_submission.py evaluation/results/team_name.csv
```

The CSV is written to `evaluation/results/<NAME>.csv` (`NAME` is set in section 7 of the
notebook). The run takes a bit over a minute.

The output is deterministic: LightGBM runs with a fixed seed, `deterministic=True`,
`force_row_wise=True` and one thread, so the same data and pinned packages give the same file.
Built and checked on Linux, Python 3.12. Reference checksum of the submitted file:

```
md5  d8e198693b0fc12229614136354452cf
```

(`md5sum evaluation/results/<NAME>.csv` on Linux, `md5 -q` on macOS,
`Get-FileHash -Algorithm MD5` in PowerShell.) Only the file name depends on `NAME`.

## Layout

| Path | What |
|---|---|
| `possession/data.py` | loading the raw tracking and tags, basic signals |
| `possession/rules.py` | clear-control, dead-ball and carry rules |
| `possession/features.py` | player features for the model |
| `possession/model.py` | split, LightGBM training, scores |
| `possession/viterbi.py` | Viterbi smoothing over H/A/D |
| `possession/submission.py` | writing the CSV |
| `games.py`, `starter/`, `evaluation/` | from the organisers' starter repository (MIT, see `LICENSE-metrica`) |

## Credits

`games.py`, `starter/` and `evaluation/` come from the More than a Hack 2027 starter repository by
Metrica Sports, under the MIT licence in `LICENSE-metrica`. The match data is not included and is
licensed for the competition only.
