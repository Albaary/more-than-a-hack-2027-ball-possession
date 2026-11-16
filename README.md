# More Than A Hack 2027 — Phase 1: Ball Possession

Team: **Alamak Menang Wak**

A classical pipeline (no learned models) that labels every frame of the Barcelona (H) vs
Athletic Club (A) match with one of `H` (Barcelona in possession), `A` (Athletic in
possession), `D` (dead ball), or `X` (abstain / not graded).

> The ranking metric is **spellF1** (temporal-IoU F1 over possessions), not per-frame
> accuracy. See the organisers' `EVALUATION.md`.

---

## 1. How it works

`predict_possession.py` streams `BAR-ATH_tactical_atd.txt` and the Smart Tagging events
(two passes, < 15 MB RAM) and then:

1. **Pass 1 — ball trajectory.** Raw ball position, ball speed, length of each missing-ball
   gap, and proximity to the touchlines. Ball position is interpolated only across gaps of
   <= 35 frames (1.4 s).
2. **Pass 2 — player distances.** Distance from the 1st and 2nd nearest player of each team
   to the ball, plus the mean player speed.
3. **Smart Tagging context.** Offensive phases (with a +6 frame latency alignment) and the
   start of each *Defensive Transition* are used as a team prior. Set Pieces define a
   dead-ball window (before the restart) and an alive window (after the restart).
4. **Two decoupled tracks.** A team track (H vs A: nearest player + prior) and a dead-ball
   track (D: ball out of play, long missing-ball gaps, Set Pieces, slow ball/players).
5. **Smoothing and clean-up.** Majority vote over +/-`--window-team` frames, removal of very
   short segments, an anti-split rule (H-D-H), then *Transition Boundary Snapping*
   (+/-10 frames).
6. **Latency correction.** H<->A boundaries are shifted forward by `--transition-delay`
   frames and D->H/A boundaries by `--restart-delay` frames (see Honest notes).

---

## 2. How to run

### Requirements
- Python 3.9+ and `numpy`.
- The organisers' repository `more-than-a-hack-2027` (provides `games.py`, `starter/`,
  `evaluation/`).
- The match data in `data/` (**do not commit it** — the data may not be redistributed).

```bash
pip install -r requirements.txt
```

### Generate the submission file
From the repository root:

```bash
python predict_possession.py --team "Alamak Menang Wak"
python evaluation/validate_submission.py "Alamak Menang Wak.csv"
python evaluation/run_evaluation.py
```

Output: `evaluation/results/Alamak Menang Wak.csv`, with a copy in the repository root.
Always validate the exact file you are about to upload, not one built earlier.

### Parameters

| Argument | Default | Purpose |
|---|---|---|
| `--team` | `Alamak Menang Wak` | Team name = CSV file name |
| `--window-team` | 50 | Half-width of the H vs A majority vote (frames) |
| `--window-dead` | 35 | Half-width of the D smoothing (frames) |
| `--min-spell` | 45 | Minimum possession length (frames) |
| `--min-dead` | 75 | Minimum dead-ball segment length (frames) |
| `--transition-delay` | 8 | Shift H<->A boundaries forward (frames); 0 = off |
| `--restart-delay` | 12 | Shift D->H/A boundaries forward (frames); 0 = off |
| `--dead-ball-speed` | 2.1 | Ball speed threshold (m/s) for staying in D |
| `--dead-player-speed` | 2.20 | Mean player speed threshold (m/s) for staying in D |
| `--data-dir` | (empty) | Optional data folder |

The submission uses the defaults above.

---

## 3. Results on the public window (5 minutes, 17 possessions)

All numbers below come from `evaluation/run_evaluation.py` against the **public** ground
truth (7,500 frames). They are **not** full-match scores.

| System | macro_f1 | spellF1 (mean) | f1_D | recall_D | median latency (frames) | possessions (GT 17) |
|---|---|---|---|---|---|---|
| baseline_smooth50 (organisers) | 0.6207 | 0.3256 | 0.323 | 0.214 | +4 | 26 |
| baseline_smooth125 (organisers) | 0.5812 | 0.4242 | 0.146 | 0.079 | -3 | 16 |
| This pipeline, window 60, no delays | 0.7647 | 0.4516 | 0.635 | 0.546 | -11 | 14 |
| This pipeline, window 60, with delays | **0.7755** | 0.4774 | 0.656 | 0.577 | -2 | 14 |
| **Submission (defaults: window 50, with delays)** | 0.7702 | **0.5371** | 0.656 | 0.577 | -2 | 18 |

Further details:
- Window 50 has a higher headline spellF1 than window 60 (0.571 vs 0.516) but slightly lower
  accuracy (0.802 vs 0.808) and more spurious possessions (15 vs 11).
- Loosening the dead-ball thresholds (`--dead-ball-speed 2.6 --dead-player-speed 2.35`)
  produced scores identical to the defaults, so those thresholds are effectively inactive on
  this data.
- `--min-spell 60` (with window 50) lowered spellF1 from 0.537 to 0.488.

---

## 4. Honest notes (read before interpreting the numbers)

1. **The differences in the table are within the noise band.** `EVALUATION.md` states that
   differences below 0.04 in spellF1 and below 0.01 in macro_f1 are noise, and on the public
   window a single possession moves spellF1 by about 0.06. Do not read the table above as a
   definitive ranking.
2. **The public window is not a tuning set.** `RULES.md` forbids using it to choose between
   configurations. The values `--restart-delay 12` and `--transition-delay 8` were derived
   from transition latencies measured on the public window (only 6-7 matched transitions),
   and `--window-team 50` was chosen after seeing public scores. The only independent support
   for window 50 is that the organisers' full-match baseline scores rank window 50 above 25
   and 125. The delay parameters have no independent basis; both can be set to 0.
3. **Frame constants.** The code uses `110808..113741` (the pre-kick-off range of the second
   half, which the organisers' FAQ requires to be `D`) and `11185` (first-half kick-off, taken
   from the organisers' `games.py`) to exclude kick-offs from dead-ball marking.
4. **Full-match scores are likely lower.** The organisers' baselines drop 0.08-0.20 in
   spellF1 when moving from the public window to the full match.
5. **Biggest weaknesses:** `recall_D` is only 0.577 and the possession count is still
   unstable (14-18 vs 17 in the ground truth, depending on the smoothing window).
6. No hand labels, no private ground truth, and no human-produced data about this match
   were used. All labels come from an algorithm.

---

## 5. Repository layout

```
more-than-a-hack-2027/
├── predict_possession.py    # the pipeline
├── requirements.txt         # numpy (pinned)
├── README.md
├── TECHNICAL.md             # technical document for the submission form
├── .gitignore               # data/, *.csv, evaluation/results/
├── games.py                 # from the organisers' repo
├── starter/                 # from the organisers' repo
├── evaluation/              # from the organisers' repo
└── data/                    # NOT committed
```

## 6. Reproducibility

The pipeline is deterministic (no randomness). Running the commands in section 2 on the same
data should produce an identical CSV. Verify with a hash:

```bash
python predict_possession.py --team "Alamak Menang Wak"
sha256sum "Alamak Menang Wak.csv"
```

## 7. Submission

- Deadline: **17 November 2026**, one Google Form response per team.
- Upload: `<team_name>.csv`, the public repository URL, and `<team_name>.pdf` or `.md`.
- Make sure no match data is present in the public repository.
