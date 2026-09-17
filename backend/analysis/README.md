# Analysis working folder

Scratch folder for the pre-defense empirical rigor pass requested on 2026-09-14.
Nothing here has been folded into `thesis-manuscript/` yet — see `TASK1_NOTES.md`
and (later) the master changelog before anything moves into the thesis text.

Inputs used (copied from `~/Downloads/`, not tracked in git — see `.gitignore`):
- `cameo_real_per_drug_results.csv` — the 53-held-out-drug leave-drugs-out MAE table
  behind Table 5.3 of the thesis. Verified against the thesis's reported aggregates
  (mean MAE 89.1 / 100.4 / 103.7, win rate 52.8% / 22.6% / 17.0% / 7.5%) — exact match,
  confirmed against the two duplicate copies of the same file as well.

Scripts are inline Python run via the shell and saved as `.csv` outputs; no notebook
was created since each step is short and reproducible from the commands in the chat log.
