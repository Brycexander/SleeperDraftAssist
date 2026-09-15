# Free expert source audit

Audited 2026-09-13. The public FantasyPros ROS page exposes individual expert columns and update dates, but this audit did not establish a complete, freely automatable feed of the top ten prior-season ROS accuracy finishers across QB/RB/WR/TE. The app therefore uses a validated manual JSON import path and does not silently substitute Sleeper projections.

| Source URL | Retrieved UTC | Published UTC | Season/week | Scoring/scope | Expert IDs | Accuracy season/place/evidence | Accessible rows | Decision |
|---|---|---|---|---|---|---|---|---|
| https://www.fantasypros.com/nfl/fantasy-football-rankings/ros-rb.php | 2026-09-13 | page shows Sep 07, 2026 | 2026 ROS / Standard example | Standard / RB | public page columns | individual accuracy qualification not established | public HTML shows 3 expert columns for RB | reference only; no automatic adapter |
| https://www.fantasypros.com/nfl/accuracy/?year=2025 | 2026-09-13 | standings page | 2025 accuracy | unspecified without selected table | standings page | standings exist, but complete free ROS top-10 export was not verified | not normalized to Sleeper IDs | reference only; manual evidence required |

The manual import contract is documented in `docs/superpowers/specs/2026-09-13-expert-roster-advice-design.md`. A candidate must include at least two of the last three completed seasons of comparable overall ROS accuracy, field sizes, graded-week counts, current-week ROS ranks for all four offensive positions, matching scoring, a publication date within seven days, and stable player IDs. The app selects five to eight candidates with no more than two per publisher and shows imported data as user-provided evidence.
