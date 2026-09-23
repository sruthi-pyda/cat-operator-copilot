# CAT Operator Copilot — 3-Person Hackathon

Team: Sruthi, Aneesha, Saanvi

## Phases

1. Phase 0 — Project Setup
2. Phase 1 — Synthetic Data
3. Phase 2 — Individual Features (parallel)
4. Phase 3 — Integration

See development plan PDF for details.

## Branch Strategy

```
main
develop
feature/sruthi-data-models
feature/aneesha-safety-optimization
feature/saanvi-passport-ui
```

## File Ownership

| Owner | Files |
|---|---|
| Sruthi | `features/behavior/`, `features/prediction/`, `models/`, `scripts/`, `data/processed/`, `reports/` |
| Aneesha | `features/safety/`, `features/attention/`, `features/optimization/`, `config/safety_rules.yaml` |
| Saanvi | `features/passport/`, `features/training/`, `features/buddy/`, `features/dashboard/`, `app/ui/` |
| All | `shared/`, `docs/`, `requirements.txt`, `README.md` |

## Quick Start

Use a project-local virtual environment. DeepFace pulls TensorFlow, and a global
install can break other Python projects on the same machine (see `docs/DECISIONS.md` D007).

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
.venv/bin/python -m pip install -r requirements.txt           # macOS / Linux
```

Generate the dataset, or take it from the data branch without merging:

```bash
python scripts/generate_synthetic_data.py
```

```bash
git checkout origin/feature/sruthi-data-models -- data/synthetic/
git reset -q data/synthetic/
```

Run the tests:

```bash
.venv/Scripts/python.exe -m pytest -q
```

## Running the system

The shift dashboard — operator, plan, prediction, conditions, live operation,
attention queue, Buddy, Training Hub and end-of-shift comparison:

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m streamlit run app/ui/dashboard.py --server.port 8502
```

Replay a recorded session as a live shift:

```bash
.venv/Scripts/python.exe replay.py --session S004454
```

Face registration and login for the three real demo operators:

```bash
.venv/Scripts/python.exe -m features.passport.registration register --operator-id OP1003
```

```bash
.venv/Scripts/python.exe -m features.passport.registration login
```

`PYTHONIOENCODING=utf-8` is required on Windows wherever DeepFace is imported: its
logger prints emoji and the default console encoding otherwise kills the process (D014).

## Notes

- All data is synthetic and carries `synthetic_flag`. None of it is real CAT telemetry.
- Safety decisions are deterministic and configured in `config/safety_rules.yaml`.
  The Buddy may explain a safety event; it never makes or overrides one.
- Prediction targets (`actual_task_duration_min`, `actual_fuel_used_l`) are never model inputs.
