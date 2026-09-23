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

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py
```
