# CAT Operator Copilot — Architecture

## Overview

Seven features sharing one session context layer, one Attention Manager, and one Task Planning Dashboard.

## Features

1. **Operator Passport** — Identity, authorization, baseline context
2. **Safety Guardian** — Deterministic contextual safety (never LLM-driven)
3. **Behavioral Fingerprint** — Context-adjusted behavior attribution
4. **Predictive Task Intelligence** — ETA + fuel prediction with uncertainty
5. **Optimal Task Sequencing + Task Planning Dashboard** — Weighted optimizer + shift home screen
6. **Operator Training Hub** — Triggered micro-lessons closing Safety→Behavior→Training loop
7. **Grounded AI Operating Buddy** — Evidence-grounded Q&A, safe-state gated

## Cross-cutting: Attention Manager

Every subsystem produces a candidate event. Only the Attention Manager decides what reaches the operator.

Priority order:
1. Critical safety — always wins
2. High safety / urgent
3. Task replanning
4. Behavior nudge
5. Training
6. Buddy

## Data Flow

```
Operator Passport
    ↓
Shared Session Context
    ↓
Safety | Behavior | Prediction
    ↓
Attention Manager | Optimization
    ↓
Task Planning Dashboard
    ↓
Training Hub | Grounded Buddy
    ↓
Updated Operator Context
```

## Key Design Commitments

1. No single-factor recommendations
2. No raw behavioral blame
3. One attention owner
4. Buddy gated to safe states
5. Explicit conflict handling
6. Dashboard + scheduling are one experience
7. Safety→Behavior→Training is the primary closed loop

## Owner Map

| Feature | Owner |
|---|---|
| Behavioral Fingerprint, Predictive Task Intelligence | Sruthi (Team Member 1) |
| Safety Guardian, Attention Manager, Optimization | Aneesha (Team Member 2) |
| Passport, Training, Buddy, Dashboard, UI | Saanvi (Team Member 3) |
| Shared schemas, data | All |
