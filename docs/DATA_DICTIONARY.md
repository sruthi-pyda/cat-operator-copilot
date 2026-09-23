# Data Dictionary

All synthetic datasets include `data_source`, `synthetic_flag`, `data_quality_confidence`.

## operators.csv

| Column | Type | Description |
|---|---|---|
| operator_id | str | OP1001–OP1030 |
| name | str | Display name |
| role | str | Job role |
| years_experience | float | Years of experience |
| machine_skill_level | str | novice/intermediate/advanced/expert |
| task_skill_level | str | novice/intermediate/advanced/expert |
| authorized_machine_types | str | Comma-separated machine types |
| certification_status | str | active/expired/none |
| certification_expiry | date | Expiry date |
| baseline_idle_ratio | float | Personal baseline idle ratio |
| baseline_cycle_time_sec | float | Personal baseline cycle time |
| baseline_fuel_l_per_cycle | float | Personal baseline fuel per cycle |
| baseline_safety_event_rate | float | Historical safety event rate |
| recent_workload_hours | float | Hours worked recently |
| fatigue_proxy | float | 0–1 fatigue estimate |
| training_status | str | up_to_date/due/overdue |
| face_registered | bool | True for OP1001–OP1003 only |
| face_embedding_path | str | Path to stored embedding |
| face_model | str | Model used for registration |
| face_registration_timestamp | datetime | When registered |
| data_source | str | synthetic |
| synthetic_flag | bool | true |

## machines.csv

| Column | Type | Description |
|---|---|---|
| machine_id | str | EXC001–EXC010 |
| machine_model | str | CAT model name |
| machine_type | str | excavator/dozer/grader/etc |
| machine_year | int | Year of manufacture |
| engine_hours | float | Total engine hours |
| machine_condition | str | good/fair/poor |
| fuel_capacity_l | float | Total fuel tank capacity |
| current_fuel_level_pct | float | 0–100 |
| attachment_type | str | bucket/blade/grapple/etc |
| rated_capacity | float | tonnes |
| maintenance_status | str | ok/due/overdue |
| last_maintenance_timestamp | datetime | Last service date |
| maintenance_due_hours | float | Engine hours at next service |
| engine_health_score | float | 0–1 |
| hydraulic_health_score | float | 0–1 |
| data_source | str | synthetic |
| synthetic_flag | bool | true |

## tasks.csv

| Column | Type | Description |
|---|---|---|
| task_id | str | T00001–T01500 |
| task_type | str | excavation/loading/trenching/grading/hauling/stockpiling |
| task_phase | str | setup/active/wrap_up |
| workload | str | light/medium/heavy |
| target_output | float | Target units |
| priority | int | 1 (highest)–5 |
| deadline_timestamp | datetime | Hard deadline |
| location_id | str | LOC001–LOC010 |
| dependency_task_id | str | Task ID or null |
| estimated_travel_distance_km | float | Travel distance |
| required_machine_type | str | Required machine type |
| required_attachment | str | Required attachment |
| required_skill_level | str | Minimum skill level |
| status | str | pending/active/completed/blocked |

## task_sessions.csv

| Column | Type | Description |
|---|---|---|
| session_id | str | S000001–S005000 |
| task_id | str | FK → tasks.csv |
| operator_id | str | FK → operators.csv |
| machine_id | str | FK → machines.csv |
| start_timestamp | datetime | Session start |
| end_timestamp | datetime | Session end |
| actual_task_duration_min | float | **TARGET — never use as input** |
| actual_fuel_used_l | float | **TARGET — never use as input** |
| actual_idle_time_min | float | Observed idle time |
| actual_cycle_count | int | Observed cycle count |
| data_source | str | synthetic |
| synthetic_flag | bool | true |
| data_quality_confidence | float | 0–1 |

## telemetry.csv

Time-series rows at ~30–60 second intervals. See Section 6 of Claude Instructions for full column list.

**Note:** `actual_task_duration_min` and `actual_fuel_used_l` from task_sessions are prediction targets and must never be used as model features.

## safety_events.csv

Includes deliberately contrasting examples:
- Worker nearby + machine stationary → not automatically dangerous
- Worker nearby + machine swinging + in envelope + closing speed > 0 → potentially critical

## training_records.csv

Training triggered only by repeated, high-confidence, operator-attributable issues.
