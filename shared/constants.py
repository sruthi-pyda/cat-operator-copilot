# Team member operator IDs.
#
# Reassigned 2026-09-24 (see docs/DECISIONS.md D040). OP1001 is the operator the
# generator shapes to have a coachable idle pattern, so it is the only identity
# whose Safety -> Behavior -> Training loop fires end to end. Saanvi holds it
# because the face registered against OP1001 is the one that drives the demo.
#
# This mapping is descriptive only -- nothing imports it. The binding that
# matters is which face is enrolled under which id in data/face_registrations/.
OPERATORS = {
    "SAANVI": "OP1001",
    "SRUTHI": "OP1002",
    "ANEESHA": "OP1003",
}

# Machine IDs
MACHINES = [f"EXC{i:03d}" for i in range(1, 11)]

# Data source constants
DATA_SOURCE_SYNTHETIC = "synthetic"
DATA_QUALITY_CONFIDENCE_DEFAULT = 0.85

# Task types
TASK_TYPES = [
    "excavation",
    "loading",
    "trenching",
    "grading",
    "hauling",
    "stockpiling",
]

# Severity levels
SEVERITY_LEVELS = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Attribution categories
ATTRIBUTION_TYPES = [
    "operator-driven",
    "context-driven",
    "mixed",
    "insufficient_evidence",
]

# Attention Manager decisions
ATTENTION_DECISIONS = ["show_now", "queue", "bundle", "suppress"]
