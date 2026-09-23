# Team member operator IDs
OPERATORS = {
    "SRUTHI": "OP1001",
    "ANEESHA": "OP1002",
    "SAANVI": "OP1003"
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
