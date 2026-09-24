"""Loads and looks up the Training Hub's content pack.

Content lives in `features/training/content/` as YAML data files, never as
Python, so lessons can be edited without touching code. This module is the only
place that knows their shape.

    issue_type -> lesson -> quiz -> scenarios

Rules enforced at load time, so a malformed pack fails loudly at import of the
content rather than silently mis-coaching an operator later:

  - every lesson declares at least one issue_type, and no issue_type is claimed
    by two lessons (the mapping must be unambiguous);
  - every lesson's quiz_id resolves to a quiz;
  - every quiz has 5-10 questions, unique option ids, and an answer_id that is
    one of its own options;
  - every scenario's lesson_id resolves to a lesson.

The pack is synthetic demonstration material. `synthetic_flag` rides on every
record and `disclaimer` is carried through for display, because anything shown
to an operator must say what it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from shared.config import load_settings, load_yaml, resolve_path

LESSONS_FILE = "lessons.yaml"
QUIZZES_FILE = "quizzes.yaml"
SCENARIOS_FILE = "scenarios.yaml"

MIN_QUIZ_QUESTIONS = 5
MAX_QUIZ_QUESTIONS = 10


class ContentError(ValueError):
    """The content pack on disk is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class QuizOption:
    option_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"option_id": self.option_id, "text": self.text}


@dataclass(frozen=True)
class QuizQuestion:
    question_id: str
    prompt: str
    options: tuple[QuizOption, ...]
    answer_id: str
    explanation: str = ""

    def is_correct(self, option_id: Optional[str]) -> bool:
        return option_id == self.answer_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "prompt": self.prompt,
            "options": [o.to_dict() for o in self.options],
            "answer_id": self.answer_id,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class Quiz:
    quiz_id: str
    lesson_id: str
    title: str
    questions: tuple[QuizQuestion, ...]
    synthetic_flag: bool = True

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    def question(self, question_id: str) -> Optional[QuizQuestion]:
        for question in self.questions:
            if question.question_id == question_id:
                return question
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "quiz_id": self.quiz_id,
            "lesson_id": self.lesson_id,
            "title": self.title,
            "questions": [q.to_dict() for q in self.questions],
            "total_questions": self.total_questions,
            "synthetic_flag": self.synthetic_flag,
        }


@dataclass(frozen=True)
class Lesson:
    lesson_id: str
    title: str
    issue_types: tuple[str, ...]
    quiz_id: str
    duration_min: int = 0
    summary: str = ""
    objectives: tuple[str, ...] = ()
    key_points: tuple[str, ...] = ()
    practice_prompt: str = ""
    # Optional lesson video: a path under features/training/content/media/ or a
    # URL. Empty means no video is attached, which the UI states plainly rather
    # than rendering a broken player.
    video_url: str = ""
    disclaimer: str = ""
    synthetic_flag: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "title": self.title,
            "issue_types": list(self.issue_types),
            "quiz_id": self.quiz_id,
            "duration_min": self.duration_min,
            "summary": self.summary,
            "objectives": list(self.objectives),
            "key_points": list(self.key_points),
            "practice_prompt": self.practice_prompt,
            "disclaimer": self.disclaimer,
            "synthetic_flag": self.synthetic_flag,
        }


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    lesson_id: str
    title: str
    situation: str = ""
    context_factors: tuple[str, ...] = ()
    what_the_model_saw: str = ""
    discussion_questions: tuple[str, ...] = ()
    takeaway: str = ""
    disclaimer: str = ""
    synthetic_flag: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "lesson_id": self.lesson_id,
            "title": self.title,
            "situation": self.situation,
            "context_factors": list(self.context_factors),
            "what_the_model_saw": self.what_the_model_saw,
            "discussion_questions": list(self.discussion_questions),
            "takeaway": self.takeaway,
            "disclaimer": self.disclaimer,
            "synthetic_flag": self.synthetic_flag,
        }


@dataclass(frozen=True)
class TrainingContent:
    """The whole validated pack, plus the issue_type -> lesson_id index."""

    lessons: tuple[Lesson, ...]
    quizzes: tuple[Quiz, ...]
    scenarios: tuple[Scenario, ...]
    _issue_index: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = {issue: lesson.lesson_id for lesson in self.lessons for issue in lesson.issue_types}
        object.__setattr__(self, "_issue_index", index)

    # --- lookup ---------------------------------------------------------------

    def issue_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._issue_index))

    def lesson_id_for_issue(self, issue_type: str) -> Optional[str]:
        """The lesson that covers this issue type, or None if nothing covers it."""
        return self._issue_index.get(issue_type)

    def lesson_for_issue(self, issue_type: str) -> Optional[Lesson]:
        lesson_id = self.lesson_id_for_issue(issue_type)
        return self.lesson(lesson_id) if lesson_id else None

    def lesson(self, lesson_id: str) -> Optional[Lesson]:
        for lesson in self.lessons:
            if lesson.lesson_id == lesson_id:
                return lesson
        return None

    def quiz(self, quiz_id: str) -> Optional[Quiz]:
        for quiz in self.quizzes:
            if quiz.quiz_id == quiz_id:
                return quiz
        return None

    def quiz_for_lesson(self, lesson_id: str) -> Optional[Quiz]:
        lesson = self.lesson(lesson_id)
        return self.quiz(lesson.quiz_id) if lesson else None

    def scenarios_for_lesson(self, lesson_id: str) -> tuple[Scenario, ...]:
        return tuple(s for s in self.scenarios if s.lesson_id == lesson_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lessons": [lesson.to_dict() for lesson in self.lessons],
            "quizzes": [quiz.to_dict() for quiz in self.quizzes],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "issue_types": list(self.issue_types()),
        }


# --- parsing ------------------------------------------------------------------

def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping or mapping[key] in (None, ""):
        raise ContentError(f"{where} is missing required field {key!r}")
    return mapping[key]


def _parse_lesson(raw: dict[str, Any], disclaimer: str, synthetic: bool) -> Lesson:
    lesson_id = str(_require(raw, "lesson_id", "lesson"))
    issue_types = _tuple_of_str(raw.get("issue_types"))
    if not issue_types:
        raise ContentError(f"lesson {lesson_id} declares no issue_types")
    return Lesson(
        lesson_id=lesson_id,
        title=str(_require(raw, "title", f"lesson {lesson_id}")),
        issue_types=issue_types,
        quiz_id=str(_require(raw, "quiz_id", f"lesson {lesson_id}")),
        duration_min=int(raw.get("duration_min", 0) or 0),
        summary=str(raw.get("summary", "") or ""),
        objectives=_tuple_of_str(raw.get("objectives")),
        key_points=_tuple_of_str(raw.get("key_points")),
        practice_prompt=str(raw.get("practice_prompt", "") or ""),
        video_url=str(raw.get("video_url", "") or ""),
        disclaimer=disclaimer,
        synthetic_flag=bool(raw.get("synthetic_flag", synthetic)),
    )


def _parse_question(raw: dict[str, Any], quiz_id: str) -> QuizQuestion:
    question_id = str(_require(raw, "question_id", f"quiz {quiz_id} question"))
    where = f"quiz {quiz_id} question {question_id}"
    options = tuple(
        QuizOption(option_id=str(_require(o, "option_id", where)), text=str(o.get("text", "")))
        for o in (_require(raw, "options", where) or [])
    )
    if len(options) < 2:
        raise ContentError(f"{where} needs at least two options")
    option_ids = [o.option_id for o in options]
    if len(set(option_ids)) != len(option_ids):
        raise ContentError(f"{where} has duplicate option ids")
    answer_id = str(_require(raw, "answer_id", where))
    if answer_id not in option_ids:
        raise ContentError(f"{where} answer_id {answer_id!r} is not one of its options")
    return QuizQuestion(
        question_id=question_id,
        prompt=str(_require(raw, "prompt", where)),
        options=options,
        answer_id=answer_id,
        explanation=str(raw.get("explanation", "") or ""),
    )


def _parse_quiz(raw: dict[str, Any], synthetic: bool) -> Quiz:
    quiz_id = str(_require(raw, "quiz_id", "quiz"))
    questions = tuple(_parse_question(q, quiz_id) for q in (_require(raw, "questions", f"quiz {quiz_id}") or []))
    if not MIN_QUIZ_QUESTIONS <= len(questions) <= MAX_QUIZ_QUESTIONS:
        raise ContentError(
            f"quiz {quiz_id} has {len(questions)} questions; "
            f"expected {MIN_QUIZ_QUESTIONS}-{MAX_QUIZ_QUESTIONS}"
        )
    question_ids = [q.question_id for q in questions]
    if len(set(question_ids)) != len(question_ids):
        raise ContentError(f"quiz {quiz_id} has duplicate question ids")
    return Quiz(
        quiz_id=quiz_id,
        lesson_id=str(raw.get("lesson_id", "") or ""),
        title=str(raw.get("title", "") or ""),
        questions=questions,
        synthetic_flag=bool(raw.get("synthetic_flag", synthetic)),
    )


def _parse_scenario(raw: dict[str, Any], disclaimer: str, synthetic: bool) -> Scenario:
    scenario_id = str(_require(raw, "scenario_id", "scenario"))
    return Scenario(
        scenario_id=scenario_id,
        lesson_id=str(_require(raw, "lesson_id", f"scenario {scenario_id}")),
        title=str(_require(raw, "title", f"scenario {scenario_id}")),
        situation=str(raw.get("situation", "") or ""),
        context_factors=_tuple_of_str(raw.get("context_factors")),
        what_the_model_saw=str(raw.get("what_the_model_saw", "") or ""),
        discussion_questions=_tuple_of_str(raw.get("discussion_questions")),
        takeaway=str(raw.get("takeaway", "") or ""),
        disclaimer=disclaimer,
        synthetic_flag=bool(raw.get("synthetic_flag", synthetic)),
    )


def _read_pack(path: Path, key: str) -> tuple[list[dict[str, Any]], str, bool]:
    if not path.exists():
        raise ContentError(f"Training content file not found: {path}")
    document = load_yaml(path)
    entries = document.get(key)
    if not entries:
        raise ContentError(f"{path} contains no {key!r} entries")
    return list(entries), str(document.get("disclaimer", "") or ""), bool(document.get("synthetic_flag", True))


def _validate(content: TrainingContent) -> None:
    lesson_ids = [lesson.lesson_id for lesson in content.lessons]
    if len(set(lesson_ids)) != len(lesson_ids):
        raise ContentError("duplicate lesson_id in lessons.yaml")

    quiz_ids = [quiz.quiz_id for quiz in content.quizzes]
    if len(set(quiz_ids)) != len(quiz_ids):
        raise ContentError("duplicate quiz_id in quizzes.yaml")

    claimed: dict[str, str] = {}
    for lesson in content.lessons:
        if content.quiz(lesson.quiz_id) is None:
            raise ContentError(f"lesson {lesson.lesson_id} references unknown quiz {lesson.quiz_id!r}")
        for issue_type in lesson.issue_types:
            if issue_type in claimed:
                raise ContentError(
                    f"issue_type {issue_type!r} is claimed by both "
                    f"{claimed[issue_type]} and {lesson.lesson_id}"
                )
            claimed[issue_type] = lesson.lesson_id

    for quiz in content.quizzes:
        if quiz.lesson_id and content.lesson(quiz.lesson_id) is None:
            raise ContentError(f"quiz {quiz.quiz_id} references unknown lesson {quiz.lesson_id!r}")

    for scenario in content.scenarios:
        if content.lesson(scenario.lesson_id) is None:
            raise ContentError(
                f"scenario {scenario.scenario_id} references unknown lesson {scenario.lesson_id!r}"
            )


def default_content_dir(settings: Optional[dict[str, Any]] = None) -> Path:
    settings = settings if settings is not None else load_settings()
    configured = (settings.get("training") or {}).get("content_dir", "features/training/content")
    return resolve_path(configured)


def load_content(content_dir: Optional[Path] = None) -> TrainingContent:
    """Read, parse and validate the content pack. Results are cached per directory."""
    directory = Path(content_dir) if content_dir is not None else default_content_dir()
    return _load_content_cached(str(directory.resolve()))


@lru_cache(maxsize=8)
def _load_content_cached(directory: str) -> TrainingContent:
    path = Path(directory)
    raw_lessons, lesson_disclaimer, lesson_synthetic = _read_pack(path / LESSONS_FILE, "lessons")
    raw_quizzes, _, quiz_synthetic = _read_pack(path / QUIZZES_FILE, "quizzes")
    raw_scenarios, scenario_disclaimer, scenario_synthetic = _read_pack(path / SCENARIOS_FILE, "scenarios")

    content = TrainingContent(
        lessons=tuple(_parse_lesson(r, lesson_disclaimer, lesson_synthetic) for r in raw_lessons),
        quizzes=tuple(_parse_quiz(r, quiz_synthetic) for r in raw_quizzes),
        scenarios=tuple(_parse_scenario(r, scenario_disclaimer, scenario_synthetic) for r in raw_scenarios),
    )
    _validate(content)
    return content


def clear_content_cache() -> None:
    _load_content_cached.cache_clear()


def lesson_id_for_issue(issue_type: str, content: Optional[TrainingContent] = None) -> Optional[str]:
    """Map an issue_type to its lesson_id, or None when no lesson covers it."""
    return (content or load_content()).lesson_id_for_issue(issue_type)
