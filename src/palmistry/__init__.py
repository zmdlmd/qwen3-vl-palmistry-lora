from .prompts import (
    DEFAULT_STUDENT_STRUCTURED_PROMPT,
    STYLE_OPTIONS,
    build_followup_prompt,
    build_report_prompt,
    build_teacher_structured_prompt,
    normalize_style,
)

__all__ = [
    "DEFAULT_STUDENT_STRUCTURED_PROMPT",
    "PalmistryPipeline",
    "STYLE_OPTIONS",
    "build_followup_prompt",
    "build_report_prompt",
    "build_teacher_structured_prompt",
    "normalize_style",
]


def __getattr__(name: str):
    if name == "PalmistryPipeline":
        from .pipeline import PalmistryPipeline

        return PalmistryPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
