from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from models import Action, Ticket, WorkflowStage


@dataclass(frozen=True)
class GradeResult:
    score: float
    reason: str
    done: bool
    breakdown: Dict[str, float]


def grade_action(difficulty: str, stage: WorkflowStage, ticket: Ticket, action: Action) -> GradeResult:
    if difficulty == "easy":
        return grade_easy_task(stage, ticket, action)
    if difficulty == "medium":
        return grade_medium_task(stage, ticket, action)
    if difficulty == "hard":
        return grade_hard_task(stage, ticket, action)
    return GradeResult(score=0.0, reason="Unsupported task difficulty.", done=False, breakdown={"invalid": 0.0})


def grade_easy_task(stage: WorkflowStage, ticket: Ticket, action: Action) -> GradeResult:
    if stage != WorkflowStage.CLASSIFY:
        return GradeResult(score=-0.2, reason="Wrong action for easy task.", done=True, breakdown={"wrong_action": -0.2})
    classification_score = _score_classification(ticket, action)
    return GradeResult(
        score=classification_score,
        reason="Easy task classification reward.",
        done=True,
        breakdown={"classification": classification_score},
    )


def grade_medium_task(stage: WorkflowStage, ticket: Ticket, action: Action) -> GradeResult:
    if stage == WorkflowStage.CLASSIFY:
        classification_score = _score_classification(ticket, action)
        return GradeResult(
            score=classification_score,
            reason="Medium task classification score.",
            done=False,
            breakdown={"classification": classification_score},
        )
    if stage == WorkflowStage.RESPOND:
        response_score = _score_response_quality(ticket, action)
        return GradeResult(
            score=response_score,
            reason="Medium task response quality score.",
            done=response_score >= 0.2,
            breakdown={"response_quality": response_score},
        )
    return GradeResult(
        score=-0.2,
        reason="Wrong action for medium task.",
        done=False,
        breakdown={"wrong_action": -0.2},
    )


def grade_hard_task(stage: WorkflowStage, ticket: Ticket, action: Action) -> GradeResult:
    if stage == WorkflowStage.CLASSIFY:
        classification_score = _score_classification(ticket, action)
        return GradeResult(
            score=classification_score,
            reason="Hard task classification score.",
            done=False,
            breakdown={"classification": classification_score},
        )
    if stage == WorkflowStage.RESPOND:
        response_score = _score_response_quality(ticket, action)
        return GradeResult(
            score=response_score,
            reason="Hard task intermediate response score.",
            done=False,
            breakdown={"response_quality": response_score},
        )
    if stage == WorkflowStage.RESOLVE:
        resolution_score = _score_resolution(ticket, action)
        return GradeResult(
            score=resolution_score,
            reason="Hard task resolution score.",
            done=True,
            breakdown={"resolution": resolution_score},
        )
    return GradeResult(score=-0.2, reason="Wrong action.", done=False, breakdown={"wrong_action": -0.2})


def _score_classification(ticket: Ticket, action: Action) -> float:
    if action.classification is None:
        return -0.2
    if action.classification == ticket.ticket_type:
        return 0.3
    return -0.2


def _score_response_quality(ticket: Ticket, action: Action) -> float:
    if not action.response_text:
        return -0.2

    lower_text = action.response_text.lower()
    components = [
        any(token in lower_text for token in ("sorry", "apolog", "understand", "thanks")),  # empathy
        any(token in lower_text for token in (ticket.ticket_type.value, "issue", "case", "order")),  # context
        any(token in lower_text for token in ("next", "will", "within", "eta", "refund", "tracking", "resolve")),  # action plan
    ]
    matched = sum(1 for value in components if value)
    if matched == 0:
        return -0.2

    score = round((matched / 3.0) * 0.3, 4)

    # Deterministic partial penalty for escalation mismatch.
    if ticket.requires_escalation and not action.escalate:
        score = max(-0.2, round(score - 0.1, 4))
    elif not ticket.requires_escalation and action.escalate:
        score = max(-0.2, round(score - 0.05, 4))

    return score


def _score_resolution(ticket: Ticket, action: Action) -> float:
    if action.resolve is not True:
        return -0.2

    if ticket.requires_escalation and not action.escalate:
        return 0.2

    return 0.4
