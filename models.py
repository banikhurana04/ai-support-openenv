from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TicketType(str, Enum):
    REFUND = "refund"
    DELAY = "delay"
    COMPLAINT = "complaint"
    QUERY = "query"


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class WorkflowStage(str, Enum):
    CLASSIFY = "classify"
    RESPOND = "respond"
    RESOLVE = "resolve"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    RESPOND = "respond"
    RESOLVE = "resolve"


class ConversationTurn(BaseModel):
    speaker: Literal["customer", "agent", "system"]
    message: str = Field(min_length=1)


class Ticket(BaseModel):
    id: str
    issue_type: TicketType
    ticket_id: str
    difficulty: Literal["easy", "medium", "hard"]
    customer_name: str
    customer_tier: Literal["standard", "premium", "enterprise"]
    ticket_type: TicketType
    summary: str
    expected_resolution: str
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: int = Field(ge=1, le=5)
    requires_escalation: bool = False


class Observation(BaseModel):
    id: str
    issue_type: TicketType
    ticket_id: str
    stage: WorkflowStage
    status: TicketStatus
    summary: str
    sentiment: str
    urgency: int = Field(ge=1, le=5)
    conversation_history: List[ConversationTurn] = Field(default_factory=list)
    allowed_actions: List[ActionType] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    action_type: ActionType
    classification: Optional[TicketType] = None
    response_text: Optional[str] = None
    resolve: Optional[bool] = None
    escalate: bool = False
    notes: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("response_text")
    @classmethod
    def non_empty_response_text(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("response_text must not be blank")
        return value


class Reward(BaseModel):
    value: float
    reason: str
    breakdown: Dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
