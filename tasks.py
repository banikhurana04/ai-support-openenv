from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from models import Ticket, TicketType


@dataclass(frozen=True)
class TaskDefinition:
    difficulty: str
    objective: str
    success_condition: str
    required_stages: List[str]


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        difficulty="easy",
        objective="Classify the ticket into refund, complaint, or query.",
        success_condition="Episode succeeds when classification action matches the true issue type.",
        required_stages=["classify"],
    ),
    "medium": TaskDefinition(
        difficulty="medium",
        objective="Classify ticket correctly and provide a helpful response.",
        success_condition="Episode succeeds when classification is correct and response gets a positive response score.",
        required_stages=["classify", "respond"],
    ),
    "hard": TaskDefinition(
        difficulty="hard",
        objective="Handle full workflow across multiple steps: classify, respond, and resolve.",
        success_condition="Episode succeeds when ticket is classified, responded to, and resolved according to policy.",
        required_stages=["classify", "respond", "resolve"],
    ),
}


TASKS: Dict[str, List[Ticket]] = {
    "easy": [
        Ticket(
            id="E-1001",
            issue_type=TicketType.QUERY,
            ticket_id="E-1001",
            difficulty="easy",
            customer_name="Alice",
            customer_tier="standard",
            ticket_type=TicketType.QUERY,
            summary="I cannot find where to update my billing address in account settings.",
            expected_resolution="provide_steps",
            sentiment="neutral",
            urgency=2,
        ),
        Ticket(
            id="E-1002",
            issue_type=TicketType.DELAY,
            ticket_id="E-1002",
            difficulty="easy",
            customer_name="Ben",
            customer_tier="premium",
            ticket_type=TicketType.DELAY,
            summary="My order #4521 is delayed by two days. Please share the latest ETA.",
            expected_resolution="share_tracking_and_eta",
            sentiment="neutral",
            urgency=3,
        ),
    ],
    "medium": [
        Ticket(
            id="M-2001",
            issue_type=TicketType.REFUND,
            ticket_id="M-2001",
            difficulty="medium",
            customer_name="Carla",
            customer_tier="premium",
            ticket_type=TicketType.REFUND,
            summary="I was charged twice for order #8842 and need one charge refunded.",
            expected_resolution="issue_partial_or_full_refund",
            sentiment="negative",
            urgency=4,
        ),
        Ticket(
            id="M-2002",
            issue_type=TicketType.COMPLAINT,
            ticket_id="M-2002",
            difficulty="medium",
            customer_name="Dinesh",
            customer_tier="standard",
            ticket_type=TicketType.COMPLAINT,
            summary="Previous support chat was rude and unhelpful. I want this reviewed.",
            expected_resolution="apology_and_case_reassignment",
            sentiment="negative",
            urgency=3,
        ),
    ],
    "hard": [
        Ticket(
            id="H-3001",
            issue_type=TicketType.COMPLAINT,
            ticket_id="H-3001",
            difficulty="hard",
            customer_name="Enterprise Ops Team",
            customer_tier="enterprise",
            ticket_type=TicketType.COMPLAINT,
            summary="Three enterprise shipments missed SLA this month. Need immediate action and RCA.",
            expected_resolution="escalate_and_sla_credit",
            sentiment="negative",
            urgency=5,
            requires_escalation=True,
        ),
        Ticket(
            id="H-3002",
            issue_type=TicketType.REFUND,
            ticket_id="H-3002",
            difficulty="hard",
            customer_name="Fatima",
            customer_tier="premium",
            ticket_type=TicketType.REFUND,
            summary="Refund for canceled order #9017 was promised 10 days ago but still pending.",
            expected_resolution="escalate_finance_and_priority_refund",
            sentiment="negative",
            urgency=5,
            requires_escalation=True,
        ),
    ],
}


def get_task_pool(difficulty: str) -> List[Ticket]:
    if difficulty not in TASKS:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return TASKS[difficulty]


def get_task_definition(difficulty: str) -> TaskDefinition:
    if difficulty not in TASK_DEFINITIONS:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return TASK_DEFINITIONS[difficulty]


def sample_task(difficulty: str, rng: random.Random | None = None) -> Ticket:
    rng = rng or random.Random()
    pool = get_task_pool(difficulty)
    return rng.choice(pool)
