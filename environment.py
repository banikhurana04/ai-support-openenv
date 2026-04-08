from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from grader import GradeResult, grade_action
from models import (
    Action,
    ActionType,
    ConversationTurn,
    Observation,
    Reward,
    StepResult,
    Ticket,
    TicketStatus,
    WorkflowStage,
)
from tasks import TaskDefinition, get_task_definition, sample_task


class SupportTicketEnv:
    """
    OpenEnv-style customer support environment.

    Workflow:
        classify -> respond -> resolve
    """

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None, max_steps: int = 6) -> None:
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self._ticket: Optional[Ticket] = None
        self._task_definition: Optional[TaskDefinition] = None
        self._status: TicketStatus = TicketStatus.OPEN
        self._stage: WorkflowStage = WorkflowStage.CLASSIFY
        self._history: List[ConversationTurn] = []
        self._done: bool = False
        self._steps: int = 0
        self._cumulative_reward: float = 0.0

    def reset(self, difficulty: Optional[str] = None) -> Observation:
        """
        Reset environment with a new sampled ticket.
        """
        if difficulty is not None:
            self.difficulty = difficulty

        self._ticket = sample_task(self.difficulty, self.rng)
        self._task_definition = get_task_definition(self.difficulty)
        self._status = TicketStatus.OPEN
        self._stage = WorkflowStage.CLASSIFY
        self._history = [
            ConversationTurn(
                speaker="customer",
                message=self._ticket.summary,
            )
        ]
        self._done = False
        self._steps = 0
        self._cumulative_reward = 0.0
        return self._build_observation()

    def state(self) -> Dict[str, Any]:
        """
        Return internal state for inspection/logging.
        """
        if self._ticket is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._task_definition is None:
            raise RuntimeError("Task definition missing. Call reset() first.")

        return {
            "id": self._ticket.id,
            "issue_type": self._ticket.issue_type.value,
            "ticket_id": self._ticket.ticket_id,
            "difficulty": self._ticket.difficulty,
            "ticket_type": self._ticket.ticket_type.value,
            "status": self._status.value,
            "stage": self._stage.value,
            "done": self._done,
            "objective": self._task_definition.objective,
            "success_condition": self._task_definition.success_condition,
            "steps": self._steps,
            "cumulative_reward": self._cumulative_reward,
            "conversation_history": [turn.model_dump() for turn in self._history],
        }

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply action and return:
            (observation, reward, done, info)
        """
        if self._ticket is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._done:
            raise RuntimeError("Episode already finished. Call reset() for a new episode.")

        is_valid_action = self._validate_action_for_stage(action)
        if is_valid_action:
            grade = grade_action(self.difficulty, self._stage, self._ticket, action)
        else:
            grade = GradeResult(
                score=-0.2,
                reason="Wrong action for current stage.",
                done=False,
                breakdown={"wrong_action": -0.2},
            )
        should_end_by_task = self._should_end_by_task_requirements(grade.score)
        self._cumulative_reward += grade.score
        self._steps += 1

        unnecessary_step_penalty_applied = False
        if self._steps > len(self._task_definition.required_stages) and not (grade.done or should_end_by_task):
            unnecessary_step_penalty_applied = True
            grade.score -= 0.1
            grade.breakdown["unnecessary_step_penalty"] = -0.1
            grade.reason = f"{grade.reason} Unnecessary step penalty applied."
            self._cumulative_reward -= 0.1

        self._update_history(action)
        if is_valid_action:
            self._advance_workflow(action, done=grade.done or should_end_by_task)

        timeout_reached = False
        if not self._done and self._steps >= self.max_steps:
            timeout_reached = True
            timeout_penalty = 0.1
            adjusted_score = grade.score - timeout_penalty
            self._cumulative_reward += adjusted_score - grade.score
            self._done = True
            self._history.append(
                ConversationTurn(
                    speaker="system",
                    message="Maximum step limit reached. Episode terminated.",
                )
            )
            grade.breakdown["max_steps_penalty"] = -timeout_penalty
            grade.reason = f"{grade.reason} Episode terminated due to max steps."
            grade.score = adjusted_score

        reward = Reward(
            value=grade.score,
            reason=grade.reason,
            breakdown=grade.breakdown,
        )
        observation = self._build_observation()
        info = {
            "id": self._ticket.id,
            "issue_type": self._ticket.issue_type.value,
            "ticket_id": self._ticket.ticket_id,
            "expected_resolution": self._ticket.expected_resolution,
            "objective": self._task_definition.objective,
            "success_condition": self._task_definition.success_condition,
            "required_stages": self._task_definition.required_stages,
            "step": self._steps,
            "max_steps": self.max_steps,
            "timed_out": timeout_reached,
            "unnecessary_step_penalty_applied": unnecessary_step_penalty_applied,
            "stage": self._stage.value,
            "status": self._status.value,
            "cumulative_reward": self._cumulative_reward,
        }

        result = StepResult(observation=observation, reward=reward, done=self._done, info=info)
        return result.observation, result.reward, result.done, result.info

    def _build_observation(self) -> Observation:
        if self._ticket is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        if self._task_definition is None:
            raise RuntimeError("Task definition missing. Call reset() first.")
        return Observation(
            id=self._ticket.id,
            issue_type=self._ticket.issue_type,
            ticket_id=self._ticket.ticket_id,
            stage=self._stage,
            status=self._status,
            summary=self._ticket.summary,
            sentiment=self._ticket.sentiment,
            urgency=self._ticket.urgency,
            conversation_history=self._history.copy(),
            allowed_actions=self._allowed_actions(),
            metadata={
                "difficulty": self._ticket.difficulty,
                "customer_tier": self._ticket.customer_tier,
                "requires_escalation": self._ticket.requires_escalation,
                "objective": self._task_definition.objective,
                "success_condition": self._task_definition.success_condition,
                "required_stages": self._task_definition.required_stages,
            },
        )

    def _allowed_actions(self) -> List[ActionType]:
        if self._stage == WorkflowStage.CLASSIFY:
            return [ActionType.CLASSIFY]
        if self._stage == WorkflowStage.RESPOND:
            return [ActionType.RESPOND]
        if self._stage == WorkflowStage.RESOLVE:
            return [ActionType.RESOLVE]
        return []

    def _validate_action_for_stage(self, action: Action) -> bool:
        expected = {
            WorkflowStage.CLASSIFY: ActionType.CLASSIFY,
            WorkflowStage.RESPOND: ActionType.RESPOND,
            WorkflowStage.RESOLVE: ActionType.RESOLVE,
        }[self._stage]
        return action.action_type == expected

    def _should_end_by_task_requirements(self, score: float) -> bool:
        """
        End episode according to difficulty-specific task requirements.
        """
        if self._task_definition is None:
            raise RuntimeError("Task definition missing. Call reset() first.")

        # Easy: one-step classification task.
        if self._task_definition.difficulty == "easy":
            return self._stage == WorkflowStage.CLASSIFY

        # Medium: must complete classify and respond; require non-negative response quality.
        if self._task_definition.difficulty == "medium":
            return self._stage == WorkflowStage.RESPOND and score >= 0.2

        # Hard: full pipeline managed by regular resolve behavior.
        return False

    def _update_history(self, action: Action) -> None:
        if action.response_text:
            self._history.append(ConversationTurn(speaker="agent", message=action.response_text))

        if action.action_type == ActionType.CLASSIFY and action.classification:
            self._history.append(
                ConversationTurn(
                    speaker="system",
                    message=f"Classification set to '{action.classification.value}'.",
                )
            )
        if action.action_type == ActionType.RESOLVE and action.resolve is not None:
            resolution_text = "Ticket marked as resolved." if action.resolve else "Resolution deferred."
            self._history.append(ConversationTurn(speaker="system", message=resolution_text))
        if action.escalate:
            self._history.append(ConversationTurn(speaker="system", message="Ticket escalated to specialist team."))

    def _advance_workflow(self, action: Action, done: bool) -> None:
        if done:
            self._done = True
            if action.resolve:
                self._status = TicketStatus.RESOLVED
            else:
                self._status = TicketStatus.IN_PROGRESS
            return

        if self._stage == WorkflowStage.CLASSIFY:
            self._stage = WorkflowStage.RESPOND
            self._status = TicketStatus.IN_PROGRESS
        elif self._stage == WorkflowStage.RESPOND:
            if self._task_definition and self._task_definition.difficulty == "medium":
                # Medium tasks should allow response retries until success or max steps.
                self._stage = WorkflowStage.RESPOND
            else:
                self._stage = WorkflowStage.RESOLVE
            self._status = TicketStatus.IN_PROGRESS
        elif self._stage == WorkflowStage.RESOLVE:
            self._done = True
            self._status = TicketStatus.RESOLVED if action.resolve else TicketStatus.IN_PROGRESS
