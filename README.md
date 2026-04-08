# AI Customer Support OpenEnv

## Project overview

`ai-support-openenv` is a modular Python environment for training and evaluating AI agents on customer support workflows.  
It simulates realistic ticket handling as a staged interaction:

1. classify the issue  
2. respond to the customer  
3. resolve the ticket

The project is structured for clarity and production-style extensibility:

- `models.py` - Pydantic schemas for observations, actions, rewards, and tickets
- `tasks.py` - task definitions and real-world ticket pools (`easy`, `medium`, `hard`)
- `grader.py` - deterministic scoring logic
- `environment.py` - environment state machine with `reset()`, `step()`, and `state()`
- `inference.py` - model-driven rollout runner across all tasks
- `openenv.yaml` - environment metadata/spec

## Real-world motivation

Real customer support is not a one-shot classification problem.  
Teams need agents that can:

- identify the issue type correctly
- communicate with empathy and actionable next steps
- close the loop with policy-compliant resolution decisions

This environment helps benchmark those capabilities with explicit stage-by-stage rewards and penalties, conversation history tracking, and realistic ticket examples (delayed orders, duplicate charges, poor support complaints, unresolved refunds).

## Observation space

Each step returns an `Observation` object (Pydantic model) with fields such as:

- `id` - canonical ticket ID
- `issue_type` - ground-truth type (`refund`, `delay`, `complaint`, `query`)
- `ticket_id` - external ticket reference
- `stage` - current workflow stage (`classify`, `respond`, `resolve`)
- `status` - ticket status (`open`, `in_progress`, `resolved`)
- `summary` - customer issue text
- `sentiment` - customer sentiment label
- `urgency` - integer urgency (`1` to `5`)
- `conversation_history` - ordered list of turns (`customer`, `agent`, `system`)
- `allowed_actions` - valid action types for current stage
- `metadata` - additional context (difficulty, customer tier, escalation requirement, objectives)

## Action space

Agents call `step(action)` with an `Action` object containing:

- `action_type` - one of `classify`, `respond`, `resolve`
- `classification` - predicted issue type (for classify stage)
- `response_text` - customer-facing response (for respond stage)
- `resolve` - boolean resolution decision (for resolve stage)
- `escalate` - whether to escalate to specialist support
- `notes` - optional structured notes

The environment validates stage/action alignment and applies scoring at every step.

## Task descriptions

### Easy

- **Objective:** classify ticket type correctly
- **Required stages:** `classify`
- **Success condition:** classification matches issue type

### Medium

- **Objective:** classify + produce an appropriate response
- **Required stages:** `classify`, `respond`
- **Success condition:** correct classification and response quality threshold met

### Hard

- **Objective:** complete full workflow over multiple steps
- **Required stages:** `classify`, `respond`, `resolve`
- **Success condition:** correct classification, quality response, and policy-compliant resolution

## Reward design

Rewards are step-wise (returned on each `step()`), not only at episode end.

- `+0.3` for correct classification
- `+0.3` for appropriate response (with partial keyword-based scoring)
- `+0.4` for correct resolution
- `-0.2` for wrong actions/incorrect outcomes
- `-0.1` for unnecessary extra steps

Additional environment behavior:

- deterministic graders for reproducibility
- cumulative reward tracking
- max-steps episode cap with timeout handling

## Setup instructions

## 1) Clone and enter project

```bash
git clone <your-repo-url>
cd ai-support-openenv
```

## 2) Create virtual environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

## 4) Configure environment variables

Set:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

### Windows (PowerShell)

```powershell
$env:API_BASE_URL="https://your-api-base/v1"
$env:MODEL_NAME="your-model-name"
$env:HF_TOKEN="your-token"
```

### macOS/Linux

```bash
export API_BASE_URL="https://your-api-base/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token"
```

## How to run `inference.py`

Run all tasks sequentially (`easy` -> `medium` -> `hard`):

```bash
python inference.py
```

Expected log markers:

- `[START]`
- `[STEP]`
- `[END]`

At completion, the script prints final scores for each task.

## Docker (optional)

Build:

```bash
docker build -t ai-support-openenv .
```

Run:

```bash
docker run --rm \
  -e API_BASE_URL="https://your-api-base/v1" \
  -e MODEL_NAME="your-model-name" \
  -e HF_TOKEN="your-token" \
  ai-support-openenv
```
