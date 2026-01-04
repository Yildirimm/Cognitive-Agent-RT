### Day 1 — Simulation + Manual Control

Run:
```bash
pip install -r requirements.txt
python manual_control.py
```

---
- Notes: hit **R** to restart, **Q** to quit
---

### Day 2 — State Extraction + Motor Primitives

**Goal:**  
Extract a stable symbolic state and implement deterministic motor primitives and a simple goal-directed controller (no learning, no LLM; purely rule-based).

#### What Works
- Symbolic state extraction (`robot_pose`, `target_dist`, `target_bearing`, obstacle distances)
- Bearing correctly reflects target position (verified by moving the target off-axis)
- Proportional steering enables smooth goal tracking
- Motor primitives are deterministic and reusable
- Agent reacts immediately to changes in target location

#### Known limitation (by design)
> The agent correctly tracks the target using symbolic state and proportional steering.  
> However, in PyBullet’s racecar model, velocity-controlled wheels can stall under low-speed turning due to static friction.  
> This limits task completion despite correct high-level control logic.  
> This issue is intentionally left unresolved until **Day 3 (feedback & recovery)** and **Phase B (real-time control)**.

This is a physics / actuation limitation, not a perception or decision-making error.

#### How to Run (Day 2)

From the project root:

```bash
python3 -m agent.loop
```

###  Day 3 — Closed-Loop Cognitive Agent (Baseline)

Implemented a fully closed perception–action loop in PyBullet.

**Features**
- Symbolic state extraction (pose, bearing, distances)
- Rule-based navigation baseline
- Obstacle avoidance via ray perception
- Contact-based collision detection
- Stuck detection and recovery behaviors
- Deterministic episode termination with target latch

This baseline serves as a stable foundation for:
- LLM-based planning (Day 4)
- Real-time constrained control (Phase B)
```
bash
python -m eval.run_eval

```

### Day 4 — Closed-Loop Cognitive Agent (LLM-Augmented)

#### Goal:
Extend the baseline agent with a symbolic LLM planner, while preserving safety, determinism, and real-time constraints.

The LLM does not control the robot directly.
It proposes high-level symbolic actions that are gated and overridden when necessary.

#### Architecture Overview
```
Observe
  ↓
Symbolic State
  ↓
LLM Planner (optional, ~4 Hz)
  ↓
Action Gate (safety + recovery)
  ↓
Fallback Controller (~20 Hz)
  ↓
Act
  ↓
Log
```

#### Core Features
##### Symbolic Perception

- Target distance & bearing

- Ray-based obstacle distances (front / left / right)

- Contact and collision detection

- Motion-based stuck detection

- Terminal target latch

##### Hybrid Decision Making

- Rule-based fallback controller executes every step
- LLM planner proposes actions intermittently
- LLM outputs are:
    - JSON-only
    - schema-constrained
    - symbolically grounded
    - ignored during recovery

##### Real-Time Awareness
- Cognition (LLM): ~4 Hz (step-based throttling)
- Control (physics): ~20 Hz
- No blocking calls in the control loop
- Safe degradation when LLM is slow or unavailable

##### Robustness Mechanisms
- Collision-triggered recovery
- Directional recovery using contact normals
- Motion-based stuck detection
- Deterministic episode termination

##### Structured LLM Interface
- Symbolic action space:
    - forward
    - turn_left
    - turn_right
    - stop
- Strict JSON schema enforcement
- Deterministic decoding (temperature = 0)
- Automatic fallback on malformed or missing output

##### Logging & Evaluation
- Per-step logging:
    - symbolic state
    - executed action
    - LLM action (if any)
    - fallback usage

- Episode-level summaries:
    - success
    - steps
    - collisions
    - termination reason

#### Running the Day 4 Evaluation
```
python -m eval.run_eval
```

##### Configuration
LLM usage is controlled via configs/task.yaml:

```
use_llm: true   # or false for pure baseline
```

API keys are loaded via ```.env``` (not committed):
```
GEMINI_API_KEY=your_key_here
```

##### Design Notes
* The LLM never bypasses safety or recovery logic
* The agent remains functional when:
    - LLM output is invalid
    - LLM is slow
    - LLM is disabled
* This architecture is directly compatible with Phase B (real-time constraints)