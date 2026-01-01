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

## Day 3 — Closed-Loop Cognitive Agent (Baseline)

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
