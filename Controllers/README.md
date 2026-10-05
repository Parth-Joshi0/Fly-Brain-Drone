# Controllers

Everything that decides what the drone should do next, apart from the fly brain itself (`NeuralPathways/flybrain_controller.py`), plus the safety net that every command passes through. Each controller exposes the same `decide(flow, state) -> command dict` contract, so they're drop-in swaps for each other and for `FlyBrainController`.

## Files

| File | What it does |
|---|---|
| `commands.py` | The command dict every `decide()` returns (`EMPTY_CMD`), the states `SafetyLayer` treats as "already avoiding" (`AVOIDING_STATES`), and `apply_command()`, which turns one command into `DroneInterface` calls. No pybullet/cv2 imports, so the real-Tello scripts can use it too. |
| `safety_layer.py` | `SafetyLayer`: the obstacle-avoidance override that sits between any controller (neural, reflex FSM or manual) and the motors, and can override a command outright if something gets too close. |
| `reflex_controller.py` | `ReflexController`: the hand-written CRUISE/AVOID_LEFT/AVOID_RIGHT/WALL_ESCAPE/EMERGENCY_ESCAPE/BOUNDARY_RETURN state machine — the non-neural alternative to `FlyBrainController`. Used by `main.py` when `USE_FLYBRAIN = False`, and exclusively by `Simulator/Evaluation/run_trials.py`. No real-Tello script imports it. |
| `banana_seek_controller.py` | `BananaSeekController`: the real Tello's banana flight (`Drone/fly_tello.py`) behind the same `decide(flow, state)` contract — `BananaModel`'s detector on the camera frame drives `NeuralPathways/FoodNeuron/feeding_behaviour.py`, and the fly brain (escape, plus DNg02 yaw stabilization if built with `optomotor=True`) runs alongside as a fear reflex. `main.py` uses it when `USE_BANANA = True`, and flies it without `SafetyLayer`, as the Tello does. See its docstring for the two places it differs from the Tello on purpose (the brain's dodge flies the drone; detections are scaled to the Tello's frame size). |
| `manual_controller.py` | `ManualController`: keyboard flight. Reads the normalized input dict any `SimulatorInterface.poll_input()` produces, so it works unchanged whether input came from the PyBullet GUI or a real controller. |
| `boundary_math.py` | Stateless geometry helpers (`outside_bounds`, `well_inside_bounds`, `heading_rate_toward`, ...) for checking a position against the course's flight-area edges. Used by `reflex_controller.py` and `NeuralPathways/flybrain_controller.py` — but every real-Tello call site constructs `FlyBrainController(bounds=None, ...)`, and all of that controller's boundary handling is gated behind `if self.bounds is not None`, so in practice these only run against the simulator's bounded course. |
