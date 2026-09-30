"""
Per-trial metrics collection, plus aggregation/printing across many trials.

`TrialMetrics.update(...)` is meant to be called once per DECISION cycle
(the ~20-30 FPS vision/controller loop), not once per physics tick.
"""

import math

CONFIRM_WINDOW_STEPS = 45  # ~1.5s at ~30 decision-cycles/sec: how long an
                            # avoidance turn has to stay crash-free to count
                            # as "successful"


class TrialMetrics:
    def __init__(self, goal_x):
        self.goal_x = goal_x
        self.start_pos = None
        self._last_pos = None

        self.decision_steps = 0
        self.distance_traveled = 0.0
        self.min_obstacle_distance = float("inf")
        self._speed_sum = 0.0
        self._speed_samples = 0
        self.emergency_stops = 0
        self.collisions = 0
        self.completed = False
        self.crashed = False

        self.avoidance_attempts = 0
        self.avoidance_successes = 0
        self._was_avoiding = False
        self._avoidance_pending_deadline = None

    def start(self, position):
        self.start_pos = position
        self._last_pos = position

    def update(self, position, speed, obstacle_distance, emergency_stop_count,
               collided, is_avoidance_command):
        self.decision_steps += 1

        dx = position[0] - self._last_pos[0]
        dy = position[1] - self._last_pos[1]
        self.distance_traveled += math.sqrt(dx * dx + dy * dy)
        self._last_pos = position

        self._speed_sum += speed
        self._speed_samples += 1

        if obstacle_distance is not None:
            self.min_obstacle_distance = min(self.min_obstacle_distance, obstacle_distance)

        if emergency_stop_count > self.emergency_stops:
            self.emergency_stops = emergency_stop_count
            self.collisions = emergency_stop_count

        if collided:
            self.crashed = True
            self._avoidance_pending_deadline = None  # any pending attempt just failed

        if position[0] >= self.goal_x:
            self.completed = True

        # Rising-edge detection: count a new avoidance attempt only when a
        # turn command *starts*, not on every cycle it's held.
        if is_avoidance_command and not self._was_avoiding:
            self.avoidance_attempts += 1
            self._avoidance_pending_deadline = self.decision_steps + CONFIRM_WINDOW_STEPS
        self._was_avoiding = is_avoidance_command

        if (self._avoidance_pending_deadline is not None
                and self.decision_steps >= self._avoidance_pending_deadline
                and not self.crashed):
            self.avoidance_successes += 1
            self._avoidance_pending_deadline = None

    def finalize(self, decision_dt):
        self.flight_time = self.decision_steps * decision_dt
        self.average_speed = self._speed_sum / max(1, self._speed_samples)
        self.successful = self.completed and not self.crashed
        return self

    def summary_line(self):
        return (
            f"completed={self.completed} crashed={self.crashed} "
            f"distance={self.distance_traveled:.2f}m time={self.flight_time:.1f}s "
            f"avg_speed={self.average_speed:.2f}m/s collisions={self.collisions} "
            f"avoidance={self.avoidance_successes}/{self.avoidance_attempts}"
        )


def print_summary(trials):
    n = len(trials)
    successful = sum(1 for t in trials if t.successful)
    collisions = sum(t.collisions for t in trials)
    avg_distance = sum(t.distance_traveled for t in trials) / max(1, n)
    avg_time = sum(t.flight_time for t in trials) / max(1, n)
    avg_speed = sum(t.average_speed for t in trials) / max(1, n)
    total_avoid_attempts = sum(t.avoidance_attempts for t in trials)
    total_avoid_success = sum(t.avoidance_successes for t in trials)
    avg_min_obstacle_dist = sum(
        t.min_obstacle_distance for t in trials if t.min_obstacle_distance != float("inf")
    ) / max(1, sum(1 for t in trials if t.min_obstacle_distance != float("inf")))

    print("\n=== Evaluation summary ===")
    print(f"Trials: {n}")
    print(f"Successful: {successful}")
    print(f"Collisions: {collisions}")
    print(f"Success rate: {successful / max(1, n) * 100:.0f}%")
    print(f"Average distance: {avg_distance:.2f} m")
    print(f"Average flight time: {avg_time:.1f} s")
    print(f"Average forward speed: {avg_speed:.2f} m/s")
    print(f"Average min obstacle distance: {avg_min_obstacle_dist:.2f} m")
    print(f"Avoidance events: {total_avoid_success}/{total_avoid_attempts} successful")
