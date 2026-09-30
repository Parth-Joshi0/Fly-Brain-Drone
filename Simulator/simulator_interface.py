"""
Abstract contract between main.py (and ManualController) and whatever is
actually hosting the drone - the PyBullet sim today, real hardware (a
physical controller for manual mode, a real course with no obstacle IDs
to report) later. Mirrors Drone/drone_interface.py: main.py only
ever talks to this, never to PyBullet directly, so testing against the
real drone means writing one new class here and changing one line in
main.py - nothing in safety_layer.py/reflex_controller.py or neural_pathways/ needs to change.
"""

from abc import ABC, abstractmethod

# Normalized, platform-independent input state - what ManualController and
# main.py's emergency-key handling read. A concrete SimulatorInterface is
# responsible for turning whatever it actually reads (PyBullet key codes,
# a gamepad, a physical RC transmitter, ...) into this same shape, so
# nothing downstream needs to know or care which platform is running.
EMPTY_INPUT = {
    "forward": False,
    "backward": False,
    "strafe_left": False,
    "strafe_right": False,
    "up": False,
    "down": False,
    "yaw_left": False,
    "yaw_right": False,
    "hover": False,
    "land_pressed": False,
    "reset_pressed": False,
    "mode_toggle_pressed": False,
}


class SimulatorInterface(ABC):

    @abstractmethod
    def connect(self):
        """One-time setup: physics client / GUI window / gravity / test
        course, or connecting to real hardware. Returns
        {"bounds": {...}, "goal_x": float} describing the flight area, for
        the autonomous controller's soft boundary and the "course
        complete" check."""

    @abstractmethod
    def create_drone(self, start_pos):
        """Returns a DroneInterface instance for this platform."""

    @abstractmethod
    def poll_input(self):
        """Returns a dict shaped like EMPTY_INPUT describing the current
        manual/emergency input state."""

    @abstractmethod
    def update_debug_view(self, position, yaw_degrees):
        """Update any 3D visualization (debug camera, heading line). A
        no-op on real hardware."""

    @property
    @abstractmethod
    def physics_dt(self):
        """Seconds represented by one drone.step() call - used to convert
        a step count into elapsed time (e.g. for optical-flow
        derotation)."""

    @abstractmethod
    def tick(self):
        """Called once per loop iteration, after drone.step(): paces the
        loop (e.g. sleep to real time in sim; no-op / rate-limit on real
        hardware)."""

    def is_connected(self):
        """False once the simulator/hardware link is gone for good (e.g. the
        sim window was closed) - lets main.py tell that apart from a
        one-off failed command it can just retry."""
        return True

    def update_test_obstacles(self, drone_position, drone_yaw_degrees, dt):
        """Debug helper: on request (e.g. a mouse click), spawns a block
        that flies in a straight line at the drone, to trigger the escape
        reflex on demand instead of waiting to stumble into a real
        obstacle. Not abstract - only a simulator can conjure obstacles
        out of nowhere, so this defaults to a no-op (e.g. on real
        hardware) and a concrete sim overrides it."""
