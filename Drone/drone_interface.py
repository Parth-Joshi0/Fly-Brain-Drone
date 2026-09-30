"""
Abstract contract between the FlyBrain/reflex controller and whatever is
actually flying the drone. The controller only ever talks to this - never
to PyBullet, MAVLink, the Tello SDK, or anything else directly.

To support a new drone later (real hardware, a different SDK), implement
a new class against this same interface and swap it in main.py. Nothing
in NeuralPathways/ or safety_layer.py/reflex_controller.py needs to change.
"""

from abc import ABC, abstractmethod


class DroneInterface(ABC):

    @abstractmethod
    def takeoff(self):
        """Climb to the configured hover altitude."""

    @abstractmethod
    def land(self):
        """Descend and shut down."""

    @abstractmethod
    def hover(self):
        """Zero out horizontal velocity and yaw rate; hold position."""

    @abstractmethod
    def move_forward(self, speed):
        """speed: desired forward velocity in m/s."""

    @abstractmethod
    def move_backward(self, speed):
        pass

    @abstractmethod
    def move_left(self, speed):
        pass

    @abstractmethod
    def move_right(self, speed):
        pass

    @abstractmethod
    def turn_left(self, rate):
        """rate: desired yaw rate in rad/s."""

    @abstractmethod
    def turn_right(self, rate):
        pass

    @abstractmethod
    def get_camera_frame(self):
        """Returns the latest forward camera frame as a BGR NumPy array."""

    @abstractmethod
    def get_state(self):
        """Returns a dict describing the drone's current status - altitude,
        speed, collision/emergency flags, etc. See PyBulletDrone.get_state
        for the exact fields this implementation provides."""

    @abstractmethod
    def step(self):
        """Advance one control cycle. Must be called every loop iteration
        for the low-level stabilization to actually run."""

    @abstractmethod
    def reset(self):
        """Return the drone to its start state (used between eval trials)."""
