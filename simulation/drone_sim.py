"""
Raw PyBullet quadcopter body: creation, force/torque application, and state
readout. This file knows nothing about "forward" or "hover" - it only knows
how to build a rigid body and push it around with forces. All flight logic
(stabilization, high-level commands) lives in interfaces/pybullet_drone.py.
"""

import pybullet as p


class QuadcopterBody:
    """A single rigid body standing in for a quadcopter's frame.

    v1 keeps this to one box-shaped rigid body (no articulated rotor links)
    - thrust and torques are applied directly at its center of mass. This is
    a common simplification for early-stage drone sims: it skips modeling
    each of the 4 individual motors, but still obeys real rigid-body physics
    (tilt it and thrust redirects horizontally, exactly like a real quad).
    """

    def __init__(self, start_pos=(0, 0, 0), mass=1.0, half_extents=(0.15, 0.15, 0.04)):
        self.mass = mass

        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.9, 0.2, 0.2, 1]
        )

        self.id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=start_pos,
        )

        # pybullet bodies have ~zero drag by default, so once moving the
        # drone would coast on pure momentum until the attitude PID
        # actively tilts back to brake it - which feels sluggish/floaty
        # for manual flying. Real drones have real air resistance; giving
        # it some here makes releasing a key bring it to a stop quickly
        # instead of drifting, without needing to rely on the PID alone.
        p.changeDynamics(self.id, -1, linearDamping=0.9, angularDamping=0.9)

        # v1 keeps the body as a plain box - no decorative rotor arms. An
        # earlier attempt attached 4 arm markers via createConstraint to
        # separate zero-mass bodies, which destabilized the physics (fixed
        # constraints to massless bodies fight the solver every step). A
        # nicer-looking mesh can replace this box later without touching
        # any control code, as long as it keeps mass/COM reasonable.

        self.start_pos = start_pos

    def apply_control(self, thrust, torque_roll, torque_pitch, torque_yaw):
        """Apply one control cycle's worth of force/torque.

        thrust: total upward force in Newtons, along the body's OWN +Z axis
            (not world +Z) - this is what makes tilting the body redirect
            thrust sideways, same as a real quadcopter.
        torque_roll/pitch/yaw: Nm, about the body's own X/Y/Z axes.

        LINK_FRAME means both are expressed in the body's local frame, so we
        don't have to manually rotate them by the current orientation.
        """
        p.applyExternalForce(
            self.id, -1,
            forceObj=[0, 0, thrust],
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
        )
        p.applyExternalTorque(
            self.id, -1,
            torqueObj=[torque_roll, torque_pitch, torque_yaw],
            flags=p.LINK_FRAME,
        )

    def get_state(self):
        position, orientation = p.getBasePositionAndOrientation(self.id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.id)
        return {
            "position": position,
            "orientation": orientation,  # quaternion
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }

    def get_contacts(self):
        return p.getContactPoints(bodyA=self.id) or []

    def closest_distance(self, other_body_id, max_distance=5.0):
        pts = p.getClosestPoints(self.id, other_body_id, distance=max_distance)
        if not pts:
            return None
        return min(pt[8] for pt in pts)  # contactDistance field

    def reset(self, start_pos=None):
        pos = start_pos if start_pos is not None else self.start_pos
        p.resetBasePositionAndOrientation(self.id, pos, [0, 0, 0, 1])
        p.resetBaseVelocity(self.id, [0, 0, 0], [0, 0, 0])
