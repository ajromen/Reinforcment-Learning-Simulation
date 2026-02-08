import json
import os


class SimulationSettings:
    def __init__(self):
        self.bone_width = 10.0
        self.body_friction = 1.0
        self.body_mass = 0.8
        self.overlap_allowed = False
        self.joint_limits = True
        self.scale = 15
        self.max_motor_force = 4e6
        self.substeps = 30
        self.max_joint_angle = 20
        self.min_joint_angle = 5
        self.add_sprints = False
        self.gravity = 1200
        self.max_motor_rate = 18
        self.expected_max_angular_velocity = 15
        self.expected_max_linear_velocity = 15

    def save_to_file(self, filepath):
        data = {
            "bone_width": self.bone_width,
            "body_friction": self.body_friction,
            "body_mass": self.body_mass,
            "overlap_allowed": self.overlap_allowed,
            "scale": self.scale,
            "max_motor_force": self.max_motor_force,
            "substeps": self.substeps,
            "joint_limits": self.joint_limits,
            "max_joint_angle": self.max_joint_angle,
            "min_joint_angle": self.min_joint_angle,
            "add_springs": self.add_sprints,
            "gravity": self.gravity,
            "max_motor_rate": self.max_motor_rate,
            "expected_max_angular_velocity": self.expected_max_angular_velocity,
            "expected_max_linear_velocity": self.expected_max_linear_velocity,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Settings file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.bone_width = data["bone_width"]
        self.body_friction = data["body_friction"]
        self.body_mass = data["body_mass"]
        self.overlap_allowed = data["overlap_allowed"]
        self.joint_limits = data["joint_limits"]
        self.scale = data["scale"]
        self.max_motor_force = data["max_motor_force"]
        self.substeps = data["substeps"]
        self.max_joint_angle = data["max_joint_angle"]
        self.min_joint_angle = data["min_joint_angle"]
        self.add_sprints = data["add_springs"]
        self.gravity = data["gravity"]
        self.max_motor_rate = data["max_motor_rate"]
        self.expected_max_angular_velocity = data["expected_max_angular_velocity"]
        self.expected_max_linear_velocity = data["expected_max_linear_velocity"]


SKIP_MODEL_STEP = False
DEBUG_DRAW = False
SHOW_MUSCLES = True
GROUND_Y = 700
GROUND_FRICTION = 1.0

# training
NUM_OF_STEPS_PER_EPISODE = 540  # 18s vizuelno
NUM_OF_EPIOSDES_PER_SIMULATION = 1500
STOP_AT_SIMULATION_END = False
