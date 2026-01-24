import math
from venv import create

import pymunk
from pygame import Vector2

from src.models.bone import Bone
from src.models.creature import Creature
from src.models.joint import Joint
from src.utils.constants import SIMULATION_BONE_WIDTH, BODY_FRICTION, BODY_MASS, SCALE, OVERLAP_ALLOWED, MOTOR_MAX_FORCE
from working_example import rest_angle


class CreaturePymunk:
    def __init__(self, creature: Creature, space):
        self.creature = creature
        self.space = space

        self.hubs = {}
        self.bodies = {}
        self.body_shapes = {}
        self.motors = []
        
        self.creature_to_pymunk()

    def creature_to_pymunk(self):
        self.create_joints()

    def create_bones(self):
        for b in self.creature.bones:
            j1 = self.get_joint(b.joint1_id)
            j2 = self.get_joint(b.joint2_id)
            p1 = self.world_pos(j1.x, j1.y)
            p2 = self.world_pos(j2.x, j2.y)

            mid = (p1 + p2) * 0.5
            vec = p2 - p1
            length = max(vec.length, SIMULATION_BONE_WIDTH + 2)
            angle = math.atan2(vec.y, vec.x)

            mass = max(BODY_MASS, length / SCALE * BODY_MASS)
            moment = pymunk.moment_for_box(mass, (length, SIMULATION_BONE_WIDTH))
            body = pymunk.Body(mass, moment)
            body.position = mid
            body.angle = angle

            shape = pymunk.Poly.create_box(body, (length, SIMULATION_BONE_WIDTH))
            shape.friction = BODY_FRICTION
            shape.filter = pymunk.ShapeFilter(group=1)
            self.space.add(body, shape)

            self.bodies[b.id] = body
            self.body_shapes[b.id] = shape

            pj1 = pymunk.PivotJoint(body, self.hubs[j1.id], p1)
            pj2 = pymunk.PivotJoint(body, self.hubs[j2.id], p2)
            pj1.collide_bodies = OVERLAP_ALLOWED
            pj2.collide_bodies = OVERLAP_ALLOWED
            self.space.add(pj1, pj2)

            # rest_angle = body.angle
            # rs1 = pymunk.DampedRotarySpring(body, self.hubs[j1.id], rest_angle, stiffness=1e4, damping=1e2)
            # rs2 = pymunk.DampedRotarySpring(body, self.hubs[j2.id], rest_angle, stiffness=1e4, damping=1e2)

            # space.add(rs1, rs2)

    def create_joints(self):
        for j in self.creature.joints:
            if len(j.bone_ids) < 1:
                continue

            hub = self.create_hub(self.world_pos(j.x, j.y))
            self.hubs[j.id] = hub

            # for b_id in j.bone_ids:
            #     b = self.get_bone(b_id)

    def create_muscles(self):
        for m in self.creature.muscles:
            b1 = m.bone1_id
            b2 = m.bone2_id
            if b1 not in self.bodies or b2 not in self.bodies:
                continue

            motor = pymunk.SimpleMotor(self.bodies[b1], self.bodies[b2], 0.0)
            motor.max_force = MOTOR_MAX_FORCE
            self.space.add(motor)
            self.motors.append(motor)

            # motor_meta.append({
            #     "freq": random.uniform(0.6, 1.6),
            #     "phase": random.uniform(0, math.pi * 2),
            #     "amp": random.uniform(0.6, MOTOR_AMPLITUDE)
            # })

    def world_pos(self, x, y):
        return pymunk.Vec2d(x * SCALE, y * SCALE)


    def create_hub(self, pos):
        hub_mass = BODY_MASS
        hub_inertia = pymunk.moment_for_circle(hub_mass, 0, 6)
        hub = pymunk.Body(hub_mass, hub_inertia)
        hub.position = pos
        hub_shape = pymunk.Circle(hub, 5)
        # completely disable hub collisions (categories=0, mask=0)
        hub_shape.filter = pymunk.ShapeFilter(categories=0b0, mask=0b0)
        hub_shape.sensor = True
        self.space.add(hub, hub_shape)
        return hub

    def get_bone(self, bone_id):
        return self.creature.get_bone(bone_id)

    def get_joint(self, joint_id):
        return self.creature.get_joint(joint_id)

    def get_muscle(self, muscle_id):
        return self.creature.get_muscle(muscle_id)
