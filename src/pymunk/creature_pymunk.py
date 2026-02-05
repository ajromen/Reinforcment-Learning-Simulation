import math
import sys
from typing import Dict

import numpy as np
import pymunk
from pymunk import Body, Poly, SimpleMotor

from src.models.creature import Creature
from src.utils.constants import SIMULATION_BONE_WIDTH, BODY_FRICTION, BODY_MASS, SCALE, OVERLAP_ALLOWED, \
    MOTOR_MAX_FORCE, MAX_JOINT_ANGLE, WINDOW_WIDTH, WINDOW_HEIGHT, GROUND_Y, MAX_MOTOR_RATE, \
    EXPECTED_MAX_ANGULAR_VELOCITY, EXPECTED_MAX_LINEAR_VELOCITY, MIN_JOINT_ANGLE, JOINT_LIMITS, ADD_SPRINGS


class CreaturePymunk:
    def __init__(self, creature: Creature, space):
        self.creature = creature
        self.space = space

        self.hubs: Dict[str, Body] = {}
        self.bodies: Dict[str, Body] = {}
        self.body_shapes: Dict[str, Poly] = {}
        self.motors: list[SimpleMotor] = []
        self.bounds = []
        self.pivots = []
        self.limits = []
        self.num_of_joints = len(creature.joints)

        self.creature_to_pymunk()

        self.space.damping = 0.9

    def creature_to_pymunk(self):
        self._find_bounds_initial()
        self.create_joints()
        self.create_bones()
        self.create_muscles()
        if JOINT_LIMITS:
            self.create_limits()

    def debug_move(self, dx: float):
        for body in list(self.bodies.values()) + list(self.hubs.values()):
            body.position += (dx, 0)
            body.velocity = (0, 0)
            body.angular_velocity = 0

    def get_center(self):
        sum_x = 0
        sum_y = 0
        for j in self.hubs.values():
            x, y = j.position
            sum_y += y
            sum_x += x

        return sum_x / self.num_of_joints, sum_y / self.num_of_joints

    def _find_bounds_initial(self):
        max_w = 0
        max_h = 0

        for j in self.creature.joints:
            if j.x > max_w:
                max_w = j.x
            if j.y > max_h:
                max_h = j.y
        self.bounds = [max_w, max_h]

    def _find_bounds(self):
        min_x = math.inf
        max_x = -math.inf
        min_y = math.inf
        max_y = -math.inf

        for hub in self.hubs.values():
            p = hub.position
            if p.x > max_x:
                max_x = p.x
            if p.x < min_x:
                min_x = p.x
            if p.y > max_y:
                max_y = p.y
            if p.y < min_y:
                min_y = p.y

        return min_x, max_x, min_y, max_y

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

            # body.damping = 0.9
            body.angular_damping = 0.9

            body.velocity_func = pymunk.Body.update_velocity

            shape = pymunk.Poly.create_box(body, (length, SIMULATION_BONE_WIDTH))
            shape.friction = BODY_FRICTION
            if OVERLAP_ALLOWED:
                shape.filter = pymunk.ShapeFilter(group=1)  # ako je 1 onda nema kolizije
            self.space.add(body, shape)

            self.bodies[b.id] = body
            self.body_shapes[b.id] = shape

            pj1 = pymunk.PivotJoint(body, self.hubs[j1.id], p1)
            pj2 = pymunk.PivotJoint(body, self.hubs[j2.id], p2)
            pj1.collide_bodies = False
            pj2.collide_bodies = False
            self.space.add(pj1, pj2)
            self.pivots.extend([pj1, pj2])

    def create_joints(self):
        for j in self.creature.joints:
            if len(j.bone_ids) < 1:
                continue

            hub = self.create_hub(self.world_pos(j.x, j.y))
            self.hubs[j.id] = hub

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

    def create_limits(self):
        MAX_ANGLE = math.radians(MAX_JOINT_ANGLE)
        MIN_ANGLE = math.radians(MIN_JOINT_ANGLE)
        visited: Dict[tuple, bool] = {}
        for joint in self.creature.joints:
            bodies: list[Body] = []
            for bid in joint.bone_ids:
                bodies.append(self.bodies[bid])

            for i in range(len(bodies)):
                for j in range(len(bodies)):
                    b1 = bodies[i]
                    b2 = bodies[j]
                    if b1 == b2:
                        continue

                    key = tuple(sorted([b1.id, b2.id]))
                    if key in visited:
                        continue
                    visited[key] = True


                    # ko je levo od koga
                    v1 = b1.rotation_vector
                    v2 = b2.rotation_vector

                    x = v1.cross(v2)
                    is_left = True if x > 0 else False

                    # realni ugao izmedju njih
                    a = (b2.angle-math.pi) % (2 * math.pi)
                    b = b1.angle % (2 * math.pi)
                    rest = b2.angle - b1.angle
                    diff = (a-b) % (2 * math.pi)
                    if diff > math.pi:
                        diff = 2 * math.pi - diff

                    MAX_BEND = diff

                    if is_left:
                        min_angle = rest - MAX_BEND
                        max_angle = rest + MAX_BEND - MIN_ANGLE
                    else:
                        min_angle = rest - MAX_BEND + MIN_ANGLE
                        max_angle = rest + MAX_ANGLE

                    limit = pymunk.RotaryLimitJoint(b1, b2, min_angle, max_angle)

                    # najbolja stvar ikad zaustavlja koliziju povezanih objekata
                    limit.collide_bodies = False
                    self.limits.append(limit)
                    self.space.add(limit)

                    if not ADD_SPRINGS:
                        continue

                    center_b1 = b1.center_of_gravity
                    center_b2 = b2.center_of_gravity
                    dist = center_b1.get_distance((center_b2.x, center_b2.y))
                    spring = pymunk.DampedSpring(b1, b2, (center_b1.x, center_b1.y), (center_b2.x, center_b2.y), dist,
                                                 10, 10)
                    self.limits.append(spring)
                    self.space.add(spring)

    def world_pos(self, x, y):
        x_pos = x * SCALE + WINDOW_WIDTH // 2 - self.bounds[0] * SCALE / 2
        y_pos = y * SCALE - self.bounds[1] * SCALE + GROUND_Y - 20
        return pymunk.Vec2d(x_pos, y_pos)

    def create_hub(self, pos):
        hub_mass = BODY_MASS
        hub_inertia = pymunk.moment_for_circle(hub_mass, 0, 6)
        hub = pymunk.Body(hub_mass, hub_inertia)
        hub.position = pos
        hub_shape = pymunk.Circle(hub, 5)
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

    def get_state(self):
        inp = []

        for m in self.motors:
            inp.append(np.clip(m.rate / MAX_MOTOR_RATE, -1, 1))

        for b in self.bodies.values():
            angle = b.angle
            inp.append(math.sin(angle))
            inp.append(math.cos(angle))

            angular_velocity = b.angular_velocity
            # ako je angular vel>20 x>1 -> clip an 1
            inp.append(np.clip(angular_velocity / EXPECTED_MAX_ANGULAR_VELOCITY, -1, 1))

        # pozicije hubova u odnosu na centar i brzine
        cx, cy = self.get_center()
        min_x, max_x, min_y, max_y = self._find_bounds()
        w = max(max_x - min_x, 1e-6)
        h = max(max_y - min_y, 1e-6)
        for hub in self.hubs.values():
            rx = np.clip((hub.position.x - cx) / w, -1, 1)
            ry = np.clip((hub.position.y - cy) / h, -1, 1)

            vx = np.clip(hub.velocity.x / EXPECTED_MAX_LINEAR_VELOCITY, -1, 1)
            vy = np.clip(hub.velocity.y / EXPECTED_MAX_LINEAR_VELOCITY, -1, 1)

            inp.extend([rx, ry, vx, vy])

        inp.append(np.clip((GROUND_Y - cy) / GROUND_Y, -1, 1))

        return np.array(inp)

    def restart(self):
        for bone_id, body in self.bodies.items():
            if body in self.space.bodies:
                self.space.remove(body)
            shape = self.body_shapes.get(bone_id)
            if shape and shape in self.space.shapes:
                self.space.remove(shape)

        for hub in self.hubs.values():
            for shape in hub.shapes:
                if shape in self.space.shapes:
                    self.space.remove(shape)
            if hub in self.space.bodies:
                self.space.remove(hub)

        for motor in self.motors:
            if motor in self.space.constraints:
                self.space.remove(motor)

        for limit in self.limits:
            self.space.remove(limit)

        for pivot in self.pivots:
            self.space.remove(pivot)

        self.bodies.clear()
        self.hubs.clear()
        self.body_shapes.clear()
        self.motors.clear()
        self.pivots.clear()
        self.limits.clear()

        self.creature_to_pymunk()
        
    def is_upside_down(self, threshold: float = 0.3) -> bool:
    
        return False


    @staticmethod
    def get_number_of_inputs(creature: Creature):
        j = len(creature.joints)
        m = len(creature.muscles)
        b = len(creature.bones)
        return j * 4 + b * 3 + m + 1  # dist do zemlje

    
    
    
