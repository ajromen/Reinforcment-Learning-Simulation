import sys
import math
import numpy as np
import pygame
import pymunk
from pymunk import pygame_util

WIDTH, HEIGHT = 1200, 800
FPS = 60
GROUND_Y = 600
MOTOR_RATE = 20

pygame.init()

window = pygame.display.set_mode((WIDTH, HEIGHT))

clock = pygame.time.Clock()
draw_options = pygame_util.DrawOptions(window)

space = pymunk.Space()
space.gravity = (0.0, 1200.0)

static_body = space.static_body
seg = pymunk.Segment(static_body, (0, GROUND_Y), (WIDTH, GROUND_Y), 2.0)
seg.friction = 1.0
seg.elasticity = 0.0
space.add(seg)

moment = pymunk.moment_for_box(1.2, (50, 10))
body1 = pymunk.Body(1.2, moment)
body1.position = (150, 400)
poly = pymunk.Poly.create_box(body1, (50, 10))
poly.friction = 1.0
poly.elasticity = 0.0
space.add(body1, poly)

moment = pymunk.moment_for_box(1.2, (50, 10))
body2 = pymunk.Body(1.2, moment)
body2.position = (200, 400)
poly = pymunk.Poly.create_box(body2, (50, 10))
poly.friction = 1.0
poly.elasticity = 0.0
space.add(body2, poly)

anchor = (175, 400)

joint = pymunk.PivotJoint(body1, body2, body1.world_to_local(anchor), body2.world_to_local(anchor))

joint.collide_bodies = False

rlimit = pymunk.RotaryLimitJoint(body1, body2, -math.radians(80), math.radians(80))
space.add(rlimit)

motor = pymunk.SimpleMotor(body1, body2, 0.0)
motor.max_force = 3e5
space.add(motor)

space.add(joint)

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if(e.key == pygame.K_UP):
                motor.rate = MOTOR_RATE
            elif e.key == pygame.K_DOWN:
                motor.rate = -MOTOR_RATE
            else:
                motor.rate = 0.0
        else:
            motor.rate = 0.0

    dt = 1.0 / FPS
    for _ in range(5):
        space.step(dt / 5.0)

    window.fill((30, 30, 30))
    space.debug_draw(draw_options)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
