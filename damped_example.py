import sys
import pygame
import pymunk
from pymunk import pygame_util

WIDTH, HEIGHT = 1200, 800
FPS = 60

pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
draw = pygame_util.DrawOptions(window)

space = pymunk.Space()
space.gravity = (0, 1200)

static_body = space.static_body
ground = pymunk.Segment(static_body, (0, 600), (WIDTH, 600), 5.0)  # thickness = 5
ground.friction = 1.0
ground.elasticity = 0.0
space.add(ground)


def make_bone(x, y):
    body = pymunk.Body(1.2, pymunk.moment_for_box(1.2, (60,12)))
    body.position = x, y
    poly = pymunk.Poly.create_box(body, (60,12))
    poly.friction = 1.0

    space.add(body, poly)
    return body

boneA = make_bone(300, 300)
boneB = make_bone(360, 300)
boneC = make_bone(450, 360)

pivot = pymunk.PivotJoint(boneA, boneB, (330,300))
pivot.collide_bodies = False
space.add(pivot)

motorAB = pymunk.SimpleMotor(boneA, boneB, 0)
motorAB.max_force = 2e5
space.add(motorAB)

midA = boneA.position
midC = boneC.position
rest_length = (boneA.position - boneC.position).length
springAC = pymunk.DampedSpring(
    boneA, boneC,
    (0,0), (0,0),
    rest_length=rest_length,
    stiffness=5000,
    damping=300
)
space.add(springAC)


running = True
while running:
    keys = pygame.key.get_pressed()
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    if keys[pygame.K_UP]:
        motorAB.rate = 3
    elif keys[pygame.K_DOWN]:
        motorAB.rate = -3
    else:
        motorAB.rate = 0

    if keys[pygame.K_w]:
        springAC.rest_length = max(10, springAC.rest_length - 2)
    elif keys[pygame.K_s]:
        springAC.rest_length += 2

    dt = 1.0 / FPS
    for _ in range(5):
        space.step(dt / 5)

    window.fill((30,30,30))
    space.debug_draw(draw)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
