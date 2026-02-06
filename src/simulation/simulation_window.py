import io
import math
import sys
import time
from typing import List

import numpy as np
import pygame
import pymunk
from matplotlib import pyplot as plt
from pygame import Vector2
from pymunk import pygame_util

from src.agents.agent import Agent
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.simulation.simulation_settings import SimulationSettings
from src.simulation.simulation_stats import SimulationStats
from src.ui.colors import background_secondary, foreground, light_background, background_dots, background_primary, \
    foreground_secondary, bone_rgb
from src.ui.image_manager import ImageManager
from src.ui.text_renderer import TextRenderer

from src.ui.ui_settings import FPS, WINDOW_WIDTH, WINDOW_HEIGHT
from src.simulation.simulation_settings import GROUND_Y, GROUND_FRICTION, NUM_OF_EPIOSDES_PER_SIMULATION, \
    NUM_OF_STEPS_PER_EPISODE, STOP_AT_SIMULATION_END, SKIP_MODEL_STEP, SHOW_MUSCLES, DEBUG_DRAW


class SimulationWindow:
    def __init__(self, creature: Creature, model: Agent, save_path, load_old=False):
        pygame.init()
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        ImageManager.load_for_simulation(ImageManager)
        self.space = pymunk.Space()
        self.settings = SimulationSettings()
        self.creature = CreaturePymunk(creature, self.space, self.settings)
        self.draw = pygame_util.DrawOptions(self.window)
        self.setup_space()
        self.model = model
        self.step = 0
        self.last_center: tuple[float, float] = self.creature.get_center()
        self.curr_center: tuple[float, float] = self.last_center
        self.save_path = save_path

        self.camera_offset = pygame.math.Vector2(0, 0)
        self.screen_center = pygame.math.Vector2(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)

        self.progress_graph = None

        self.stats = SimulationStats(NUM_OF_STEPS_PER_EPISODE)

        self.show_muscles = SHOW_MUSCLES
        self.debug_view = DEBUG_DRAW
        self.skip_model = SKIP_MODEL_STEP

    def start(self):
        clock = pygame.time.Clock()
        i = 0
        running = True
        alt = False
        visual = True
        ctrl = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.end_simulation()
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        visual = not visual
                    if not visual:
                        continue
                    elif event.key == pygame.K_n:
                        self.restart_episode()
                    elif event.key == pygame.K_LALT:
                        alt = True
                    elif event.key == pygame.K_LCTRL:
                        ctrl = True
                    elif event.key == pygame.K_e:
                        self.end_simulation()
                        return
                    elif event.key == pygame.K_m:
                        self.show_muscles = not self.show_muscles
                    elif event.key == pygame.K_d:
                        self.debug_view = not self.debug_view
                    elif event.key == pygame.K_p:
                        self.skip_model = not self.skip_model
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LALT:
                        alt = False
                    if event.key == pygame.K_LCTRL:
                        ctrl = False

            if i % 2 == 0:
                self.model_step()

            self.run_pymunk(visual)
            if i % 2 == 0:
                end = self.model_reward()
                if not visual and end:
                    print(
                        "Episode end \n Max dist: " + str(
                            self.stats.last_dist_per_episode[-1]) + "\n Starting episode: " + str(
                            self.stats.number_of_episodes + 1))

            i += 1
            if visual:
                self.show()
                self.show_ui(alt, ctrl, clock.get_fps())

            if STOP_AT_SIMULATION_END and NUM_OF_EPIOSDES_PER_SIMULATION <= self.stats.number_of_episodes:
                self.end_simulation()
                return

            if visual:
                clock.tick(FPS)
                pygame.display.flip()

    def load_everything(self):
        self.model.load_from_file(self.save_path + "model.pt")
        self.stats.load_from_file(self.save_path + "stats.json")
        self.settings.load_from_file(self.save_path + "settings.json")

    def setup_space(self):
        self.space.gravity = (0, self.settings.gravity)
        self._place_ground()

    def end_simulation(self):
        self.model.end_simulation(self.save_path + "model.pt")
        self.stats.save_to_file(self.save_path + "stats.json")
        self.settings.save_to_file(self.save_path + "settings.json")

    def _place_ground(self):
        static_body = self.space.static_body
        self.ground = pymunk.Segment(static_body, (0, GROUND_Y), (WINDOW_WIDTH, GROUND_Y), 5.0)
        self.ground.friction = GROUND_FRICTION
        self.ground_1 = pymunk.Segment(static_body, (WINDOW_WIDTH, GROUND_Y), (WINDOW_WIDTH * 2, GROUND_Y), 5.0)
        self.ground_1.friction = GROUND_FRICTION
        self.space.add(self.ground_1)
        self.space.add(self.ground)

    def show_ui(self, alt, ctrl, fps):
        TextRenderer.render_text("Steps: " + str(self.step) + "/" + str(NUM_OF_STEPS_PER_EPISODE), 15, foreground,
                                 (10, 10), self.window)
        TextRenderer.render_text(
            "Episodes: " + str(self.stats.number_of_episodes) + "/" + str(NUM_OF_EPIOSDES_PER_SIMULATION), 15,
            foreground, (10, 30), self.window)

        if alt:
            TextRenderer.render_text("N = Next episode", 16, foreground, (10, 50), self.window)
            TextRenderer.render_text("V = Toggle visual (increases simulation speed to maximum)", 16, foreground,
                                     (10, 70), self.window)
            TextRenderer.render_text("E = End simulation", 16, foreground, (10, 90), self.window)
            TextRenderer.render_text("M = Remove muscles visually", 16, foreground, (10, 110), self.window)
            TextRenderer.render_text("D = Debug view", 16, foreground, (10, 130), self.window)
            TextRenderer.render_text("P = Pause/Unpause model (only for testing)", 16, foreground, (10, 150),
                                     self.window)
        else:
            TextRenderer.render_text("Hold L_ALT to se options", 16, foreground, (10, 50), self.window)

        if ctrl:
            TextRenderer.render_text("FPS: " + f'{fps:.2f}', 16, foreground, (WINDOW_WIDTH - 300, 210), self.window)
            TextRenderer.render_text("Max dist (episode): " + f"{self.stats.get_dist_m():.2f}m", 16,
                                     foreground,
                                     (WINDOW_WIDTH - 300, 230), self.window)
            TextRenderer.render_text("Max dist (total): " + f'{self.stats.max_dist:.2f}m', 16, foreground,
                                     (WINDOW_WIDTH - 300, 250), self.window)
            TextRenderer.render_text("Last reward: " + f"{self.stats.last_reward:.2f}", 16, foreground,
                                     (WINDOW_WIDTH - 300, 270),
                                     self.window)

            formatted = self.stats.get_elapsed_time()
            TextRenderer.render_text("Elapsed time: " + formatted, 16, foreground, (WINDOW_WIDTH - 300, 290),
                                     self.window)
            TextRenderer.render_text("Last episode time: " + self.stats.get_last_episode_time(), 16, foreground,
                                     (WINDOW_WIDTH - 300, 310),
                                     self.window)
            TextRenderer.render_text("Last episode rewards: " + self.stats.get_last_episode_reward(), 16, foreground,
                                     (WINDOW_WIDTH - 300, 330),
                                     self.window)
            TextRenderer.render_text("Last episode activation: " + self.stats.get_last_episode_activation(), 16,
                                     foreground,
                                     (WINDOW_WIDTH - 300, 350),
                                     self.window)
        else:
            TextRenderer.render_text("Hold L_CTRL to see more info", 16, foreground, (WINDOW_WIDTH - 300, 210),
                                     self.window)
        if self.progress_graph:
            self.window.blit(self.progress_graph, (WINDOW_WIDTH - self.progress_graph.get_width() - 10, 10))

    def _plot_distance_surface(self):
        if self.stats.number_of_episodes == 0:
            return None

        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)  # small figure
        ax.plot(self.stats.dist_per_episode, color=light_background, linewidth=2)
        ax.set_title("Max distances per episode", color=light_background)
        fig.patch.set_color(background_secondary)
        ax.patch.set_color(background_secondary)
        ax.tick_params(colors=light_background)

        ax.spines['bottom'].set_color(background_secondary)  # axes lines
        ax.spines['top'].set_color(background_secondary)
        ax.spines['left'].set_color(background_secondary)
        ax.spines['right'].set_color(background_secondary)

        ax.grid(True, color=background_primary)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        img = pygame.image.load(buf).convert_alpha()
        return img

    def show(self):
        self.move_camera(self.curr_center)
        self.move_ground()

        self.window.fill(background_secondary)
        self.draw_ground_markers()
        if self.debug_view:
            self.space.debug_draw(self.draw)
            return

        off_x = self.screen_center[0] - self.camera_offset.x - WINDOW_WIDTH // 2

        if self.show_muscles:
            for motor in self.creature.motors:
                a = Vector2(motor.a.position.x - off_x, motor.a.position.y)
                b = Vector2(motor.b.position.x - off_x, motor.b.position.y)

                d = b - a
                length = d.length()
                if length == 0:
                    return

                angle = d.angle_to(Vector2(1, 0))
                img = ImageManager.muscle

                scaled = pygame.transform.smoothscale(img, (int(length), int(self.settings.scale)))
                rotated = pygame.transform.rotate(scaled, angle)
                mid = (a + b) * 0.5
                rect = rotated.get_rect(center=mid)
                self.window.blit(rotated, rect)

        for bone_id, bone_body in self.creature.bodies.items():
            shape = self.creature.body_shapes[bone_id]
            # get box corners rotated by body angle
            points = shape.get_vertices()
            points = [p.rotated(bone_body.angle) + bone_body.position for p in points]
            points = [(p.x - off_x, p.y) for p in points]  # adjust camera
            pygame.draw.polygon(self.window, bone_rgb, points)

        for hub in self.creature.hubs.values():
            pos = (hub.position.x - off_x, hub.position.y)
            pygame.draw.circle(self.window, bone_rgb, pos, 5)

        ground_a = (self.ground.a.x - off_x, self.ground.a.y - 5)
        ground_b = (self.ground.b.x - off_x, self.ground.b.y - 5)
        pygame.draw.rect(self.window, background_primary,
                         (ground_a[0], ground_a[1], ground_b[0] - ground_a[0], 500),
                         border_radius=5)

        ground_a = (self.ground_1.a.x - off_x, self.ground_1.a.y - 5)
        ground_b = (self.ground_1.b.x - off_x, self.ground_1.b.y - 5)
        pygame.draw.rect(self.window, background_primary,
                         (ground_a[0], ground_a[1], ground_b[0] - ground_a[0], 500),
                         border_radius=5)

    def draw_ground_markers(self):
        left = -self.camera_offset.x
        right = left + WINDOW_WIDTH
        y = GROUND_Y
        color = foreground_secondary

        for x in range(int(left // 50) * 50, int(right), 50):
            screen_x = x + self.camera_offset.x
            if x % 200 == 0:
                pygame.draw.line(self.window, color, (screen_x, y), (screen_x, y - 20), 2)
                TextRenderer.render_text(f"{x // 200 - 3} m", 16, color, (screen_x - 10, y - 50), self.window)
            else:
                pygame.draw.line(self.window, color, (screen_x, y), (screen_x, y - 10), 1)

    def move_camera(self, center):
        target = pygame.math.Vector2(self.screen_center.x - center[0], 0)
        self.camera_offset = target
        tx, ty = target
        self.draw.transform = pymunk.Transform(a=1, b=0, c=0, d=1, tx=tx, ty=ty)

    def move_ground(self):
        left = -self.camera_offset.x
        if self.ground.b.x < left:
            self.space.remove(self.ground)
            move = self.ground_1.b.x
            self.ground = pymunk.Segment(self.space.static_body, (move, GROUND_Y), (move + WINDOW_WIDTH, GROUND_Y),
                                         5.0)
            self.ground.friction = GROUND_FRICTION
            self.space.add(self.ground)

        elif self.ground_1.b.x < left:
            self.space.remove(self.ground_1)
            move = self.ground.b.x
            self.ground_1 = pymunk.Segment(self.space.static_body, (move, GROUND_Y), (move + WINDOW_WIDTH, GROUND_Y),
                                           5.0)
            self.ground_1.friction = GROUND_FRICTION
            self.space.add(self.ground_1)

    def run_pymunk(self, visual):
        PHYSICS_DT = 1 / 60

        for _ in range(self.settings.substeps):
            self.space.step(PHYSICS_DT / self.settings.substeps)

    def restart_episode(self):
        self.creature.restart()

        self.stats.episode_end(self.step, self.curr_center[0])
        center = self.creature.get_center()
        self.last_center = center
        self.curr_center = center

        self._place_ground()
        self.model.episode_end()
        self.progress_graph = self._plot_distance_surface()
        self.step = 0

    def model_step(self):
        if self.skip_model:
            return
        self.last_center = self.curr_center
        self.curr_center = self.creature.get_center()
        self.step += 1
        activation = self.model.step(self.creature.get_state())

        self.stats.act_sum = 0
        for i, a in enumerate(activation):
            self.creature.motors[i].rate = float(a) * self.settings.max_motor_rate
            self.stats.act_sum += abs(a)
        self.stats.activations += self.stats.act_sum

        self.stats.update_max_x(self.curr_center[0])

    def model_reward(self):
        cx1, cy = self.curr_center
        cx0, _ = self.last_center
        reward = (cx1 - cx0) * 5
        done = False

        # reward += cx1 - WINDOW_WIDTH // 2  # dodaje ukupnu udaljenost kao najveci faktor
        reward -= self.stats.act_sum

        # TODO is_upside_down
        # if self.creature.is_upside_down():
        #     reward -= 300
        #     done = True

        if cy > GROUND_Y + 10:
            done = True
            reward -= 1000
        elif self.step >= NUM_OF_STEPS_PER_EPISODE:
            done = True

        self.model.reward(reward)
        self.stats.curr_episode_rewards += reward
        self.stats.last_reward = reward

        if done:
            self.restart_episode()
            return True

        return False
