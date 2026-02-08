import io
import os
from pathlib import Path

import numpy as np
import pygame
from matplotlib import pyplot as plt
from pygame import Vector2

from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.ui import colors
from src.ui.colors import bone_rgb, background_primary, light_background, background_secondary
from src.ui.image_manager import ImageManager
from src.ui.text_renderer import TextRenderer
from src.utils.creature_loader import CreatureLoader


class ImageGenerator:
    @staticmethod
    def generate_creature_image(creature: Creature, save_path: str):
        if os.path.exists(save_path):
            return

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pygame.init()
        TextRenderer.clear_all()
        ImageManager.clear_all()

        joints, bones, muscles = CreatureLoader.creature_to_ui(creature)

        min_x, min_y, max_x, max_y = float("inf"), float("inf"), float("-inf"), float("-inf")
        for joint in joints:
            x, y = joint.coords
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        p = 30
        width = max_x - min_x + 2 * p
        height = max_y - min_y + 2 * p

        target_ratio = 4 / 3
        if width / height > target_ratio:
            height = int(width / target_ratio)
        else:
            width = int(height * target_ratio)

        window = pygame.display.set_mode((width, height), pygame.SRCALPHA)
        ImageManager.load_for_simulation()

        offset_x = p - min_x + (width - (max_x - min_x + 2 * p)) // 2
        offset_y = p - min_y + (height - (max_y - min_y + 2 * p)) // 2

        surface = pygame.Surface((width, height))
        surface.fill(background_primary)
        if ImageManager.dots_grid_small:
            surface.blit(ImageManager.dots_grid_small, (0, 0))

        for joint in joints:
            joint.coords = Vector2(joint.coords) + Vector2(offset_x, offset_y)
            joint.show(surface)

        for bone in bones:
            a = Vector2(bone.joint1.coords)
            b = Vector2(bone.joint2.coords)
            bone.midpoint = ((a.x + b.x) / 2, (a.y + b.y) / 2)

        if ImageManager.dots_grid_small:
            surface.blit(ImageManager.dots_grid_small, (0, 0))

        for muscle in muscles:
            muscle.show(surface)

        for bone in bones:
            bone.show(surface)

        for joint in joints:
            joint.show(surface)

        pygame.image.save(surface, save_path)
        pygame.quit()
        ImageManager.clear_all()
        TextRenderer.clear_all()

    @staticmethod
    def generate_graph(array, filepath: str, title: str = "", name_x: str = "", name_y: str = ""):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)  # small figure
        ax.plot(array, color=light_background, linewidth=2)
        ax.set_title(title, color=light_background)
        fig.patch.set_color(background_secondary)
        ax.patch.set_color(background_secondary)
        ax.tick_params(colors=light_background)

        ax.set_xlabel(name_x, color=light_background)
        ax.set_ylabel(name_y, color=light_background)

        ax.spines['bottom'].set_color(background_secondary)
        ax.spines['top'].set_color(background_secondary)
        ax.spines['left'].set_color(background_secondary)
        ax.spines['right'].set_color(background_secondary)

        ax.grid(True, color=background_primary)
        fig.tight_layout()

        fig.savefig(filepath, format="png", dpi=100)
        plt.close(fig)

    @staticmethod
    def generate_comparison_graph(
            array1,
            array2,
            filepath: str,
            title: str = "",
            name_x: str = "",
            name_y: str = "",
            array1_name="A",
            array2_name="B",
            batch_size=12
    ):
        if len(array1) != len(array2):
            raise ValueError("Arrays must have the same length")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        ax.plot(array1, linewidth=2, label=array1_name)
        ax.plot(array2, linewidth=2, label=array2_name)

        x_equal = []
        y_equal = []
        num = 0
        num_per_batch = []
        for i, (a, b) in enumerate(zip(array1, array2)):
            if abs(a - b) < 1e-6:
                x_equal.append(i)
                y_equal.append(a)
                num += 1

            if (i + 1) % batch_size == 0:
                num_per_batch.append(num)
                num = 0

        if x_equal:
            ax.scatter(x_equal, y_equal, s=40, zorder=5, label="Equal", c="w")

        ax.set_title(title, color=light_background)
        fig.patch.set_color(background_secondary)
        ax.patch.set_color(background_secondary)
        ax.tick_params(colors=light_background)

        ax.set_xlabel(name_x, color=light_background)
        ax.set_ylabel(name_y, color=light_background)

        for spine in ax.spines.values():
            spine.set_color(background_secondary)

        ax.grid(True, color=background_primary)
        ax.legend()

        fig.tight_layout()
        fig.savefig(filepath, format="png", dpi=100)
        plt.close(fig)

        return num_per_batch

    @staticmethod
    def generate_pillar_graph(array, filepath: str,
                              title: str = "",
                              name_x: str = "",
                              name_y: str = "" ):
        array = np.asarray(array, dtype=int).flatten()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        x = np.arange(len(array))
        ax.bar(x, array)

        ax.set_title(title, color=light_background)
        fig.patch.set_color(background_secondary)
        ax.patch.set_color(background_secondary)
        ax.tick_params(colors=light_background)

        ax.set_xlabel(name_x, color=light_background)
        ax.set_ylabel(name_y, color=light_background)

        for spine in ax.spines.values():
            spine.set_color(background_secondary)

        ax.grid(axis="y", color=background_primary)

        fig.tight_layout()
        fig.savefig(filepath, format="png", dpi=100)
        plt.close(fig)

    @staticmethod
    def generate_activations_per_neuron_single_episode(
            activations_per_neuron: list[list[float]],
            filepath: str,
            title: str = "Neuron Activations (Single Episode)"
    ):
        if not activations_per_neuron:
            return

        activations_array = np.array(activations_per_neuron)
        num_neurons = activations_array.shape[1]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        for neuron_idx in range(num_neurons):
            ax.plot(activations_array[:, neuron_idx], label=f"Neuron {neuron_idx + 1}", linewidth=2)

        ax.set_title(title, color=light_background)
        fig.patch.set_color(background_secondary)
        ax.patch.set_color(background_secondary)
        ax.tick_params(colors=light_background)
        ax.set_xlabel("Step", color=light_background)
        ax.set_ylabel("Activation", color=light_background)
        ax.grid(True, color=background_primary)
        ax.legend()
        fig.tight_layout()
        fig.savefig(filepath, format="png", dpi=100)
        plt.close(fig)

    @staticmethod
    def generate_episode_comparison_grid(
            first_activations,
            first_rewards,
            best_activations,
            best_rewards,
            filepath: str,
            title: str = "First vs Best Episode"
    ):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, color=light_background)

        first_act = np.array(first_activations)
        for i in range(first_act.shape[1]):
            axs[0, 0].plot(first_act[:, i], label=f"Neuron {i}")
        axs[0, 0].set_title("First Episode – Activations")
        axs[0, 0].set_xlabel("Step")
        axs[0, 0].set_ylabel("Activation")

        axs[0, 1].plot(first_rewards)
        axs[0, 1].set_title("First Episode – Rewards per Step")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("Reward")

        best_act = np.array(best_activations)
        for i in range(best_act.shape[1]):
            axs[1, 0].plot(best_act[:, i], label=f"Neuron {i}")
        axs[1, 0].set_title("Best Episode – Activations")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Activation")

        axs[1, 1].plot(best_rewards)
        axs[1, 1].set_title("Best Episode – Rewards per Step")
        axs[1, 1].set_xlabel("Step")
        axs[1, 1].set_ylabel("Reward")

        fig.patch.set_facecolor(background_secondary)


        ImageGenerator._style_axis(
            axs[0, 0],
            "First Episode – Activations",
            "Step", "Activation",
            background_primary,
            background_secondary,
            light_background
        )

        ImageGenerator._style_axis(
            axs[0, 1],
            "First Episode – Rewards",
            "Step", "Reward",
            background_primary,
            background_secondary,
            light_background
        )

        ImageGenerator._style_axis(
            axs[1, 0],
            "Best Episode – Activations",
            "Step", "Activation",
            background_primary,
            background_secondary,
            light_background
        )

        ImageGenerator._style_axis(
            axs[1, 1],
            "Best Episode – Rewards",
            "Step", "Reward",
            background_primary,
            background_secondary,
            light_background
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filepath)
        plt.close()

    @staticmethod
    def _style_axis(ax, title, name_x, name_y,
                    background_primary,
                    background_secondary,
                    light_background):

        ax.set_title(title, color=light_background)

        ax.set_xlabel(name_x, color=light_background)
        ax.set_ylabel(name_y, color=light_background)

        ax.tick_params(colors=light_background)

        ax.set_facecolor(background_secondary)

        for spine in ax.spines.values():
            spine.set_color(background_secondary)

        ax.grid(axis="y", color=background_primary)