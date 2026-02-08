import json
import math
import os.path
import time
from datetime import datetime

import numpy as np
from torch.cuda import seed_all

from src.ui.ui_settings import WINDOW_WIDTH


class EpisodeStats:
    def __init__(self, activations_per_neuron: list,
                 max_dist: float,
                 last_dist: float,
                 rewards_per_step: list,
                 time,
                 index):
        self.activations_per_neuron = activations_per_neuron
        self.max_dist = max_dist
        self.last_dist = last_dist
        self.rewards_per_step = rewards_per_step
        self.time = time
        self.index = index

    def to_dict(self):
        return {
            "activations_per_neuron": self.activations_per_neuron,
            "max_dist": self.max_dist,
            "last_dist": self.last_dist,
            "rewards_per_step": self.rewards_per_step,
            "time": self.time,
            "index": self.index
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            activations_per_neuron=data.get("activations_per_neuron", []),
            max_dist=data.get("max_dist", 0),
            last_dist=data.get("last_dist", 0),
            rewards_per_step=data.get("rewards_per_step", []),
            time=data.get("time", 0),
            index=data.get("index", 0)
        )


class SimulationStats:
    def __init__(self, steps_per_episode, device):
        # per simulation
        self.dist_per_episode = []
        self.time_per_episode = []
        self.rewards_per_episode = []
        self.last_dist_per_episode = []
        self.activation_per_episode = []
        self.start_time = time.time()
        self.simulation_time = 0
        self.number_of_episodes = 0
        self.steps_per_episode = steps_per_episode
        self.date_time = datetime.now()
        self.device = str(device)
        self.max_dist = 0
        self.first_episode_data: EpisodeStats | None = None
        self.best_episode_data: EpisodeStats | None = None

        # per episode
        self.activations_per_neuron = []
        self.rewards_per_step = []
        self.episode_start_time = time.time()
        self.last_reward = 0
        self.max_x_episode = 0
        self.act_sum = 0
        self.activations = 0
        self.is_first_episode = True

    def episode_end(self, steps, final_dist):
        final_dist = float((final_dist - WINDOW_WIDTH // 2) / 200)
        dist = self.get_dist_m()
        self.dist_per_episode.append(dist)
        duration = time.time() - self.episode_start_time
        if dist > self.max_dist:
            self.max_dist = dist
            self.best_episode_data = EpisodeStats(
                self.activations_per_neuron.copy(),
                self.max_dist,
                final_dist,
                self.rewards_per_step.copy(),
                duration,
                self.number_of_episodes
            )
        if self.number_of_episodes == 0:
            self.first_episode_data = EpisodeStats(
                self.activations_per_neuron.copy(),
                self.max_dist,
                final_dist,
                self.rewards_per_step.copy(),
                duration,
                0
            )
        self.number_of_episodes += 1
        self.time_per_episode.append(duration)
        self.episode_start_time = time.time()
        self.last_dist_per_episode.append(final_dist)

        if steps == 0:
            return

        self.activations_per_neuron = []

        reward = np.average(self.rewards_per_step)
        self.rewards_per_episode.append(float(reward))
        self.rewards_per_step = []

        activation = self.activations / steps
        self.activation_per_episode.append(float(activation))
        self.activations = 0

        self.max_x_episode = -math.inf

    def get_elapsed_time(self) -> str:
        elapsed = int(time.time() - self.start_time + self.simulation_time)
        return f"{elapsed // 3600:02d}h:{(elapsed % 3600) // 60:02d}m:{elapsed % 60:02d}s"

    def get_last_episode_time(self) -> str:
        if self.number_of_episodes == 0:
            return "00m:00s"
        last = int(self.time_per_episode[-1])
        return f"{(last % 3600) // 60:02d}m:{last % 60:02d}s"

    def get_last_episode_reward(self) -> str:
        if self.number_of_episodes == 0:
            return "0"
        return f"{self.rewards_per_episode[-1]:.2f}"

    def get_last_episode_activation(self) -> str:
        if self.number_of_episodes == 0:
            return "0"
        return f"{self.activation_per_episode[-1]:.2f}"

    def get_dist_m(self):
        return (self.max_x_episode - WINDOW_WIDTH // 2) / 200

    def update_max_x(self, center_x):
        if center_x > self.max_x_episode:
            self.max_x_episode = center_x

    def save_to_file(self, filepath):
        data = {
            "simulation_time": time.time() - self.start_time + self.simulation_time,
            "number_of_episodes": self.number_of_episodes,
            "max_dist": self.max_dist,
            "dist_per_episode": self.dist_per_episode,
            "time_per_episode": self.time_per_episode,
            "rewards_per_episode": self.rewards_per_episode,
            "last_dist_per_episode": self.last_dist_per_episode,
            "activation_per_episode": self.activation_per_episode,
            "date_time": self.date_time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.device,
            "first_episode_data": self.first_episode_data.to_dict() if self.first_episode_data else None,
            "best_episode_data": self.best_episode_data.to_dict() if self.best_episode_data else None,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

    def load_from_file(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Stats file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        self.simulation_time = data["simulation_time"]
        self.number_of_episodes = data["number_of_episodes"]
        self.max_dist = data["max_dist"]
        self.dist_per_episode = data["dist_per_episode"]
        self.time_per_episode = data["time_per_episode"]
        self.rewards_per_episode = data["rewards_per_episode"]
        self.last_dist_per_episode = data["last_dist_per_episode"]
        self.activation_per_episode = data["activation_per_episode"]
        self.date_time = datetime.strptime(data["date_time"], "%Y-%m-%d %H:%M:%S")
        self.device = data["device"]
        self.first_episode_data = EpisodeStats.from_dict(data["first_episode_data"]) if data[
            "first_episode_data"] else None
        self.best_episode_data = EpisodeStats.from_dict(data["best_episode_data"]) if data[
            "best_episode_data"] else None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()  # convert array to list
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)
