import json
import math
import os.path
import time
from datetime import datetime

from src.ui.ui_settings import WINDOW_WIDTH


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

        # per episode
        self.episode_start_time = time.time()
        self.last_reward = 0
        self.curr_episode_rewards = 0
        self.max_x_episode = 0
        self.act_sum = 0
        self.activations = 0

    def episode_end(self, steps, final_dist):
        dist = self.get_dist_m()
        self.dist_per_episode.append(dist)
        if dist > self.max_dist:
            self.max_dist = dist
        self.number_of_episodes += 1
        self.time_per_episode.append(time.time() - self.episode_start_time)
        self.episode_start_time = time.time()
        self.last_dist_per_episode.append(float((final_dist - WINDOW_WIDTH // 2) / 200))

        if steps == 0:
            return

        reward = self.curr_episode_rewards / steps
        self.rewards_per_episode.append(float(reward))
        self.curr_episode_rewards = 0

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
            "max_dist": self.max_dist,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

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
        self.max_dist = data["max_dist"]
