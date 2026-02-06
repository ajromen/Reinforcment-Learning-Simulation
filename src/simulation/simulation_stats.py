import json
import math
import os.path
import time

from src.utils.constants import WINDOW_WIDTH


class SimulationStats:
    def __init__(self, steps_per_episode):
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

        # per episode
        self.episode_start_time = 0
        self.end_time = 0
        self.last_reward = 0
        self.curr_episode_rewards = []
        self.max_dist = 0
        self.max_x_episode = 0
        self.act_sum = 0
        self.activations = []

    def episode_end(self, steps, final_dist):
        self.number_of_episodes += 1
        self.time_per_episode.append(time.time() - self.episode_start_time)
        self.last_dist_per_episode.append(final_dist)

        reward = sum(self.curr_episode_rewards) / steps
        self.rewards_per_episode.append(reward)

        activation = sum(self.activations) / steps
        self.activation_per_episode.append(activation)

        self.max_x_episode = -math.inf

    def get_elapsed_time(self):
        return time.time() - self.start_time + self.simulation_time

    def get_dist_m(self):
        return (self.max_x_episode - WINDOW_WIDTH // 2) / 200

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
            "steps_per_episode": self.steps_per_episode,
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
        self.steps_per_episode = data["steps_per_episode"]
