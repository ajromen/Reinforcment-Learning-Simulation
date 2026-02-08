from pathlib import Path

import numpy as np
from numpy.ma.core import equal

from src.agents.agent import Agent
from src.markdown.image_generator import ImageGenerator
from src.models.creature import Creature

from src.simulation.simulation_settings import SimulationSettings
from src.simulation.simulation_stats import SimulationStats
from src.ui.ui_settings import APP_NAME


class MarkdownMaker:
    def __init__(self,
                 creature: Creature,
                 save_path: str,
                 assets_path: str,
                 model: Agent,
                 settings: SimulationSettings,
                 stats: SimulationStats):
        self.text = ""
        self.assets_path = assets_path
        self.creature = creature
        self.model = model
        self.save_path = save_path
        self.settings = settings
        self.stats = stats

    def _generate_assets(self):
        ImageGenerator.generate_creature_image(self.creature, self.save_path + self.assets_path + "/creature.png")
        ImageGenerator.generate_graph(self.stats.dist_per_episode,
                                      self.save_path + self.assets_path + "/max_dist.png",
                                      "Max distance per episode",
                                      "Episode", "Distance")

        ImageGenerator.generate_graph(self.stats.last_dist_per_episode,
                                      self.save_path + self.assets_path + "/last_dist.png",
                                      "Last distance per episode",
                                      "Episode", "Distance")

        ImageGenerator.generate_graph(self.stats.time_per_episode,
                                      self.save_path + self.assets_path + "/time.png",
                                      "Time per episode",
                                      "Episode", "Time (s)")

        ImageGenerator.generate_graph(self.stats.activation_per_episode,
                                      self.save_path + self.assets_path + "/activation.png",
                                      "Avg activation per episode",
                                      "Episode", "Activation avg")

        ImageGenerator.generate_graph(self.stats.rewards_per_episode,
                                      self.save_path + self.assets_path + "/rewards.png",
                                      "Rewards per episode",
                                      "Episode", "Distance")

        num_per_batch = ImageGenerator.generate_comparison_graph(self.stats.last_dist_per_episode,
                                                                 self.stats.dist_per_episode,
                                                                 self.save_path + self.assets_path + "/distances.png",
                                                                 "Distances per episode",
                                                                 "Episode", "Distance",
                                                                 "Last dist", "Max dist",
                                                                 self.model.max_batches)

        ImageGenerator.generate_pillar_graph(num_per_batch,
                                             self.save_path + self.assets_path + "/distances_bar.png",
                                             "Max reached per batch",
                                             "Batch number", "Times")

        if self.stats.first_episode_data is not None and self.stats.best_episode_data is not None:
            ImageGenerator.generate_episode_comparison_grid(
                self.stats.first_episode_data.activations_per_neuron[100:161],
                self.stats.first_episode_data.rewards_per_step[100:161],
                self.stats.best_episode_data.activations_per_neuron[100:161],
                self.stats.best_episode_data.rewards_per_step[100:161],
                self.save_path + self.assets_path + "/episode_comparison.png",
                "First vs Best Episode (2s)"
            )

    def generate_markdown(self):
        self._generate_assets()

        self.add_h1("Reinforcement Learning Simulation Summary")
        self.add_hr()

        self.add_h2("Simulation Information")
        self.add_li(self.bold("Method :") + self.model.name)
        self.add_li(self.bold("Date Time:") + self.stats.date_time.strftime("%d-%m-%Y %H:%M:%S"))
        self.add_li(self.bold("Device:") + self.stats.device.upper())
        self.add_li(self.bold("Physics Timestamp:") + " 1/60s")
        self.add_li(self.bold("Physics Substeps:") + str(self.settings.substeps))
        self.add_li(self.bold("Total Simulation Time:") + self.stats.get_elapsed_time())
        self.add_li(self.bold("Number of Steps per Episode:") + str(self.stats.steps_per_episode))
        self.add_li(self.bold("Number of Episodes:") + str(self.stats.number_of_episodes))
        self.add_br()
        self.add_hr()

        self.add_h2("Creature Information")
        self.add_li(self.bold("Creature ID:") + str(self.creature.id))
        self.add_li(self.bold("Joints:") + str(len(self.creature.joints)))
        self.add_li(self.bold("Bones:") + str(len(self.creature.bones)))
        self.add_li(self.bold("Muscles:") + str(len(self.creature.muscles)))
        self.add_li(self.bold("Joint Degrees Min:") + str(self.settings.min_joint_angle))
        self.add_li(self.bold("Joint Degrees Max:") + str(self.settings.max_joint_angle))
        self.add_li(self.bold("Scale:") + str(self.settings.scale))
        self.add_br()
        self.add_image(self.assets_path + "creature.png", "Creature image")
        self.add_hr()

        self.add_h2("Method Description")
        self.add_text(self.model.description)
        self.add_br()

        self.add_h2("Network Configuration")
        self.add_li(self.bold("Method:") + self.model.full_name + " (" + self.model.name + ")")
        self.add_li(self.bold("Inputs:") + str(self.model.input_size))
        self.add_li(self.bold("Outputs:") + str(self.model.output_size))

        self.add_h3("Network Architecture")
        if self.model.actor is not None:
            self.add_li(self.bold("Actor"))
            self.add_list_from_dict(self.model.actor.description, nesting=1)
        if self.model.critic is not None:
            self.add_li(self.bold("Critic"))
            self.add_list_from_dict(self.model.critic.description, nesting=1)

        self.add_h3("Hyperparameters")
        self.add_list_from_dict(self.model.hyperparameters)

        self.add_br()
        self.add_hr()

        self.add_h2("Results")
        self.add_h3("Distances")
        self.add_text(
            """From the distances graph we can see when the final distance matches the maximum distance.
             From this graph we can conclude in which episodes the possibility of creature going further was limited by time and not by fitness.""")

        self.add_image(self.assets_path + "distances.png", "Max distance graph")

        self.add_text(
            """This graph show how many times per episode is maximum distance equal to final distance. 
            If number is growing we can consider that model is improving.""")

        self.add_image(self.assets_path + "distances_bar.png", "Max reached per batch")

        self.add_h3("Times per episode")
        self.add_text(
            "Here we can see spikes in time when parameter update is being called and also times when the episode is terminated prematurely."
            "(Letting the simulation run visually will be visible because of the longer time)")
        self.add_image(self.assets_path + "time.png", "Last distance graph")

        self.add_h3("Activation per episode")
        self.add_text("Number increases as the muscle is activated more strongly. Per episode average is displayed.")
        self.add_image(self.assets_path + "activation.png", "Last distance graph")
        self.add_text("")

        self.add_h3("Rewards per episode")
        self.add_text("Main goal of any method maximize rewards. Per episode average is displayed.")
        self.add_image(self.assets_path + "rewards.png", "Last distance graph")

        self.add_br()
        self.add_hr()

        if self.stats.first_episode_data is not None and self.stats.best_episode_data is not None:
            self.add_h3("Best vs First Episode")

            first = self.stats.first_episode_data
            best = self.stats.best_episode_data
            columns = ["", self.bold("First Episode"), self.bold("Best Episode")]
            rows = [
                [self.bold("Episode Index"), str(first.index), str(best.index)],
                [self.bold("Max Distance"), f2d(first.max_dist) + "m", f2d(best.max_dist) + "m"],
                [self.bold("Last Distance"), f2d(first.last_dist) + "m", f2d(best.last_dist) + "m"],
                [self.bold("Average Activation"),
                 f2d(np.average(np.sum(np.abs(first.activations_per_neuron), axis=1))),
                 f2d(np.average(np.sum(np.abs(best.activations_per_neuron), axis=1)))],
                [self.bold("Average Rewards"),
                 f2d(np.average(first.rewards_per_step)),
                 f2d(np.average(best.rewards_per_step))],
                [self.bold("Time"),
                 self.get_time_from_s(first.time),
                 self.get_time_from_s(best.time),
                 ]
            ]
            self.add_table(columns, rows)

            self.add_h3("Graph Comparison")
            self.add_text("Activation per neuron and rewards per step.")
            self.add_image(self.assets_path + "/episode_comparison.png")

            self.add_br()
            self.add_hr()

        self.add_h2("Notes")
        self.add_li("This report was generated automatically after simulation completion.")
        self.add_li("All conclusions should be made visually by the reader.")
        self.add_li("For more information go to the " + self.link("github repository",
                                                                "https://github.com/ajromen/Reinforcment-Learning-Simulation") + ".")


    def save_markdown(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(self.text)

    def get_time_from_s(self, time: float):
        min = (int(time) % 3600) // 60
        sec = int(time) % 60
        if min == 0 and sec == 0:
            return "<1s"
        return f"{min:02d}m:{sec:02d}s"

    # headings

    def add_h1(self, text: str):
        self.text += f"# {text}\n\n"

    def add_h2(self, text: str):
        self.text += f"## {text}\n\n"

    def add_h3(self, text: str):
        self.text += f"### {text}\n\n"

    # text

    def add_text(self, text: str):
        self.text += f"{text}\n\n"

    def bold(self, text):
        return f"**{text}** "

    def link(self, name, address):
        return f"[{name}]({address})"

    # breaks

    def add_br(self):
        self.text += "  \n"

    def add_hr(self):
        self.text += "\n---\n\n"

    # images

    def add_image(self, path: str, alt: str = ""):
        self.text += f"![{alt}]({path})\n\n"

    def add_list_from_dict(self, dict, nesting=0):
        for k, v in dict.items():
            self.text += "\t" * nesting
            self.add_li(k + ": " + self.inline_code(str(v)))
        self.text += "\n"

    def add_li(self, text):
        self.text += f"- {text}\n"

    # code

    def inline_code(self, code: str):
        return f"`{code}`"

    # table

    def add_table(self, headers, rows):
        self.text += "| " + " | ".join(headers) + " |\n"
        self.text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            self.text += "| " + " | ".join(map(str, row)) + " |\n"
        self.text += "\n"


def f2d(text):
    return f"{text:.2f}"
