from pathlib import Path

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



    def generate_markdown(self):
        self._generate_assets()
        self.add_h4(APP_NAME)
        self.add_h1(self.model.name + " method")
        self.add_image(self.assets_path + "creature.png", "Creature image")
        self.add_image(self.assets_path + "max_dist.png", "Max distance graph")
        self.add_image(self.assets_path + "last_dist.png", "Last distance graph")

    def save_markdown(self, save_path: str):
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(self.text)

    # headings

    def add_h1(self, text: str):
        self.text += f"# {text}\n\n"

    def add_h2(self, text: str):
        self.text += f"## {text}\n\n"

    def add_h3(self, text: str):
        self.text += f"### {text}\n\n"

    def add_h4(self, text: str):
        self.text += f"#### {text}\n\n"

    # text

    def add_text(self, text: str):
        self.text += f"{text}\n\n"

    def add_bold(self, text: str):
        self.text += f"**{text}**\n\n"

    def add_italic(self, text: str):
        self.text += f"*{text}*\n\n"

    # breaks

    def add_br(self):
        self.text += "  \n"

    def add_hr(self):
        self.text += "\n---\n\n"

    # images

    def add_image(self, path: str, alt: str = ""):
        self.text += f"![{alt}]({path})\n\n"

    def add_list(self, items):
        for item in items:
            self.text += f"- {item}\n"
        self.text += "\n"

    def add_numbered_list(self, items):
        for i, item in enumerate(items, start=1):
            self.text += f"{i}. {item}\n"
        self.text += "\n"

    # code

    def add_code_block(self, code: str, lang: str = ""):
        self.text += f"```{lang}\n{code}\n```\n\n"

    def add_inline_code(self, code: str):
        self.text += f"`{code}`\n"

    # table

    def add_table(self, headers, rows):
        self.text += "| " + " | ".join(headers) + " |\n"
        self.text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            self.text += "| " + " | ".join(map(str, row)) + " |\n"
        self.text += "\n"
