from src.agents.agent import Agent
from src.models.creature import Creature

from src.simulation.simulation_settings import SimulationSettings
from src.simulation.simulation_stats import SimulationStats
from src.ui.ui_settings import APP_NAME


class MarkdownMaker:
    def __init__(self,
                 creature: Creature,
                 save_path: str,
                 model: Agent,
                 settings: SimulationSettings,
                 stats: SimulationStats):
        self.text = ""
        self.creature = creature
        self.model = model
        self.save_path = save_path
        self.settings = settings
        self.stats = stats

    def generate_markdown(self):
        self.add_h2(APP_NAME)
        self.add_h1(self.model.name+" method")
        self.add_image(self.save_path+"creature.png", "Creature image")

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

    def save(self):
        with open(self.save_path + self.model.name + ".md", "w", encoding="utf-8") as f:
            f.write(self.text)
