import pygame

from src.agents.ppo_agent import PPOAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.markdown.image_generator import ImageGenerator
from src.markdown.markdown_maker import MarkdownMaker
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.scenes.analysis_scene import AnalysisScene
from src.simulation import simulation_settings
from src.simulation.simulation_settings import SimulationSettings
from src.simulation.simulation_stats import SimulationStats
from src.simulation.simulation_window import SimulationWindow
from src.utils.creature_loader import CreatureLoader

save_path = "./data/f8b0b980-4078-4b8b-a8d1-0aa225940344/"

creature = CreatureLoader.load(save_path + "creature.json")
layer_widths = [CreaturePymunk.get_number_of_inputs(creature), 30, 30, 30, len(creature.muscles)]
# ansc = AnalysisScene(creature, layer_widths, False)
# ansc.start()

# pygame.init()
# agent = ReinforceAgent(layer_widths)
agent = PPOAgent(layer_widths)
simulation_path = save_path + agent.name.lower() + "/"
# win = SimulationWindow(creature, agent, simulation_path, load_old=True)
# win.start()

# md maker
stats = SimulationStats(540,"cuda")
settings = SimulationSettings()
agent.load_from_file(simulation_path + "model.pt")
stats.load_from_file(simulation_path + "stats.json")
settings.load_from_file(simulation_path + "settings.json")

md = MarkdownMaker(creature, simulation_path, "assets/", agent, settings, stats)
md.generate_markdown()
md.save_markdown(simulation_path+"summary.md")
