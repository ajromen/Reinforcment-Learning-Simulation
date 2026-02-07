import pygame

from src.agents.ppo_agent import PPOAgent
from src.agents.reinforce_agent import ReinforceAgent
from src.markdown.image_generator import ImageGenerator
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.scenes.analysis_scene import AnalysisScene
from src.simulation.simulation_window import SimulationWindow
from src.utils.creature_loader import CreatureLoader

creature = CreatureLoader.load("./data/f8b0b980-4078-4b8b-a8d1-0aa225940344/creature.json")
layer_widths = [CreaturePymunk.get_number_of_inputs(creature), 30, 30, 30, len(creature.muscles)]
# ansc = AnalysisScene(creature,layer_widths)
# ansc.start()

# pygame.init()
# agent = ReinforceAgent(layer_widths)
agent = PPOAgent(layer_widths)
win = SimulationWindow(creature, agent, "./data/f8b0b980-4078-4b8b-a8d1-0aa225940344/"+agent.name.lower()+"/", load_old=True)
win.start()