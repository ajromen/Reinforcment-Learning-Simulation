import pygame

from src.agents.reinforce_agent import ReinforceAgent
from src.models.creature import Creature
from src.pymunk.creature_pymunk import CreaturePymunk
from src.scenes.analysis_scene import AnalysisScene
from src.simulation.simulation_window import SimulationWindow
from src.utils.creature_loader import CreatureLoader

creature = CreatureLoader.load("./data/058ba9a2-c39f-45ea-82ee-47b2d84e4987/creature.json")
layer_widths = [CreaturePymunk.get_number_of_inputs(creature),30,30,30,len(creature.muscles)]
# ansc = AnalysisScene(creature,layer_widths)
# ansc.start()

pygame.init()
agent = ReinforceAgent(layer_widths)
win = SimulationWindow(creature,agent)
win.start()