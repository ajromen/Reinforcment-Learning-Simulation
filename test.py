import pygame

from src.models.creature import Creature
from src.scenes.analysis_scene import AnalysisScene
from src.simulation.simulation_window import SimulationWindow
from src.utils.creature_loader import CreatureLoader

creature = CreatureLoader.load("./data/a7403382-371c-43d3-85cd-16d062b21813/creature.json")
# ansc = AnalysisScene(creature,[1,10,1])
# ansc.start()

pygame.init()
win = SimulationWindow(creature,None)
win.start()