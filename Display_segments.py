# imports
from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
import torch

SEG_WIDTH = 50
LENGTH = 500
FULL_WIDTH = 500
N_SEGMENTS = int(FULL_WIDTH / SEG_WIDTH)
CONFIG = [[0,0], [0,1], [28,37], [28,38], [38,37], [28,39], [0,2]]
CELL_SIZE = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
EDGES = (255, 0, 255)


def display(surface, segments):
    surface.fill(BLACK)
    for segment in segments:
        for y in range(LENGTH):
            for x in range(SEG_WIDTH):
                #print(f"y: {y}, x: {x}, x % FULL_WIDTH: {x % FULL_WIDTH}, matrix shape: {segment.matrix.shape}")

                color = WHITE if segment.matrix[ x % FULL_WIDTH][y] == 1 else BLACK
                #if segment.matrix[x % FULL_WIDTH][y] == 1 and (x == SEG_WIDTH - 1 or x == 0):
                #if x == 0 or x == SEG_WIDTH - 1 or y == 0 or y == LENGTH - 1:
                    #color = EDGES
                pygame.draw.rect(surface, color, ((x + segment.startpos) * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))