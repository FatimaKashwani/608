# imports
from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame

SEG_WIDTH = 500
LENGTH = 500
FULL_WIDTH = 500
N_SEGMENTS = int(FULL_WIDTH / SEG_WIDTH)
CONFIG = [[0,0], [0,1], [28,37], [28,38], [38,37], [28,39], [0,2]]
CELL_SIZE = 10

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (255, 0, 0)

def create_config(type):
    # Initialize an empty grid
    grid_size = FULL_WIDTH
    config = np.zeros((grid_size, grid_size), dtype=int)

    if type == 1:
            # Add a glider
            config[2:5, 1] = 1
            config[1, 3] = 1
            config[2:4, 4] = 1

            # Add a pulsar
            pulsar = np.array([
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            ])

            config[15:26, 15:26] = pulsar
    elif type == 2:
            heart = np.array([
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ])

            config[15:26, 15:26] = heart
    return config


def display(surface, segments):
    #mask = segments[0].matrix

    # here we iterate over each segment to fill in the mask
    for segment in segments:
        startpos = SEG_WIDTH * segment.position

    surface.fill(BLACK)
    for y in range(LENGTH):
        for x in range(FULL_WIDTH):

            color = WHITE if segments[0].matrix[y, x] == 1 else BLACK
            if segments[0].matrix[y, x] == 2: color = GREEN

            pygame.draw.rect(surface, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))


# Overall matrix is divided into vertical segments
class segment():
    def __init__(self, position):
        self.position = position
        self.matrix = np.zeros((FULL_WIDTH, LENGTH))

    def setMatrix(self, matrix):
        self.matrix = matrix

    # Updating the matrices; change this
    def calculate_segment(self):
        temp = np.zeros((FULL_WIDTH, LENGTH))
        # Internal cells only; skipping the first and last rows & cols
        for i in range(1, SEG_WIDTH - 2):
            for j in range(1, LENGTH - 2):

                # Establish whether each neighbour is alive or dead
                neighbours = [[i-1,j-1], [i-1,j], [i-1, j+1], [i,j-1], [i,j+1], [i+1, j-1], [i+1, j], [i+1, j+1]]
                sum = 0
                for n in range(len(neighbours)):
                    sum += self.matrix[neighbours[n][0]][neighbours[n][1]]

                # Main game of life code (am I dead or alive?)
                if self.matrix[i][j] == 1:
                    if sum >= 2 and sum <= 3:
                        temp[i][j] = 1

                else:
                    if sum == 3:
                        temp[i][j] = 1

        self.matrix = temp
                    

                
    def calculate_edges(self, left, right):
        for i in range (SEG_WIDTH):
            self.matrix[i][0] = 2
            self.matrix[i][LENGTH - 1] = 2
        for i in range (LENGTH):
            self.matrix[0][i] = 2
            self.matrix[0][SEG_WIDTH - 1] = 2


if __name__ == "__main__":

    start_config = CONFIG
    # start_config = np.zeros((FULL_WIDTH, LENGTH))
    # add alterations to the start config
    pygame.init()
    screen = pygame.display.set_mode((FULL_WIDTH, LENGTH))
    pygame.display.set_caption("Conway's Game of Life")

    clock = pygame.time.Clock()
    #grid = []
    #grid.append(segment(0))
    #grid[0].setMatrix(create_config)
    segments = [segment(x) for x in range(N_SEGMENTS)]
    num_iterations = 20
    segments[0].setMatrix(create_config(2))
    running = True
    i = 0
    while running:
        i += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        display(screen, segments)
        #for i in range(num_iterations):
        #print("iteration")

        new_segments = segments.copy()
        for j in range(len(new_segments)):
                if j == 0:
                    left = len(new_segments) - 1
                else:
                    left = j - 1
                if j == len(new_segments) - 1:
                    right = 0
                else:
                    right = j + 1
                new_segments[j].calculate_segment()
                new_segments[j].calculate_edges(segments[left], segments[right])
        # Change this to account for multiple segments later on
        if i == num_iterations:
            running = False
        segments = new_segments
        pygame.display.flip()
        clock.tick(1)

    pygame.quit()

            

