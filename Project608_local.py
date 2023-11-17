# imports
from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame

SEG_WIDTH = 50
LENGTH = 500
FULL_WIDTH = 500
N_SEGMENTS = int(FULL_WIDTH / SEG_WIDTH)
CONFIG = [[0,0], [0,1], [28,37], [28,38], [38,37], [28,39], [0,2]]
CELL_SIZE = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
EDGES = (255, 0, 255)

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
            pulsar = np.rot90(pulsar)
            config[15:26, 15:26] = pulsar
            glider = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
            ])
            config[45:48, 30:33] = glider
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
            heart = np.rot90(heart)
            config[15:26, 15:26] = heart
            config[35:46, 35:46] = heart
            config[15:26, 55:66] = heart
            config[35:46, 75:86] = heart
            config[15:26, 95:106] = heart
            config[35:46, 115:126] = heart
            config[15:26, 135:146] = heart
            config[35:46, 155:166] = heart
            config[15:26, 175:186] = heart
            config[35:46, 195:206] = heart
            config[15:26, 215:226] = heart
            config[35:46, 235:246] = heart
            config[35:46, 255:266] = heart
            config[15:26, 275:286] = heart
            config[35:46, 295:306] = heart
            config[35:46, 315:326] = heart
            config[15:26, 335:346] = heart
            config[35:46, 355:366] = heart
            config[15:26, 375:386] = heart
            config[35:46, 395:406] = heart
            config[15:26, 415:426] = heart
            config[35:46, 435:446] = heart
            config[35:46, 455:466] = heart
            config[15:26, 475:486] = heart
    elif type == 3:
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
        config2 = np.zeros((grid_size, grid_size), dtype=int)
        left_half = heart[5:, :]
        right_half = heart[:5, :]
        config[230:236, 15:26] = heart[:6, :]
        config2[0:6, 15:26] = heart[5:, :]
        return config, config2
            
            
    return config


def display(surface, segments):
    surface.fill(BLACK)
    for segment in segments:
        for y in range(LENGTH):
            for x in range(SEG_WIDTH):
                #print(f"y: {y}, x: {x}, x % FULL_WIDTH: {x % FULL_WIDTH}, matrix shape: {segment.matrix.shape}")

                color = WHITE if segment.matrix[ x % FULL_WIDTH][y] == 1 else BLACK
                if segment.matrix[x % FULL_WIDTH][y] == 2:
                    color = EDGES
                pygame.draw.rect(surface, color, ((x + segment.startpos) * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))



# Overall matrix is divided into vertical segments
class segment():
    def __init__(self, position):
        self.position = position
        self.matrix = np.zeros((FULL_WIDTH, LENGTH))
        self.startpos = SEG_WIDTH * self.position

    def setMatrix(self, matrix):
        self.matrix = matrix


    def calculate_segment(self):
        temp = np.zeros((SEG_WIDTH, LENGTH))
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
        prev = self.matrix
        self.matrix = temp
        return prev 

                
    def calculate_edges(prev, self, left, right):
        prev = prev.matrix
        # the segment to my left
        left_border = [row[-1] for row in left.matrix]
        # my left border
        my_left = prev[:,0]
        # the segment to my right
        right_border = [row[0] for row in right.matrix]
        # my right border
        my_right = prev[:,-1]

        # going through my left border
        for i in range (LENGTH):
             # Establish whether each neighbour is alive or dead
                neighbours = []
                neighbours += [prev[i,1], left_border[i]]
                if i != 0:
                    neighbours += [left_border[i-1], prev[i-1, 0], prev[i-1, 1]]
                if i != LENGTH - 1:
                    neighbours += [left_border[i+1], prev[i+1, 0], prev[i+1, 1]]
                sum = np.sum(neighbours)

                # Main game of life code (am I dead or alive?)
                if prev[i][0] == 1:
                    if sum >= 2 and sum <= 3:
                        my_left[i] = 1

                else:
                    if sum == 3:
                        my_left[i] = 1


        # going through my right border
        for i in range (LENGTH):
             # Establish whether each neighbour is alive or dead
                neighbours = []
                neighbours += [prev[i,LENGTH - 2], right_border[i]]
                if i != 0:
                    neighbours += [right_border[i-1], prev[i-1, LENGTH - 1], prev[i-1, LENGTH - 2]]
                if i != LENGTH - 1:
                    neighbours += [right_border[i+1], prev[i+1, LENGTH - 1], prev[i+1, LENGTH - 2]]
                sum = sum(neighbours)

                # Main game of life code (am I dead or alive?)
                if prev[i][0] == 1:
                    if sum >= 2 and sum <= 3:
                        my_right[i] = 1

                else:
                    if sum == 3:
                        my_right[i] = 1

        self.matrix[:, 0] = my_left
        self.matrix[:, -1] = my_right

        '''
        for i in range (SEG_WIDTH):
            self.matrix[i][0] = 2
            self.matrix[i][LENGTH - 1] = 2
        for i in range (LENGTH):
            self.matrix[0][i] = 2
            self.matrix[SEG_WIDTH - 1][i] = 2
        '''

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
    #for x in range(len(segments)):
            #segments[x].setMatrix(create_config(2))
    #segments[8].setMatrix(create_config(1))
    left, right = create_config(3)
    segments[0].setMatrix(left)
    segments[1].setMatrix(right)
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
                prev = new_segments[j].calculate_segment()
                new_segments[j].calculate_edges(prev, segments[left], segments[right])
                print("here?")
        if i == num_iterations:
            running = False

        segments = new_segments
        pygame.display.flip()
        clock.tick(1)

    pygame.quit()

            

