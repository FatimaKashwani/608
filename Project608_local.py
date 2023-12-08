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

def create_config(type):
    # Initialize an empty grid
    #grid_size = FULL_WIDTH
    config = np.zeros((SEG_WIDTH, LENGTH), dtype=int)

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
        right_half = heart[:6, :]
        config[0:6, 15:26] = left_half
        config2[44:50, 15:26] = right_half

        config[26:32, 85:96] = left_half
        config[20:26, 85:96] = right_half
        config2[48:49, 39:49] = np.transpose(np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]))
        glider = [
            [0, 0, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
            ]
        glider2 = [[0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0]]
        glider_h = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        #config2[33:39, 215:221] = glider
        config2[43:49, 215:225] = glider2

        return config, config2
    



    elif type == 4:
        glider = [
            [0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1]
            ]
        glider2 = [[0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0]]
        
        glider3 = [[0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0]]
        config[3:9, 490:496] = np.rot90(glider2)
            
    elif type == 5:
        glider_gun = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        ]
        #config[100:143, 30:39] = np.rot90(glider_gun)
        config[30:39, 100:143] = glider_gun

    return config


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



# Overall matrix is divided into vertical segments
class segment():
    def __init__(self, position):
        self.position = position
        self.matrix = np.zeros((SEG_WIDTH, LENGTH))
        self.startpos = SEG_WIDTH * self.position

    def setMatrix(self, matrix):
        self.matrix = matrix


    def calculate_segment(self):
        temp = np.zeros((SEG_WIDTH, LENGTH))
        # Internal cells only; skipping the first and last rows & cols
        for i in range(1, SEG_WIDTH - 1):
            for j in range(LENGTH):

                # Establish whether each neighbour is alive or dead
                if j == 0:
                    neighbours = [[i-1,LENGTH - 1], [i-1,j], [i-1, j+1], [i,LENGTH - 1], [i,j+1], [i+1, LENGTH - 1], [i+1, j], [i+1, j+1]]
                elif j == LENGTH - 1:
                    neighbours = [[i-1, j-1], [i-1,j], [i-1, 0], [i, j-1], [i,0], [i+1, j-1], [i+1, j], [i+1, 0]]
                else:
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


                
    def calculate_edges(self, prev, left, right):
        # the segment to my left
        #prev = prev.transpose()
        #print(prev.shape)
        left_border = left[-1, :]

        # my left border
        #my_left = prev[0,:]
        my_left = np.zeros((LENGTH))
        # the segment to my right
        right_border = right[0, :]
        # my right border
        #my_right = prev[SEG_WIDTH -1,:]
        my_right = np.zeros((LENGTH))
        # going through my left border
        for i in range (LENGTH):
             # Establish whether each neighbour is alive or dead
                '''
                neighbours = []
                neighbours += [prev[1, i], left_border[i]]
                if i != 0:
                    neighbours += [left_border[i-1], prev[0, i-1], prev[1, i-1]]
                if i != LENGTH - 1:
                    neighbours += [left_border[i+1], prev[0, i+1], prev[1, i+1]]
                sum = np.sum(neighbours)
                '''
                sum = 0
                sum += prev[1, i]
                sum += left_border[i]
                if i != 0:
                    sum += left_border[i-1]
                    sum += prev[0, i-1]
                    sum += prev[1, i-1]
                else:
                    sum += left_border[LENGTH-1]
                    sum += prev[0, LENGTH-1]
                    sum += prev[1, LENGTH-1]
                if i != (LENGTH - 1):
                    sum += left_border[i+1]
                    sum += prev[0, i+1]
                    sum += prev[1, i+1]
                else:
                    sum += left_border[0]
                    sum += prev[0, 0]
                    sum += prev[1, 0]
                # Main game of life code (am I dead or alive?)
                if prev[0, i] == 1:
                    if sum >= 2 and sum <= 3:
                        my_left[i] = 1
                    else:
                        my_left[i] = 0

                else:
                    if sum == 3:
                        my_left[i] = 1
                    else:
                        my_left[i] = 0


        # going through my right border

        for i in range (LENGTH):
             # Establish whether each neighbour is alive or dead
                sum = 0
                sum += prev[SEG_WIDTH - 2, i]
                sum += right_border[i]
                if i != 0:
                    sum += right_border[i-1]
                    sum += prev[SEG_WIDTH - 1, i-1]
                    sum += prev[SEG_WIDTH - 2, i-1]
                else:
                    sum += right_border[LENGTH-1]
                    sum += prev[SEG_WIDTH - 1, LENGTH-1]
                    sum += prev[SEG_WIDTH - 2, LENGTH-1]
                if i != LENGTH - 1:
                    sum += right_border[i+1]
                    sum += prev[SEG_WIDTH - 1, i+1]
                    sum += prev[SEG_WIDTH - 2, i+1]
                else:
                    sum += right_border[0]
                    sum += prev[SEG_WIDTH - 1, 0]
                    sum += prev[SEG_WIDTH - 2, 0]

                # Main game of life code (am I dead or alive?)
                if prev[SEG_WIDTH - 1, i] == 1:
                    if sum >= 2 and sum <= 3:
                        my_right[i] = 1
                    else:
                        my_right[i] = 0

                else:
                    if sum == 3:
                        my_right[i] = 1
                    else:
                        my_right[i] = 0
        #if np.sum(my_right) != 0:
            #print(np.sum(my_right))
        self.matrix[0] = my_left
        self.matrix[-1] = my_right

        '''
        for i in range (SEG_WIDTH):
            self.matrix[i][0] = 2
            self.matrix[i][LENGTH - 1] = 2
        for i in range (LENGTH):
            self.matrix[0][i] = 2
            self.matrix[SEG_WIDTH - 1][i] = 2
        '''
    

if __name__ == "__main__":
    if torch.cuda.is_available(): torch.cuda.set_device(0)
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
    #for seg in segments:
        #print (seg.matrix.shape)
    num_iterations = 1000
    for x in range(len(segments)):
            #if x%2 == 0:
                segments[x].setMatrix(create_config(4))
    #segments[8].setMatrix(create_config(1))
    #left, right = create_config(3)
    #segments[1].setMatrix(left)
    #segments[0].setMatrix(right)
    #segments[3].setMatrix(create_config(4))
    #segments[4].setMatrix(create_config(5))
    running = True
    i = 0
    while running:
        if i%100 == 0: print (i)
        i += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        display(screen, segments)
        #for i in range(num_iterations):
        #print("iteration")
        previouses = [s.matrix for s in segments]
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

                
                new_segments[j].calculate_edges(previouses[j], previouses[left], previouses[right])
        if i == num_iterations:
            running = False

        segments = new_segments
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

            

