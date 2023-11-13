# imports
from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SEG_WIDTH = 100
LENGTH = 100
FULL_WIDTH = 100
N_SEGMENTS = int(FULL_WIDTH / SEG_WIDTH)
CONFIG = [[0,0], [0,1], [28,37], [28,39], [38,37]]

def display(segments, fig):
    # mask = np.zeros((FULL_WIDTH, LENGTH))
    # Simplified for one segment. Update this to iterate over segments
    mask = segments[0].matrix

    # here we iterate over each segment to fill in the mask
    for segment in segments:
        startpos = SEG_WIDTH * segment.position

    ys, xs = np.where(mask.astype(bool))
    sc = plt.scatter(xs[::2], ys[::2])

    def update(mask):
        # Update the mask for the next iteration (you need to implement this part)
        # You should update the mask for each segment in your actual implementation
        # For now, I'm just using the original mask for demonstration purposes.
        new_mask = mask  # Replace this with the logic to update the mask

        # Update the scatter plot data
        ys, xs = np.where(new_mask.astype(bool))
        sc.set_offsets(np.column_stack((xs[::2], ys[::2])))

        return sc,

    ani = FuncAnimation(fig, update, frames=range(10), interval=200, blit=True)
    plt.show()

def initialize(segments, start_config):
    if np.sum(start_config) == 0:
        return
    for pixel in start_config:
        # Simplified for one segment. Update this to find which segment each pixel belongs to
        segments[0].matrix[pixel[0]][pixel[1]] = 1

        # segment.matrix = start_config[:][:]

# Overall matrix is divided into vertical segments
class segment():
    def __init__(self, position):
        self.position = position
        self.matrix = np.zeros((FULL_WIDTH, LENGTH))

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
        pass


if __name__ == "__main__":
    fig, _ = plt.subplots()
    #start_config = np.zeros((FULL_WIDTH, LENGTH))
    # add alterations to the start config
    start_config = CONFIG
    segments = [segment(i) for i in range(N_SEGMENTS)]
    num_iterations = 10
    initialize(segments, start_config)
    for i in range(num_iterations):
        display(segments, fig)
        new_segments = segments.copy()
        for j in range(len(new_segments)):
            if j == 0: left = len(new_segments) - 1
            else: left = j - 1
            if j == len(new_segments) - 1: right = 0
            else: right = j + 1
            new_segments[j].calculate_segment()
            new_segments[j].calculate_edges(segments[left],segments[right])

            

