# imports
from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
import torch
import random
from mpi4py import MPI
import os
import imageio

SEG_WIDTH = 50
LENGTH = 500
FULL_WIDTH = 500
N_SEGMENTS = int(FULL_WIDTH / SEG_WIDTH)
CONFIG = [[0,0], [0,1], [28,37], [28,38], [38,37], [28,39], [0,2]]
CELL_SIZE = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
EDGES = (255, 0, 255)
SAVE_PATH = "C:/Users/fatim/OneDrive/Desktop/COSC 608/project/imgs"

def create_config():

    config = np.zeros((SEG_WIDTH, LENGTH), dtype=int)
    glider = [[0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 0]]
    x = random.randint(1, 450)
    config[43:49, x:x+6] = glider

    return config

def update_grid_with_boundaries(local_grid, left_boundary_data, right_boundary_data):
    # Inner update code
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
                    sum += local_grid[neighbours[n][0]][neighbours[n][1]]

                # Main game of life code (am I dead or alive?)
                if local_grid[i][j] == 1:
                    if sum >= 2 and sum <= 3:
                        temp[i][j] = 1

                else:
                    if sum == 3:
                        temp[i][j] = 1
    # Outer update code

        # The adjacent column on the segment to my left
        left_border = left_boundary_data[-1, :]

        # The adjacent column on the segment to my right
        right_border = right_boundary_data[0, :]

        # Blank columns for my updated left/right edges
        my_left = np.zeros((LENGTH))
        my_right = np.zeros((LENGTH))
        # going through my left border
        for i in range (LENGTH):
             # Establish whether each neighbour is alive or dead
                sum = 0
                sum += local_grid[1, i]
                sum += left_border[i]

                sum += left_border[(i - 1) % LENGTH]
                sum += local_grid[0, (i - 1) % LENGTH]
                sum += local_grid[1, (i - 1) % LENGTH]

                sum += left_border[(i + 1) % LENGTH]
                sum += local_grid[0, (i + 1) % LENGTH]
                sum += local_grid[1, (i + 1) % LENGTH]
                # Main game of life code (am I dead or alive?)
                if local_grid[0, i] == 1:
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
                sum += local_grid[SEG_WIDTH - 2, i]
                sum += right_border[i]

                sum += right_border[(i - 1) % LENGTH]
                sum += local_grid[SEG_WIDTH - 1, (i - 1) % LENGTH]
                sum += local_grid[SEG_WIDTH - 2, (i - 1) % LENGTH]

                sum += right_border[(i + 1) % LENGTH]
                sum += local_grid[SEG_WIDTH - 1, (i + 1) % LENGTH]
                sum += local_grid[SEG_WIDTH - 2, (i + 1) % LENGTH]

                # Main game of life code (am I dead or alive?)
                if local_grid[SEG_WIDTH - 1, i] == 1:
                    if sum >= 2 and sum <= 3:
                        my_right[i] = 1
                    else:
                        my_right[i] = 0

                else:
                    if sum == 3:
                        my_right[i] = 1
                    else:
                        my_right[i] = 0


    local_grid = temp
    local_grid[0, :] = my_left
    local_grid[-1, :] = my_right

def exchange_boundary_data(local_grid, left_neighbor, right_neighbor, comm):

    comm.send(local_grid[0, :], dest=left_neighbor)
    comm.send(local_grid[-1, :], dest=right_neighbor)
    
    left_boundary_data = comm.recv(source=left_neighbor)
    right_boundary_data = comm.recv(source=right_neighbor)

    return left_boundary_data, right_boundary_data

def gather_matrices(local_grid, comm, iteration):

    # Gather matrices from all processes to the root process
    gathered_matrices = comm.gather(local_grid, root=0)
    
    if comm.rank == 0:
        # The root process assembles the final image from gathered matrices
        final_image = np.vstack(gathered_matrices)
        path_name = f'iteration_{iteration}.png'
        # Save the final image as a PNG file
        plt.imshow(final_image, cmap='binary')
        plt.title(f'Game of Life - Iteration {iteration}')
        plt.savefig(os.path.join(SAVE_PATH, path_name))
        plt.close()

def create_gif(images, gif_filename):
    with imageio.get_writer(gif_filename, mode='I') as writer:
        for image in images:
            writer.append_data(image)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    left_neighbor = (rank - 1 + size) % size
    right_neighbor = (rank + 1) % size

    local_grid = create_config()  

    num_iterations = 100

    # Main simulation loop
    for iteration in range(num_iterations):

        left_boundary, right_boundary = exchange_boundary_data(local_grid, left_neighbor, right_neighbor)

        comm.Barrier()

        update_grid_with_boundaries(local_grid, left_boundary, right_boundary)

        comm.Barrier()

    # Create GIF
    image_list = os.listdir(SAVE_PATH)
    gif_filename = f'{SAVE_PATH}/game_of_life.gif'
    create_gif(image_list, gif_filename)

    for iteration in range(num_iterations):
        os.remove(f'{SAVE_PATH}/iteration_{iteration}.png')

if __name__ == "__main__":
    main()


