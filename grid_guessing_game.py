''' Adapted from the code found on https://brilliant.org/wiki/backpropagation/
    to create a game to demostrate the effect of machine learning as a result
    of gradient descent and backpropagation.
    Created by Jane Sieving, December 2018
'''

import numpy as np
import pygame
from random import choice
from time import sleep
import network
import os

RED = (200, 50, 50)
BLUE = (0, 100, 200)

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

# define the sigmoid function
def sigmoid(x, derivative=False):

    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))

class NN():
    def __init__(self, in_len, num_hidden, out_len, alpha = .1):
        self.in_len = in_len
        self.num_hidden = num_hidden
        self.out_len = out_len
        self.alpha = alpha # learning rate
        # initialize weights randomly with mean 0 and range [-1, 1]
        # the +1 in the 1st dimension of the weight matrices is for the bias weight
        self.hidden_weights = 2*np.random.random((in_len + 1, num_hidden)) - 1
        self.output_weights = 2*np.random.random((num_hidden + 1, out_len)) - 1

    def train(self):
        # inputs
        X = np.array(self.input_data)
        # outputs
        # x.T is the transpose of x, making this a column vector
        y = np.array(self.output_data).T

        # number of iterations of gradient descent
        num_iterations = 1000

        # for each iteration of gradient descent
        for i in range(num_iterations):

            # forward phase
            # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
            input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
            hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, self.hidden_weights))))
            output_layer_outputs = np.dot(hidden_layer_outputs, self.output_weights)

            # backward phase
            # output layer error term
            output_error = output_layer_outputs - y
            # hidden layer error term
            # [:, 1:] removes the bias term from the backpropagation
            hidden_error = hidden_layer_outputs[:, 1:] * (1 - hidden_layer_outputs[:, 1:]) * np.dot(output_error, self.output_weights.T[:, 1:])

            # partial derivatives
            hidden_pd = input_layer_outputs[:, :, np.newaxis] * hidden_error[: , np.newaxis, :]
            output_pd = hidden_layer_outputs[:, :, np.newaxis] * output_error[:, np.newaxis, :]

            # average for total gradients
            total_hidden_gradient = np.average(hidden_pd, axis=0)
            total_output_gradient = np.average(output_pd, axis=0)

            # update weights
            self.hidden_weights += - self.alpha * total_hidden_gradient
            self.output_weights += - self.alpha * total_output_gradient

    def test(self, test_arr):
        self.train() #train based on already known data

        # inputs
        X = np.array([test_arr])

        # forward phase
        # np.hstack((np.ones(...), X) adds a fixed input of 1 for the bias weight
        input_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), X))
        hidden_layer_outputs = np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(input_layer_outputs, self.hidden_weights))))
        output_layer_outputs = np.dot(hidden_layer_outputs, self.output_weights)

        # print the final outputs of the neural network on the inputs X
        cls()
        print("Output After Training: {}\n".format(output_layer_outputs[0]))

        correct = input("How should this have been classified? (0/1) ")
        if correct and (correct in "0123456789"):
            correct = int(correct)
            self.input_data.append(test_arr)
            self.output_data[0].append(correct)
            print("I'll remember that.")
        else:
            print("Cool, you clearly don't care very much about what we're doing here.")
        sleep(1)

class GridWorld():
    def __init__(self, nn, width=3, height=3, cell_size=100):
        pygame.init()
        screen_size = (height * cell_size, width * cell_size)
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption = ('Grid')
        self.nn = nn
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.font = pygame.font.get_default_font()
        self._init_cells()
        self._redraw()

    def _draw_background(self):
        self.screen.fill((220, 220, 220))

    def _init_cells(self):
        self.cells = {}
        cell_size = (self.cell_size-8, self.cell_size-8)
        for i in range(self.height):
            for j in range(self.width):
                cell_coord = (i * self.cell_size+4, j * self.cell_size+4)
                self.cells[(i, j)] = Cell(self.screen, cell_coord, cell_size)

    def _draw_cells(self, data = None):
        all_cells = self.cells.values()
        if data:
            for cell, color in zip(all_cells, data):
                if color == 0:
                    cell.color = (0, 100, 200)
                else:
                    cell.color = (200, 50, 50)
                cell.draw()
        else:
            for cell in all_cells:
                cell.draw()

    def _redraw(self):
        self._draw_background()
        self._draw_cells()
        pygame.display.update()

    def paint_cell(self, mouse_pos):
        """Adds a lava tile in the cell indicated by mouse_pos."""
        cell_coord = (mouse_pos[0] // 100, mouse_pos[1] // 100)
        cell = self.cells[(cell_coord)]
        if cell.color == (0, 100, 200):
            cell.color = (200, 50, 50)
        else:
            cell.color = (0, 100, 200)

    def get_data(self):
        data = []
        for cell in self.cells.values():
            if cell.color == (0, 100, 200):
                data.append(0)
            else:
                data.append(1)
        return data

    def init_train(self):
        init_input = []
        for s in range(6):
            new_seq = [choice([0,1]) for i in range(9)]
            init_input.append(new_seq)

        init_output = [[]]
        font = pygame.font.Font(self.font, 20)
        for input_grid in init_input:
            self._draw_cells(input_grid)
            sleep(.4)
            cls()
            correct = input("How do you classify this board? (0/1) ")
            if correct and (correct in "0123456789"):
                correct = int(correct)
                init_output[0].append(correct)
                print("I'll remember that.")
            else:
                print("Cool, you clearly don't care very much about what we're doing here.")
            sleep(.5)

        self.nn.input_data = init_input
        self.nn.output_data = init_output
        self._draw_cells([0 for i in range(9)])
        return True

    def test_loop(self):
        """Update graphics and check for pygame events."""
        cls()
        print("_"*80)
        print("Great. Now click on tiles on the board to change their color and make your own \npatterns. Press 'Enter' to have me take a guess.\n")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type is pygame.QUIT:
                    running = False
                elif event.type is pygame.MOUSEBUTTONDOWN:
                    self.paint_cell(event.pos)
                    self._redraw()
                elif event.type is pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.nn.test(self.get_data())
                        self._draw_cells([0 for i in range(9)])
                        cls()
                        print("_"*80)
                        print("Click on tiles on the board to change their color and make your own \npatterns. Press 'Enter' to have me take a guess.\n")
class Cell():
    def __init__(self, draw_screen, coordinates, dimensions):
        self.draw_screen = draw_screen
        self.coordinates = coordinates
        self.dimensions = dimensions
        self.color = (127, 127, 127)

    def draw(self):
        line_width = 0
        rect = pygame.Rect(self.coordinates, self.dimensions)
        pygame.draw.rect(self.draw_screen, self.color, rect, line_width)
        pygame.display.update()
        # if self.color == BLUE:
        #     pygame.draw.rect(self.draw_screen, RED, rect, line_width)
        #     pygame.display.update()
        #     sleep(.3)
        #     pygame.draw.rect(self.draw_screen, BLUE, rect, line_width)
        #     pygame.display.update()
        #     sleep(.1)
        # else:
        #     pygame.draw.rect(self.draw_screen, BLUE, rect, line_width)
        #     pygame.display.update()
        #     sleep(.3)
        #     pygame.draw.rect(self.draw_screen, RED, rect, line_width)
        #     pygame.display.update()
        #     sleep(.1)

if __name__ == "__main__":
    # choose a random seed for reproducible results
    pygame.font.init()
    np.random.seed(1)
    nn = NN(9, 9, 1)
    grid = GridWorld(nn)
    open = grid.init_train()
    if open:
        grid.test_loop()
