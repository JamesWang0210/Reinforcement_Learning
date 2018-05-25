import sys
import numpy as np

m_input = sys.argv[1]  # Access to the Environment input.txt
val = sys.argv[2]  # Path to the Output Feedback from the Environment
action_seq = sys.argv[3]  # Path to the Action Sequence


class Environment:
    def __init__(self):
        self.maze = []  # Numpy Array that contains the maze

        self.line = 0  # Number of Lines of the Maze
        self.s = 0  # Number of Possible States in one line

        self.action = []  # To get action sequences from the file
        self.output = []  # For Output.feedback

        self.x = 0  # X coordination of current state
        self.y = 0  # Y coordination of current state

    def read_maze(self):
        self.maze = np.loadtxt(m_input, dtype=basestring)
        self.line = len(self.maze)
        self.s = len(self.maze[0])

    def read_action(self):
        self.action = np.loadtxt(action_seq, dtype=int, delimiter=" ")

    def check(self, x, y, a):
        (x_1, y_1, r, g) = (0, 0, 0, 0)

        if a == 0:  # Action West
            if y - 1 == -1:
                (x_1, y_1) = (x, y)
            else:
                if self.maze[x][y-1] == "*":
                    (x_1, y_1) = (x, y)
                else:
                    (x_1, y_1) = (x, y - 1)

        if a == 1:  # Action North
            if x - 1 == -1:
                (x_1, y_1) = (x, y)
            else:
                if self.maze[x-1][y] == "*":
                    (x_1, y_1) = (x, y)
                else:
                    (x_1, y_1) = (x - 1, y)

        if a == 2:  # Action East
            if y + 1 == self.s:
                (x_1, y_1) = (x, y)
            else:
                if self.maze[x][y+1] == "*":
                    (x_1, y_1) = (x, y)
                else:
                    (x_1, y_1) = (x, y + 1)

        if a == 3:  # Action South
            if x + 1 == self.line:
                (x_1, y_1) = (x, y)
            else:
                if self.maze[x+1][y] == "*":
                    (x_1, y_1) = (x, y)
                else:
                    (x_1, y_1) = (x + 1, y)

        if self.maze[x_1][y_1] == "G":
            r = 0
            g = 1
        else:
            r = -1
            g = 0

        self.output = np.array([x_1, y_1, r, g])

    def reset(self):
        self.x = self.line - 1
        self.y = 0

    def learning_step(self):
        self.reset()  # Reset the agent state to the "S"
        for a in range(0, len(self.action)):
            self.check(self.x, self.y, self.action[a])
            self.x = self.output[0]
            self.y = self.output[1]

            if a == 0:
                f = open(val, "w")
            else:
                f = open(val, "a")

            f.write(str(self.x) + ' ' + str(self.y) + ' ' + str(self.output[2]) +  ' ' + str(self.output[3]) + '\n')
            f.close()

e = Environment()

e.read_maze()
e.read_action()

e.reset()
e.learning_step()
