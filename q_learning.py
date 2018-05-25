import sys
import numpy as np

m_input = sys.argv[1]  # Access to the Environment input.txt
val = sys.argv[2]  # Path to the Output Values
q_val = sys.argv[3]  # Path to the Output Q-values
policy = sys.argv[4]  # Path to the Optimal Policy
num_epoch = int(sys.argv[5])  # Number of Episodes
len_epoch = int(sys.argv[6])  # Maximum Length of One Episode
rate = float(sys.argv[7])  # Learning Rate Alpha
factor = float(sys.argv[8])  # Discount Factor Gama
ep = float(sys.argv[9])  # Epsilon Value for Epsilon-Greedy Strategy


class QLearning:
    def __init__(self):
        self.maze = []  # Numpy Array that contains the maze
        self.Q = np.zeros((1, 1))  # Numpy Array that carries all the Q-values
        self.Q_1 = np.zeros((1, 1))  # Numpy Array that carries all the Q-values with respect to four possible actions

        self.line = 0  # Number of Lines of the Maze
        self.s = 0  # Number of Possible States in one line

        self.step = 0  # Number of steps taken in one episode

        self.x = 0  # X coordination of current state
        self.y = 0  # Y coordination of current state
        self.x_1 = 0  # Intermediate Var to save x
        self.y_1 = 0  # Intermediate Var to save y
        self.first_y = 0  # The first '.' in the first line

        self.q_prime = 0  # Max Q-value for the next state with respect to four possible actions

    def read_maze(self):
        self.maze = np.loadtxt(m_input, dtype=basestring)
        self.line = len(self.maze)
        self.s = len(self.maze[0])

    def initialization(self):
        self.Q = np.zeros((self.line, self.s), dtype=float)
        self.Q_1 = np.zeros((4*self.line, self.s), dtype=float)

    def reset(self):
        self.x = self.line - 1
        self.y = 0

    def check(self, x, y, a):
        (x_1, y_1) = (0, 0)

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

        (self.x_1, self.y_1) = (x_1, y_1)  # New State
        Q_prime = self.Q_1[(4*x_1):(4*x_1+4), y_1]  # All Possible Q-values for the Next State
        self.q_prime = np.max(Q_prime)  # The Maximum of Q-values for State Prime

    def learning(self):
        self.step = 0
        while self.step <= len_epoch and self.maze[self.x][self.y] != 'G':
            q_1 = self.Q_1[(4*self.x):(4*self.x+4), self.y]  # With 1-epsilon probability
            q_2 = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])  # With epsilon probability

            if ep == 0:
                a = np.argmax(q_1)
            else:
                a = np.random.choice([np.argmax(q_1), q_2], p=[(1-ep), ep])
            self.check(self.x, self.y, a)  # Find the max Q-value for state prime
            '''
            Update the Q-value of Current State
            '''
            q_1[a] = (1 - rate)*q_1[a] + rate*(-1 + factor*self.q_prime)

            self.Q[self.x, self.y] = np.max(q_1)
            self.Q_1[(4 * self.x):(4 * self.x + 4), self.y] = q_1

            (self.x, self.y) = (self.x_1, self.y_1)  # Move to the next state
            self.step += 1

    def file_output(self):
        for s in range(0, self.s):
            if self.maze[0][s] == '.' or self.maze[0][s] == 'G':
                self.first_y = s
                break

        for l in range(0, self.line):
            for s in range(0, self.s):
                (x, y) = (l, s)
                if self.maze[x][y] != "*":
                    if (x, y) == (0, self.first_y):
                        f_1 = open(val, 'w')
                        f_2 = open(q_val, 'w')
                        f_3 = open(policy, 'w')
                    else:
                        f_1 = open(val, 'a')
                        f_2 = open(q_val, 'a')
                        f_3 = open(policy, 'a')

                    po = float(np.argmax(self.Q_1[(4*x):(4*x+4), y]))

                    f_1.write(str(x) + ' ' + str(y) + ' ' + str(self.Q[x, y]) + '\n')
                    f_2.write(str(x) + ' ' + str(y) + ' ' + str(0) + ' ' + str(float(self.Q_1[4*x, y])) + '\n'
                              + str(x) + ' ' + str(y) + ' ' + str(1) + ' ' + str(float(self.Q_1[4*x+1, y])) + '\n'
                              + str(x) + ' ' + str(y) + ' ' + str(2) + ' ' + str(float(self.Q_1[4*x+2, y])) + '\n'
                              + str(x) + ' ' + str(y) + ' ' + str(3) + ' ' + str(float(self.Q_1[4*x+3, y])) + '\n')
                    f_3.write(str(x) + ' ' + str(y) + ' ' + str(po) + '\n')

                    f_1.close()
                    f_2.close()
                    f_3.close()

q = QLearning()

q.read_maze()
q.initialization()

for e in range(0, num_epoch):
    q.reset()
    q.learning()

q.file_output()
