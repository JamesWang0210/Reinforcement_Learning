import sys
import numpy as np

m_input = sys.argv[1]  # Access to the Environment input.txt
val = sys.argv[2]  # Path to the Output Values
q_val = sys.argv[3]  # Path to the Output Q-values
policy = sys.argv[4]  # Path to the Optimal Policy
num_epoch = int(sys.argv[5])  # Number of Episodes
factor = float(sys.argv[6])  # Discount Factor Gama


class ValueInteration:
    def __init__(self):
        self.maze = []  # Numpy Array that contains the maze
        self.V = np.zeros((1, 1))  # Numpy Array that carries all the V-values
        self.V_1 = np.zeros((1, 1))  # Intermediate Numpy Array that carries all the V-values
        self.R = np.array([-1, -1, -1, -1])  # Immediate Reward

        self.line = 0  # Number of Lines of the Maze
        self.s = 0  # Number of Possible States in one line
        self.Point = []  # Contain All the Points
        self.All_Q = []  # List that contains all the four Q-values for each point

        self.first_y = 0  # The first '.' in the first line

    def read_maze(self):
        self.maze = np.loadtxt(m_input, dtype=basestring)
        self.line = len(self.maze)
        self.s = len(self.maze[0])

    def initialization(self):
        self.V = np.zeros((self.line, self.s), dtype=float)
        self.V_1 = np.zeros((self.line, self.s), dtype=float)
        for l in range(0, self.line):
            for s in range(0, self.s):
                (x, y) = (l, s)
                self.Point.append((x, y))

    def env(self, x, y):
        (a, b, c, d) = (0, 0, 0, 0)

        if y - 1 == -1:
            a = self.V[x, y]
        else:
            if self.maze[x][y-1] == "*":
                a = self.V[x, y]
            else:
                a = self.V[x, y-1]

        if x - 1 == -1:
            b = self.V[x, y]
        else:
            if self.maze[x-1][y] == "*":
                b = self.V[x, y]
            else:
                b = self.V[x-1, y]

        if y + 1 == self.s:
            c = self.V[x, y]
        else:
            if self.maze[x][y+1] == "*":
                c = self.V[x, y]
            else:
                c = self.V[x, y+1]

        if x + 1 == self.line:
            d = self.V[x, y]
        else:
            if self.maze[x+1][y] == "*":
                d = self.V[x, y]
            else:
                d = self.V[x+1, y]

        E = np.array([a, b, c, d])
        return E

    def learning(self):
        self.V = self.V_1
        self.All_Q = []
        self.V_1 = np.zeros((self.line, self.s), dtype=float)
        for p in range(0, len(self.Point)):
            x = self.Point[p][0]
            y = self.Point[p][1]
            if self.maze[x][y] == "*":
                Q = np.array([0, 0, 0, 0])
            else:
                if self.maze[x][y] == "G":
                    Q = np.array([0, 0, 0, 0])
                else:
                    Q = self.R + factor * self.env(x, y)

            self.All_Q.append(Q)
            self.V_1[x, y] = float(np.max(Q))

    def output_file(self):
        self.learning()  # Get the Final Q-values

        for s in range(0, self.s):
            if self.maze[0][s] == '.' or self.maze[0][s] == 'G':
                self.first_y = s
                break

        for p in range(0, len(self.Point)):
            x = self.Point[p][0]
            y = self.Point[p][1]
            if self.maze[x][y] != "*":
                if (x, y) == (0, self.first_y):
                    f_v = open(val, 'w')
                    f_q = open(q_val, "w")
                    f_p = open(policy, "w")
                else:
                    f_v = open(val, 'a')
                    f_q = open(q_val, "a")
                    f_p = open(policy, "a")

                po = float(np.argmax(self.All_Q[p]))

                f_v.write(str(x) + ' ' + str(y) + ' ' + str(self.V[x, y]) + '\n')

                f_q.write(str(x) + ' ' + str(y) + ' ' + str(0) + ' ' + str(float(self.All_Q[p][0])) + '\n'
                          + str(x) + ' ' + str(y) + ' ' + str(1) + ' ' + str(float(self.All_Q[p][1])) + '\n'
                          + str(x) + ' ' + str(y) + ' ' + str(2) + ' ' + str(float(self.All_Q[p][2])) + '\n'
                          + str(x) + ' ' + str(y) + ' ' + str(3) + ' ' + str(float(self.All_Q[p][3])) + '\n')

                f_p.write(str(x) + ' ' + str(y) + ' ' + str(po) + '\n')
                f_v.close()
                f_q.close()
                f_p.close()


m = ValueInteration()

m.read_maze()
m.initialization()

for e in range(0, num_epoch):
    m.learning()

m.output_file()
