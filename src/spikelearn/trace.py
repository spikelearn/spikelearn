#Copyright Argonne 2022. See LICENSE.md for details.

import numpy as np

class Trace:

    def __init__(self, N, t0, t1, tracelim):
        self.N = N
        self.t0 = t0
        self.t1 = t1
        self.tracelim = tracelim
        self.reset()

    def reset(self):
        self.t = np.zeros(self.N)

    def update(self, x):
        self.t = self.t0*x + self.t1*self.t
        self.t[self.t>self.tracelim] = self.tracelim

    def __call__(self):
        return self.t

