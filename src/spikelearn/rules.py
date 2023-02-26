#Copyright Argonne 2022. See LICENSE.md for details.
"""
Implement static and plastic synapses

A synapse object should implement three methods:

- A `__call__` method taking an arbitrary number of inputs and returning
  the synapse output
- An `update(xo)` method that passes the output of the neurons
- A `reset()` method

Update and reset can be dummies, but a SpikingNet expects that
they will be implemented.

"""

from .trace import Trace
import numpy as np


class LearningRule:
    """Base class implementing a learning rule"""

    def __init__(self, rule_params, tre=None, tro=None, tracelim=10, Wlim=1):
        self.rule_params = rule_params
        self.tre = tre
        self.tro = tro
        if (self.tre is None) or (self.tro is None):
            self.has_traces = False
        else:
            self.has_traces = True

        self.tracelim = tracelim
        self.Wlim = Wlim

    def init(self, Ne, No, syn_type=None):
        if self.has_traces:
            self.te = Trace(Ne, self.tre[0], self.tre[1], self.tracelim)
            self.to = Trace(No, self.tro[0], self.tro[1], self.tracelim)
        else:
            self.te = None
            self.to = None
        self.syn_type = syn_type

    def update(self, xe, xo, W=None, learn=True):

        if self.has_traces:
            self.te.update(xe)
            self.to.update(xo)

        if learn:

            dW = self.apply_rule(xe, xo)
            W += dW
            W[W > self.Wlim] = self.Wlim
            if self.syn_type is None:
                W[W < -self.Wlim] = -self.Wlim
            else:
                W[W < 0] = 0
        return W


    def reset(self):
        if self.has_traces:
            self.te.reset()
            self.to.reset()

    def apply_rule(self, xe, xo):
        raise NotImplemented

class STDPRule(LearningRule):

    def apply_rule(self, xe, xo):

        dW = self.rule_params["Ap"]*np.outer(xo,self.te())
        dW -= self.rule_params["An"]*np.outer(self.to(), xe)
        return dW        


class ModulatedLearningRule:
    """Base class implementing a modulated learning rule"""

    def __init__(self, rule_params, tre=None, tro=None, trm=None, 
            tracelim=10, Wlim=1):
        self.rule_params = rule_params
        self.tre = tre
        self.tro = tro
        self.trm = trm
        if (self.tre is None) or (self.tro is None) or (self.trm is None):
            self.has_traces = False
        else:
            self.has_traces = True

        self.tracelim = tracelim
        self.Wlim = Wlim

    def init(self, Ne, No, Nm, syn_type=None):
        if self.has_traces:
            self.te = Trace(Ne, self.tre[0], self.tre[1], self.tracelim)
            self.to = Trace(No, self.tro[0], self.tro[1], self.tracelim)
            self.tm = Trace(Nm, self.trm[0], self.trm[1], self.tracelim)

        else:
            self.te = None
            self.to = None
            self.tm = None
        self.syn_type = syn_type

    def update(self, xe, xo, xm, W=None, learn=True):

        if self.has_traces:
            self.te.update(xe)
            self.to.update(xo)
            self.tm.update(xm)

        if learn:

            dW = self.apply_rule(xe, xo, xm)
            W += dW
            W[W > self.Wlim] = self.Wlim
            if self.syn_type is None:
                W[W < -self.Wlim] = -self.Wlim
            else:
                W[W < 0] = 0
        return W


    def reset(self):
        if self.has_traces:
            self.te.reset()
            self.to.reset()
            self.tm.reset()

    def apply_rule(self, xe, xo, xm):
        raise NotImplemented


class ModSTDPRule(ModulatedLearningRule):

    def apply_rule(self, xe, xo, xm):
        
        dW = self.rule_params["Ap"]*np.outer(xo,self.te())
        dW -= self.rule_params["An"]*np.outer(self.to(), xe)
        return self.tm()*dW


class MSERule(ModulatedLearningRule):
    """Simple plastic synapse implementing non-hebbian MSE rule

    """

    def apply_rule(self, xe, xo, xm):        
        dW = self.rule_params["lr"]*np.outer(xm-self.to(),self.te())
        return dW

