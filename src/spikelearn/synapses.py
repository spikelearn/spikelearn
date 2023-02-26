#Copyright Argonne 2022. See LICENSE.md for details.
"""
Implement static and plastic binary synapses

A synapse object should implement three methods:

- A `__call__` method taking an arbitrary number of inputs and returning
  the synapse output
- An `update(xo)` method that passes the output of the neurons
- A `reset()` method

Update and reset can be dummies, but a SpikingNet object expects that
they will be implemented.

"""

import numpy as np
from .rules import STDPRule


class BaseSynapse:
    """
    Base class for a synapse.

    Args:

        Ne : dimensions of presynaptic neurons
        No : dimensions of postsynaptic neurons
        transform : input transform
        syn_type : type of synapse, one of exc, inh, hybrid, None

   
    """

    def __init__(self, Ne, No, W0, transform=None, learning_rule=None, syn_type=None):

        self.Ne = Ne
        self.No = No
        self.W = W0
        self.out = np.zeros(self.No)

        if syn_type is None:
            self.syn_type = 'hybrid'
        else:
            self.syn_type = syn_type

        self.has_transform = transform is not None
        if self.has_transform:
            self.transform = transform
        else:
            self.transform = lambda x: x
        
        
        self._set_learning_rule(learning_rule)


    def _set_learning_rule(self, learning_rule):

        self.learning_rule = learning_rule
        if self.learning_rule is None:
            self._plastic = False
        else:
            self._plastic = True
            self.learning_rule.init(self.Ne, self.No)


    def __call__(self, xe):
        self.xe = xe
        return self.calc(xe)


    def calc(self, xe):
        if self.syn_type == "inh":
            self.out = - self.W @ self.transform(xe)
        else:
            self.out =  self.W @ self.transform(xe)
        return self.out


    def reset(self):
        if self.has_transform:
            self.transform.reset()
        if self._plastic:
            self.learning_rule.reset()


    def update(self, xo, learn=True):
        if self._plastic:
            if learn:
                self.W = self.learning_rule.update(self.xe, xo, self.W, learn)
            else:
                self.learning_rule.update(self.xe, xo)

    @property
    def W(self):
        """Returns the synaptic weights"""
        return self._W
    
    @W.setter
    def W(self, W):
        self._W = W


class StaticSynapse(BaseSynapse):
    """
    Base class for a synapse where synaptic weights are stored in a 2D array.
       
    Args:

        Ne : number of presynaptic neurons
        No : number of postsynaptic neurons
        W0 : a 2D array with the initial synaptic weights
        transform : input transform
        syn_type : type of synapse, one of exc, inh, hybrid, None

    """

    def __init__(self, Ne, No, W0, transform=None, syn_type=None):

        super().__init__(Ne, No, W0, transform, None, syn_type)



class OneToOneSynapse(BaseSynapse):
    """
    One to one static synapse

    Implement one to one static synapse chaining each input to its corresponding
    output.

    Args:

        Ne : Dimensions of neurons in the layer
        W0 : Synaptic weights
        transform : input transform
        syn_type : type of synapse, one of exc, inh, hybrid, None

    """

    def __init__(self, Ne, W0, transform=None, learning_rule=None,
                 syn_type=None):

        super().__init__(Ne, Ne, W0, transform, learning_rule, syn_type)


    def calc(self, xe):
        self.xe = xe
        if self.syn_type == "inh":
            return - self.W * self.transform(xe)
        else:
            return self.W * self.transform(xe)


class STDPSynapse(BaseSynapse):

    def __init__(self, Ne, No, W0, tre, tro, transform=None,
        rule_params = None, Wlim=1, syn_type=None, tracelim=10):

        self.learning_rule = STDPRule(rule_params, tre, tro, tracelim, Wlim)
        super().__init__(Ne, No, W0, transform, self.learning_rule, syn_type)


    def old_apply_rule(self, xe, xo):
        
        dW = self.rule_params["Ap"]*np.outer(xo,self.te())
        dW -= self.rule_params["An"]*np.outer(self.to(), xe)
        return dW


