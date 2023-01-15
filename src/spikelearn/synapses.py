"""
Implement static and plastic synapses

A synapse object should implement three methods:

- A `__call__` method taking an arbitrary number of inputs and returning
  the synapse output
- An update(xo) method that passes the output of the neurons
- A reset() method

Update and reset can be dummies, but a SpikingNet expects that
they will be implemented.

"""

import numpy as np
from .transforms import LowPass, PassThrough
from .trace import Trace


class BaseSynapse:
    """
    Base synapse, where synaptic weights are stored in a 2D array.
    """

    def __init__(self, Ne, No, W0, transform=None, syn_type=None):
        """
        
        Args:

        Ne : number of presynaptic neurons
        No : number of postsynaptic neurons
        W0 : a 2D array with the initial synaptic weights
        transforms : input transform
        syn_type : type of synapse, one of exc, inh, hybrid, None
        """

        self.Ne = Ne
        self.No = No
        self._W = W0
        self.out = np.zeros(self.No)
        
        self.has_transform = transform is not None
        if self.has_transform:
            self.transform = transform
        else:
            self.transform = lambda x:  x
        self.syn_type = syn_type

    def __call__(self, xe):
        if self.syn_type == "inh":
            self.out = - self.W @ self.transform(xe)
        else:
            self.out =  self.W @ self.transform(xe)
        return self.out


    def reset(self):
        if self.has_transform:
            self.transform.reset()

    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, W):
        self._W = W

    def update(self, pos, learn=True):
        pass


StaticSynapse = BaseSynapse


class OneToOneSynapse(BaseSynapse):

    def __call__(self, xe):
        if self.syn_type == "inh":
            return - self.W * self.transform(xe)
        else:
            return self.W * self.transform(xe)


class PlasticSynapse(BaseSynapse):
    """Simple plastic synapse implementing STDP

    """

    def __init__(self, Ne, No, W0, tre, tro, transform=None,
        rule_params = None, Wlim=1, syn_type=None, tracelim=10):
        """
        Args:

        Ne : number of presynaptic neurons
        No : number of postsynaptic neurons
        W0 : a 2D array with the initial synaptic weights
        tre : presynaptic trace tuple
        tro : postsynaptic trace tuple
        transform : transform function applied to inputs, defaults None
        rule_params: parameters defining synaptic plasticity rule
        Wlim : clamping parameter for synaptic weights
        syn_type : type of synapse ("exc", "inh", None)
        tracelim : clamping parameter for synaptic traces
        
        """

        super().__init__(Ne, No, W0, syn_type=syn_type, transform=transform)

        self.tre = tre
        self.tro = tro
        self.Wlim = Wlim
        self.tracelim = tracelim
        self.rule_params = rule_params

        self.te = Trace(self.Ne, self.tre[0], self.tre[1], self.tracelim)
        self.to = Trace(self.No, self.tro[0], self.tro[1], self.tracelim)


    def reset(self):

        super().reset()
        self.te.reset()
        self.to.reset()

    def __call__(self, xe):

        self.xe = self.transform(xe)
        if self.syn_type == "inh":
            return - self.W @ self.xe
        else:
            return self.W @ self.xe


    def update(self, xo, learn=True):

        self.te.update(self.xe)
        self.to.update(xo)

        if learn:

            dW = self.apply_rule(self.xe, xo)
            self.W += dW
            self.W[self.W > self.Wlim] = self.Wlim
            if self.syn_type is None:
                self.W[self.W < -self.Wlim] = -self.Wlim
            else:
                self.W[self.W < 0] = 0

    def apply_rule(self, xe, xo):
        
        dW = self.rule_params["Ap"]*np.outer(xo,self.te())
        dW -= self.rule_params["An"]*np.outer(self.to(), xe)
        return dW


class TernarySynapse(PlasticSynapse):
    """Simple plastic synapse implementing modulated STDP

    """

    def __init__(self, Ne, No, W0, tre, tro, trm, transform=None,
        transform_mod=None,
        rule_params = None, Wlim=1, syn_type=None, tracelim=10):
        """
        Args:
            Ne : number of presynaptic neurons
            No : number of postsynaptic neurons
            W0 : a 2D array with the initial synaptic weights
            tre : presynaptic trace tuple
            tro : postsynaptic trace tuple
            trm : modulatory trace tuple
            transform : transform function applied to inputs
            transform_mod : transform function applied to modulatory input
            rule_params: parameters defining synaptic plasticity rule
            Wlim : clamping parameter for synaptic weights
            syn_type : type of synapse ("exc", "inh", None)
            tracelim : clamping parameter for synaptic traces
        
        """

        super().__init__(Ne, No, W0, tre, tro, transform=transform,
            rule_params=rule_params, syn_type=syn_type, tracelim=tracelim)

        self.trm = trm
        self.Wlim = Wlim
        self.tm = Trace(self.No, self.trm[0], self.trm[1], self.tracelim)

        self.has_transform_mod = transform_mod is not None

        if self.has_transform_mod:
            self.transform_mod = transform_mod
        else:
            self.transform_mod = lambda x : x


    def reset(self):

        super().reset()
        self.tm.reset()
        if self.has_transform_mod:
            self.transform_mod.reset()


    def __call__(self, xe, xm):

        self.xe = self.transform(xe)
        self.xm = self.transform_mod(xm)
        if self.syn_type == "inh":
            return - self.W @ self.xe
        else:
            return self.W @ self.xe


    def update(self, xo, learn=True):

        self.te.update(self.xe)
        self.to.update(xo)
        self.tm.update(self.xm)

        if learn:

            dW = self.apply_rule(self.xe, xo)
            self.W += dW
            self.W[self.W > self.Wlim] = self.Wlim
            if self.syn_type is None:
                self.W[self.W < -self.Wlim] = -self.Wlim
            else:
                self.W[self.W < 0] = 0

    def apply_rule(self, xe, xo, xm=None):
        
        dW = self.rule_params["Ap"]*np.outer(xo,self.te())
        dW -= self.rule_params["An"]*np.outer(self.to(), xe)
        return self.tm()*dW


STDPSynapse = PlasticSynapse


class ModSTDPSynapse(TernarySynapse):

    def apply_rule(self, xe, xo, xm=None):
        
        dW = self.rule_params["Ap"]*np.outer(xo,self.te())
        dW -= self.rule_params["An"]*np.outer(self.to(), xe)
        return self.tm()*dW


class MSESynapse(TernarySynapse):
    """Simple plastic synapse implementing non-hebbian MSE rule

    """

    def apply_rule(self, xe, xo):        
        dW = self.rule_params["lr"]*np.outer(self.xm-self.to(),self.te())
        return dW


