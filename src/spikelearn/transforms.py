#Copyright Argonne 2022. See LICENSE.md for details.

import numpy as np

class BaseTrace:
    """ Implement the base class for a trace with optional delay

    """

    def __init__(self, N, delay=False):
        """
        Instantiates a low pass function
        
        Args:

            N : Number of neurons
            delay : Optional. If true, delays input for one timestep (default False)

        """
        self.N = N
        self.value = np.zeros(self.N)
        self.delay = delay
        self._trace = np.zeros(self.N)

    def reset(self):
        """Resets internal state
        """
        self.value = np.zeros(self.N)
        self._trace = np.zeros(self.N)

    @property
    def trace(self):
        return self._trace

    def _update(self, x):
        pass

    def __call__(self, x):
        """Computes the trace
        
        Passes a new input to the trace and updates its internal state

        Args:
            x : array of inputs
        
        Returns:
            Current trace value
        """
        if self.delay:
            self._trace = self.value.copy()
            self._update(x)
        else:
            self._update(x)
            self._trace = self.value.copy()
        return self.trace


class PassThrough(BaseTrace):

    """
    Pass-through function with optional delay

    """

    def _update(self, x):
        return x


class LowPass(BaseTrace):
    """ Implement a low pass function with optional delay

    """

    def __init__(self, N, tau, delay=False):
        """
        Instantiates a low pass function
        
        Args:

            N : Number of neurons
            tau : Characteristic time
            delay : Optional. If true, delays input for one timestep (default False)

        """
        self.beta  = np.exp(-1/tau)
        super().__init__(N, delay)


    def _update(self, x):
        self.value = (1-self.beta)*x + self.beta*self.value
