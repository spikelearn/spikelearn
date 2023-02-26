from .synapses import BaseSynapse
from .rules import MSERule


class TernarySynapse(BaseSynapse):
    """Ternary synapse with pre, post, and modulatory interaction

    """

    def __init__(self, Ne, No, Nm, W0, transform=None, transform_m=None,
                 learning_rule=None, syn_type=None):
        self.Nm = Nm

        self.has_transform_m = transform_m is not None
        if self.has_transform_m:
            self.transform_m = transform
        else:
            self.transform_m = lambda x: x
        
        super().__init__(Ne, No, W0, transform, learning_rule, syn_type)
      

    def _set_learning_rule(self, learning_rule):

        self.learning_rule = learning_rule
        if self.learning_rule is None:
            self._plastic = False
        else:
            self._plastic = True
            self.learning_rule.init(self.Ne, self.No, self.Nm)

    def __call__(self, xe, xm):
        self.xe = xe
        self.xm = xm
        return self.calc(xe, xm)


    def calc(self, xe, xm):
        if self.syn_type == "inh":
            self.out = - self.W @ self.transform(xe)
        else:
            self.out =  self.W @ self.transform(xe)
        return self.out


    def reset(self):
        if self.has_transform_m:
            self.transform_m.reset()
        super().reset()


    def update(self, xo, learn=True):
        if self._plastic:
            if learn:
                self.W = self.learning_rule.update(self.xe, xo, self.xm, self.W, learn)
            else:
                self.learning_rule.update(self.xe, xo, self.xm)



class MSESynapse(TernarySynapse):
    """Plastic synapse implementing non-hebbian MSE rule

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

        learning_rule = MSERule(rule_params, tre, tro, trm, tracelim, Wlim)
        super().__init__(Ne, No, No, W0, transform, transform_mod, learning_rule,
                         syn_type)

