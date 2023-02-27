
class NeuronNode:

    def __init__(self, neuron_model):
        self._neuron = neuron_model
        self._synapses = []
        self._syn_types = []
        self._n_pre = []
        self.out = self._neuron.out
        self.total_syn_inputs = 0

    def add_synapse(self, synapse, n_pre=1):
        self._synapses.append(synapse)
        self._n_pre.append(n_pre)
        self.total_syn_inputs += n_pre

    def reset(self):
        self._neuron.reset()
        for syn in self._synapses:
            syn.reset()

    def __call__(self, *args):
        if len(self._synapses) == 0:
            self.out = self._neuron(*args)
        else:
            if self._neuron.group_synapses:
                xe_list = []
                xi_list = []
                n = 0
                for i, syn in enumerate(self._synapses):
                    n_inputs = self._n_pre[i]
                    if syn.syn_type == 'exc':
                        ap_list = xe_list
                    elif syn.syn_type == 'inh':
                        ap_list = xi_list
                    else:
                        raise ValueError("Synapse type must be either exc or inh")
                    if n_inputs == 1:
                        ap_list.append(syn(args[n]))
                    else:
                        ap_list.append(syn(*args[n:(n+n_inputs)]))
                    n += n_inputs
                if len(xe_list) > 0:
                    xe = sum(xe_list)
                else:
                    xe = None
                if len(xi_list) > 0:
                    xi = sum(xi_list)
                else:
                    xi = None
                self.out = self._neuron(xe, xi)

            else:
                neuron_args = []
                n = 0
                for i, syn in enumerate(self._synapses):
                    n_inputs = self._n_pre[i]
                    if n_inputs == 1:
                        neuron_args.append(syn(args[n]))
                    else:
                        neuron_args.append(syn(*args[n:(n+n_inputs)]))
                    n += n_inputs
                self.out = self._neuron(*neuron_args)
        return self.out


    def update(self, learn):
        for syn in self._synapses:
            syn.update(self.out, learn)

