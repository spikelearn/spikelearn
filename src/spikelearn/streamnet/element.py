#Copyright Argonne 2022. See LICENSE.md for details.

from collections import namedtuple

#Pin = namedtuple("Pin", ["name", "value", "conn"], defaults=["", None, []])

class InPin:

    def __init__(self, parent, name, from_pin=None):
        self.parent = parent
        self.name = name
        self.value = None
        self.from_conn = from_pin

    @property
    def conn(self):
        return self._from_conn

    @conn.setter
    def conn(self, from_pin):
        self._from_conn = from_pin
        if from_pin is not None:
            from_pin.add(self)

    def pull(self):
        self.value = self.conn.value




class OutPin:
    
    def __init__(self, parent, name, value=None, to_pins=set()):
        self.parent = parent
        self.name = name
        self.value = value
        self._to_conn = to_pins

    def add(self, to_pin):
        self._to_conn.add(to_pin)

    def conns(self):
        return self._to_conn

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v
                
    def push(self):
        for pin in self._to_conn:
            pin.value = self.value


def connect(from_pin, to_pin):
    to_pin.conn = from_pin



class OutPort(InPin):
    def __init__(self, name, from_pin=None):
        super().__init__(None, name, from_pin)



class InPort(OutPin):
    def __init__(self, name, value=None, to_pins=set()):
        super().__init__(None, name, value, to_pins)



class Element:

    def __init__(self, el_object, default, n_out=1):

        self.object = el_object
        self.out = default
        self.n_out = n_out

    def run_method(self, name, *args):
        m = getattr(self.object, name)
        m(*args)
        
    def __call__(self, *args):
        if self.n_out == 1:
            self.out = [self.object(*args)]
        else:
            self.out = self.object(*args)




class OldElement:

    def __init__(self, name, ipins, opins, func=None):

        self.name = name
        self.ipin = []

        if hasattr(ipins, "__getitem__"):
            for name in ipins:
                setattr(self, name, InPin(self, name, None))
                self.ipin.append(getattr(self, name))
        else:
            for i in range(ipins):
                name = "i{}".format(i+1)
                setattr(self, name, InPin(self, name, None))
                self.ipin.append(getattr(self, name))

        self.opin = []
        if hasattr(opins, "__getitem__"):
            for name in opins:
                setattr(self, name, OutPin(self, name, None))
                self.opin.append(getattr(self, name))
        else:
            for i in range(opins):
                name = "o{}".format(i+1)
                setattr(self, name, OutPin(self, name, None))
                self.opin.append(getattr(self, name))
        
        self.Nin = len(self.ipin)
        self.Nout = len(self.opin)

        if func is None:
            self.func = self.default_func
        else:
            self.func = func
    

    def default_func(self, *args):
        return [0]*self.Nout

    def init(self, *args):
        if len(args) == 1 and not hasattr(args[0], "__len__"):
            default_out = [args[0]]*self.Nout
        else:
            default_out = args

        for op, dv in zip(self.opin, default_out):
            op.value = dv

    def push(self):
        for op in self.opin:
            op.push()

    def pull(self):
        for ip in self.ipin:
            ip.pull()

    def step(self):
        input_vals = [inp.value for inp in self.ipin]
        output_vals = self.func(*input_vals)
        for outp, oval in zip(self.opin, output_vals):
            outp.value = oval




