#Copyright Argonne 2022. See LICENSE.md for details.


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


