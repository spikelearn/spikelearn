#Copyright Argonne 2022. See LICENSE.md for details.

import json
#from .streamnet import StreamGraph

def read_json(filename):
    data = json.load(open(filename, "r"))
    return parse_json(data)

def parse_json(data):
    """Ensures that the json file contains a well formed structure"""
    nin, nout = data["ports"]

    elements = {el['name'] : el["ports"] for el in data["elements"]}
    el_names = set(elements.keys())
    inport_names = set("inp{}".format(i+1) for i in range(nin))
    outport_names = set("outp{}".format(i+1) for i in range(nout))
    for name in el_names:
        if name in inport_names or name in outport_names:
            raise ValueError("Element name {} not allowed".format(name))

    all_names = el_names | inport_names | outport_names
    outdegrees = {}
    indegrees = {}
    for k, (el_nin, el_nout) in elements.items():
        for i in range(el_nin):
            indegrees[(k,i+1)] = set()
        for i in range(el_nout):
            outdegrees[(k,i+1)] = set()

    for name in inport_names:
        outdegrees[(name, 1)] = set()

    for name in outport_names:
        indegrees[(name, 1)] = set()

    for el_from, port_from, el_to, port_to in data["connections"]:
        if el_from not in all_names:
            raise ValueError("Element or port {} not found".format(el_from))
        if el_to not in all_names:
            raise ValueError("Element or port {} not found".format(el_to))

        if el_from in el_names:
            el_nin, el_nout = elements[el_from]
            if port_from > el_nout or port_from < 1:
                raise ValueError("{} is not a valid out pin number for element {}".format(port_from, el_from))
            outdegrees[(el_from, port_from)].add((el_to, port_to))
        elif el_from in inport_names:
            if port_from != 1:
                raise ValueError("{} is not a valid pin number for port {}".format(port_from, el_from))
            outdegrees[(el_from, 1)].add((el_to, port_to))
        elif el_from in outport_names:
            raise ValueError(
                "Invalid directed edge: {} cannot have an outdegree greater than zero".format(el_from))
        else:
            raise ValueError("Element or port {} not found".format(el_from))
               
        if el_to in el_names:
            el_nin, el_nout = elements[el_to]
            if port_to > el_nin or port_to < 1:
                raise ValueError("{} is not a valid in pin number for element {}".format(port_to, el_to))
            indegrees[(el_to, port_to)].add((el_from, port_from))
            if len(indegrees[(el_to, port_to)]) > 1:
                raise ValueError("in pin {} of element {} has an indegree greater than one".format(port_to, el_to))
        elif el_to in outport_names:
            if port_to != 1:
                raise ValueError("{} is not a valid pin number for port {}".format(port_to, el_to))
            indegrees[(el_to, 1)].add((el_from, port_from))
            if len(indegrees[(el_to,1)]) > 1:
                raise ValueError(
                    "Output port {} cannot have an indegree greater than one".format(el_to))
        elif el_to in inport_names:
            raise ValueError(
                "Invalid directed edge: {} cannot have an outdegree greater than zero".format(el_to))
        else:
            raise ValueError("Element or port {} not found".format(el_from))
        
    return StreamGraph(elements, inport_names, outport_names, indegrees, outdegrees)
        

        

