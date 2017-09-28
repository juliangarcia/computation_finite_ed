from numpy.random import geometric
from numpy.random import rand
from numpy import exp
import numba
import sys
import numpy as np
import json


@numba.njit
def step(population_size, a, b, c, d, intensity_of_selection, type_a):
    w_a = exp(
        intensity_of_selection * (1.0 / (population_size - 1)) * (a * (type_a - 1) + b * (population_size - type_a)))
    w_b = exp(
        intensity_of_selection * (1.0 / (population_size - 1)) * (c * type_a + d * (population_size - type_a - 1)))
    mb = (type_a * w_a) / (type_a * w_a + (population_size - type_a) * w_b)
    md = type_a / population_size
    nothing_happens = mb * md + (1 - mb) * (1 - md)
    k = geometric(1.0 - nothing_happens)  # how many steps until something happens
    if rand() < w_a / (w_a + w_b):  # is this nothing happens?
        type_a += 1
    else:
        type_a -= 1
    return k, type_a
