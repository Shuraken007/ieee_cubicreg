import numpy as np
import src.cubic_reg
import src.createdata
import src.easy_print as ep
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import spline


def test_f(x):
   z = x[0] + x[1] * 1.j
   return z**2 + 2 * z


def test_g(x):
   return 3 * x**2 + 6 * x


def test_h(x):
   return 6 * x + 6


if __name__ == '__main__':
   b
   +
