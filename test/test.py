from pypower.loadcase import loadcase
from pypower.runpf import runpf
from pypower.makeYbus import makeYbus
r = loadcase('..\\200bus\\case_ACTIVSg200')
[Ybus, Yf, Yt] = makeYbus(r)
