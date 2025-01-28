import math

from DDSim.DD4hepSimulation import DD4hepSimulation
from g4units import cm, mm, GeV, MeV, radian, m, degree
SIM = DD4hepSimulation()

SIM.numberOfEvents = 1000
SIM.enableGun = True
SIM.outputFile = "photon_angle_5to70GeV.edm4hep.root"

SIM.gun.particle = "gamma"
SIM.gun.momentumMin = 5*GeV
SIM.gun.momentumMax = 70*GeV

SIM.gun.thetaMin = 0.0168*radian
SIM.gun.thetaMax = 0.035*radian
SIM.gun.phiMin = 140*degree
SIM.gun.phiMax = 220*degree
SIM.gun.distribution = "uniform"
