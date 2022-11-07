import pickle
import numpy as np
import math
from decimal import *
from numpy.linalg import norm
from cmath import polar
import sys
from ramanbond import *


def build_supercell(n, atomnote, coord, charge, vec1, vec2):
  superatomnote = atomnote * (n*n)
  temp = []
  for i in range(-1*int(n/2), int(n/2)+1):
    for j in range(-1*int(n/2), int(n/2)+1):
      temp.append(coord + i * vec1 + j * vec2)
  
  supercoord = temp[0].copy()
  for i in range(1, len(temp)):
    supercoord = np.concatenate((supercoord, temp[i] ))

  supercharge = charge.copy()
  for i in range(n*n-1):
    supercharge = np.concatenate((supercharge, charge))

  return superatomnote, supercoord, supercharge


class polbond_periodic2d:
  def __init__(self, p, m, p1, p2, efield): ## efield in hart
    self.charge = (p.charge - m.charge) / (efield * 2)
    self.num = p.num ## number of atoms in the unit cell
    self.coord = p.coord
    self.pol = (p.dipole - m.dipole) / (efield * 2)
    self.vec1 = p.vec1
    self.vec2 = p.vec2
    self.atomnote = p.atomnote
    n = 3
    self.superatomnote, self.supercoord, self.supercharge = build_supercell(n, self.atomnote, self.coord, self.charge, self.vec1, self.vec2)

    self.dis = calc_dismatrix(self.supercoord)
    self.anglecos = calc_cosmatrix(self.supercoord)[2]


    self.polatom = {}
    for i in range(self.num):
      self.polatom.update({ i + 4 * self.num : (p.atomdip[i][0] - m.atomdip[i][0]) / (efield * 2) })

    ## store infor temporarily needed to calculate L matrix in before_L
    before_L = np.zeros(self.dis.shape)

    getcontext().prec = 28	
    for i in range(before_L.shape[0]):
      for j in range(before_L.shape[1]):
        if i != j:
          try:
            before_L[i][j] = ( (1 / (2.0 * pf(p1, p2, self.superatomnote[i], self.superatomnote[j], self.dis[i][j], self.anglecos[i][j] ))))
          except OverflowError:
            before_L[i][j] = 0.0
 

    L = np.zeros(before_L.shape)
    for i in range(L.shape[0]):
      for j in range(L.shape[1]):
        if j == i:
          L[i][j] = Decimal(-1*sum(before_L[i])+ 1.0) # add an arbitrary const C = 1.0
        elif j != i:
          L[i][j] = Decimal(before_L[i][j] + 1.0) # add an arbitrary const C 
    ############################################################################################
    ## Matrix equation: L Lambda = (induced charge in one direction)                           #
    ## for induced charge in x direction, we will have a set of lambdas as solution in lam_x   #
    ## similary, we have lam_y, lam_z                                                          #
    ## charge transfer Qij will be calculated based on lam_x, lam_y and lam_z                  #
    ############################################################################################
  
    lam = np.linalg.solve(L, self.supercharge)
  
    self.Q = np.zeros(self.dis.shape)

    for i in range(self.Q.shape[0]):
      for j in range(self.Q.shape[1]):
        if j != i:
          self.Q[i][j] = (-1.0 / (2.0 * pf(p1, p2, self.superatomnote[i], self.superatomnote[j], self.dis[i][j], self.anglecos[i][j] ))) * (lam[i] - lam[j])


#    print(self.Q[9-1+int((n*n-1) / 2)*self.num][1-1+int((n*n-1) / 2)*self.num])
#    print(self.Q[1-1+int((n*n-1) / 2)*self.num][4-1+int((n*n-1) / 2)*self.num])
#    print(self.Q[1-1+int((n*n-1) / 2)*self.num][4-1+int((n*n-1) / 2 - 1)*self.num])


    self.polbond = {}
    for i in range(4 * self.num, 5 * self.num): ## focus on the center cell
      for j in range(self.Q.shape[1]):
        if j in range(0, 4 * self.num):
          self.polbond.update( {(i, j) : self.Q[i][j] * (self.supercoord[i][2] - self.supercoord[j][2]) })
        if j in range(4 * self.num, 5 * self.num) and j < i:
          self.polbond.update( {(i, j) : self.Q[i][j] * (self.supercoord[i][2] - self.supercoord[j][2]) })

  


class collect_bandspt_periodic2d:
  def __init__(self, filename, closeshell):

    f = open(filename)
    f1 = f.readlines()
    f.close()
    for i in range(len(f1)):
      if 'Total nr. of atoms' in f1[i]:
        self.num = int(f1[i].strip('\n').split()[-1])
  
  
    self.atomdip = np.zeros((self.num, 1))

    self.charge  = np.zeros((self.num, 1))
    for i in range(len(f1)):

      if 'Deformation charges with respect to neutral atoms' in f1[i]:
        for ii in range(self.num):
          self.charge[ii][0] = float(f1[i+5+ii].strip('\n').split()[2])

  
      if 'Hirshfeld atomic dipole z' in f1[i]:
        if closeshell == True:
          for ii in range(self.num):
            self.atomdip[ii][0] = (float(f1[i+1+ii].strip('\n').split()[-1]))
        elif closeshell == False:
          for ii in range(self.num):
            self.atomdip[ii][0] = (float(f1[i+1+ii].strip('\n').split()[-2])) + (float(f1[i+1+ii].strip('\n').split()[-1]))
        else:
          print('error', closeshell)

      if 'D I P O L E' in f1[i]:
        self.dipole = float(f1[i+6].strip('\n').split()[-2])


    self.coord =  np.zeros((self.num, 3))
    self.atomnote = []

    for i in range(len(f1)):
      if 'Geometry' in f1[i]:
        while True:
          if len(f1[i].split()) == 5:
            break
          else:
            i += 1

        for ii in range(self.num):
          self.coord[ii][0] = float(f1[i+ii].strip('\n').split()[2])
          self.coord[ii][1] = float(f1[i+ii].strip('\n').split()[3])
          self.coord[ii][2] = float(f1[i+ii].strip('\n').split()[4])
          self.atomnote.append(     f1[i+ii].strip('\n').split()[1])

        self.vec1 = np.array([float(f1[i+2+self.num].strip('\n').split()[1]),
                              float(f1[i+2+self.num].strip('\n').split()[2]),
                              float(f1[i+2+self.num].strip('\n').split()[3])])
        self.vec2 = np.array([float(f1[i+3+self.num].strip('\n').split()[1]),
                              float(f1[i+3+self.num].strip('\n').split()[2]),
                              float(f1[i+3+self.num].strip('\n').split()[3])])
        break


    self.coord = A2B * self.coord
    self.vec1 =  A2B * self.vec1
    self.vec2 =  A2B * self.vec2




class ramanbond_periodic2d:
  def __init__(self, vibp, vibm, stepsize):
    self.polder =(vibp.pol - vibm.pol) / (stepsize * 2)
    self.coord = 0.5 * (vibp.coord + vibm.coord) / A2B
    self.supercoord = 0.5 * (vibp.supercoord + vibm.supercoord) / A2B
    self.atomnote = vibp.atomnote
    self.num = len(self.atomnote)
    self.superatomnote = self.atomnote * 9
    self.vec1 = vibp.vec1 / A2B
    self.vec2 = vibp.vec2 / A2B

    self.ramanbond = twopoint_numdif(vibp.polbond, vibm.polbond, stepsize)
    self.ramanatom = twopoint_numdif(vibp.polatom, vibm.polatom, stepsize)

  def dumpdata_for_plot(self):
    printcoord('geo_periodic2d.xyz', self.superatomnote, self.supercoord)
    dumpdata('ramanbond_periodic2d.p', self.superatomnote, self.supercoord, self.ramanatom, self.ramanbond)

