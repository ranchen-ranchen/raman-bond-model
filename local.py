import numpy as np
import math
from decimal import *
from numpy.linalg import norm
from cmath import polar
from ramanbond import *

class ramanbond(object):
  def __init__(self, vibp, vibm, stepsize, p1, p2, component):
    self.polder =(vibp.polADFprint - vibm.polADFprint) / (stepsize * 2)
    self.coord = 0.5 * (vibp.coord + vibm.coord)
    self.atomnote = vibp.atomnote
    self.num = len(self.atomnote)
    vibp.calc_polbond(p1, p2, component)
    vibm.calc_polbond(p1, p2, component)
    self.ramanbond = twopoint_numdif(vibp.polbond, vibm.polbond, stepsize)
    self.ramanatom = twopoint_numdif(vibp.polatom, vibm.polatom, stepsize)



class polbond(object):
  def __init__(self, f1):
  ## input is the list containing the lines of ADF outputs
    for i in range(len(f1)):
      if 'FRAGMENTS' in f1[i]:
        jj = 3
        while True:
          jj = jj + 1
          if f1[i+jj].strip('\n') == '':
            self.num = jj - 3
            break

    self.coord = np.zeros((self.num, 3))
    self.atomnote = []

    for i in range(len(f1)):
      if 'FRAGMENTS' in f1[i]:
        for j in range(self.num):
          self.coord[j][0] = float(f1[i+3+j].strip('\n').split()[4])
          self.coord[j][1] = float(f1[i+3+j].strip('\n').split()[5])
          self.coord[j][2] = float(f1[i+3+j].strip('\n').split()[6])
          self.atomnote.append(f1[i+3+j].strip('\n').split()[1])
      else:
        pass

    self.coord_bohr = np.zeros((self.num, 3))
    for i in range(self.num):
      for j in range(3):
        self.coord_bohr[i][j] = self.coord[i][j] * A2B

    self.dis = calc_dismatrix(self.coord_bohr)
    self.anglecos = calc_cosmatrix(self.coord)

    self.polADFprint = np.zeros((3, 3), dtype=np.complex_)
    polADFprint_real = np.zeros((3, 3))
    polADFprint_imag = np.zeros((3, 3))
    for i in range(1, len(f1)):
      if 'Energy     =' in f1[i]:
        self.incfreq = float(f1[i].strip('\n').split()[-2])
#      elif '---' in f1[i-1] and 'Polarizability tensor' in f1[i]:
      elif f1[i] == ' Polarizability tensor:\n':
        polADFprint_real[0][0] = float(f1[i+2].strip('\n').split()[0])
        polADFprint_real[0][1] = float(f1[i+2].strip('\n').split()[1])
        polADFprint_real[0][2] = float(f1[i+2].strip('\n').split()[2])
        polADFprint_real[1][0] = float(f1[i+3].strip('\n').split()[0])
        polADFprint_real[1][1] = float(f1[i+3].strip('\n').split()[1])
        polADFprint_real[1][2] = float(f1[i+3].strip('\n').split()[2])
        polADFprint_real[2][0] = float(f1[i+4].strip('\n').split()[0])
        polADFprint_real[2][1] = float(f1[i+4].strip('\n').split()[1])
        polADFprint_real[2][2] = float(f1[i+4].strip('\n').split()[2])
    for i in range(len(f1)):
      if 'Imaginary Polarizability tensor' in f1[i]:
        polADFprint_imag[0][0] = float(f1[i+2].strip('\n').split()[0])
        polADFprint_imag[0][1] = float(f1[i+2].strip('\n').split()[1])
        polADFprint_imag[0][2] = float(f1[i+2].strip('\n').split()[2])
        polADFprint_imag[1][0] = float(f1[i+3].strip('\n').split()[0])
        polADFprint_imag[1][1] = float(f1[i+3].strip('\n').split()[1])
        polADFprint_imag[1][2] = float(f1[i+3].strip('\n').split()[2])
        polADFprint_imag[2][0] = float(f1[i+4].strip('\n').split()[0])
        polADFprint_imag[2][1] = float(f1[i+4].strip('\n').split()[1])
        polADFprint_imag[2][2] = float(f1[i+4].strip('\n').split()[2])
    self.polADFprint = polADFprint_real + (1j) * polADFprint_imag


    ## collect hirshfeld partitioned polarizability ##
    ## a polarizability tensor has nine components, three column, called xyz here
    ## collect x column (with 6 components)
    self.pol_x = np.zeros((self.num, 6), dtype=np.complex_) 
    ## collect y column (with 6 components)
    self.pol_y = np.zeros((self.num, 6), dtype=np.complex_) 
    ## collect z column (with 6 components)
    self.pol_z = np.zeros((self.num, 6), dtype=np.complex_) 
    ## collect fragment charge
    ## 3 directions(x,y,z) and num fragments
    self.charge = np.zeros((3, self.num), dtype=np.complex_)
 
    # each fragment in each direction has two component, local and nonlocal
    # in total for each fragment, we need 6 numbers
    # these 6 numbers are loc_x, loc_y, loc_z, nonloc_x, nonloc_y, nonloc_z
    #                         x      y      z         x         y         z come from coord change
    static_flag = 1 ## assume the calculation is static
    for string in f1:
      if 'imaginary' in string.lower():
        static_flag = 0 ## the calculation is frequency dependent
      else:
        pass
    if static_flag == 1:
      for i in range(len(f1)):
        if 'Dipmat_x' in f1[i]: ## when induced charge in x direction
          jj = 0
          while True:
            jj = jj + 1
            if 'Hirshfeld fragment: 1' in f1[i+jj]:
              break
          for j in range(self.num):
            self.pol_x[j][0] = float(f1[9*j+i+jj+1].strip('\n').split()[-1]) + 0 * (1j) 
            self.pol_x[j][1] = float(f1[9*j+i+jj+2].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_x[j][2] = float(f1[9*j+i+jj+3].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_x[j][3] = float(f1[9*j+i+jj+5].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_x[j][4] = float(f1[9*j+i+jj+6].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_x[j][5] = float(f1[9*j+i+jj+7].strip('\n').split()[-1]) + 0 * (1j)
            self.charge[0][j] = float(f1[9*j+i+jj+4].strip('\n').split()[-1]) + 0 * (1j)
        elif 'Dipmat_y' in f1[i]:  ## when induced charge in y direction
          jj = 0
          while True:
            jj = jj + 1
            if 'Hirshfeld fragment: 1' in f1[i+jj]:
              break
          for j in range(self.num):
            self.pol_y[j][0] = float(f1[9*j+i+jj+1].strip('\n').split()[-1]) + 0 * (1j) 
            self.pol_y[j][1] = float(f1[9*j+i+jj+2].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_y[j][2] = float(f1[9*j+i+jj+3].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_y[j][3] = float(f1[9*j+i+jj+5].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_y[j][4] = float(f1[9*j+i+jj+6].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_y[j][5] = float(f1[9*j+i+jj+7].strip('\n').split()[-1]) + 0 * (1j)
            self.charge[1][j] = float(f1[9*j+i+jj+4].strip('\n').split()[-1]) + 0 * (1j)
        elif 'Dipmat_z' in f1[i]: ## when induced charge in z direction
          jj = 0
          while True:
            jj = jj + 1
            if 'Hirshfeld fragment: 1' in f1[i+jj]:
      	      break
          for j in range(self.num):
            self.pol_z[j][0] = float(f1[9*j+i+jj+1].strip('\n').split()[-1]) + 0 * (1j) 
            self.pol_z[j][1] = float(f1[9*j+i+jj+2].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_z[j][2] = float(f1[9*j+i+jj+3].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_z[j][3] = float(f1[9*j+i+jj+5].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_z[j][4] = float(f1[9*j+i+jj+6].strip('\n').split()[-1]) + 0 * (1j)
            self.pol_z[j][5] = float(f1[9*j+i+jj+7].strip('\n').split()[-1]) + 0 * (1j)
            self.charge[2][j] = float(f1[9*j+i+jj+4].strip('\n').split()[-1]) + 0 * (1j)
    elif static_flag == 0:
      for i in range(len(f1)):
        if 'Dipmat_x' in f1[i]: ## when induced charge in x direction
          jj = 0
          while True:
            jj = jj + 1
            if 'Hirshfeld fragment: 1' in f1[i+jj]:
              break
          jj = jj + 1
          for j in range(self.num):
            self.pol_x[j][0] = float(f1[10*j+i+jj+1].strip('\n').split()[-2]) + float(f1[10*j+i+jj+1].strip('\n').split()[-1]) * (1j)
            self.pol_x[j][1] = float(f1[10*j+i+jj+2].strip('\n').split()[-2]) + float(f1[10*j+i+jj+2].strip('\n').split()[-1]) * (1j)
            self.pol_x[j][2] = float(f1[10*j+i+jj+3].strip('\n').split()[-2]) + float(f1[10*j+i+jj+3].strip('\n').split()[-1]) * (1j)
            self.pol_x[j][3] = float(f1[10*j+i+jj+5].strip('\n').split()[-2]) + float(f1[10*j+i+jj+5].strip('\n').split()[-1]) * (1j)
            self.pol_x[j][4] = float(f1[10*j+i+jj+6].strip('\n').split()[-2]) + float(f1[10*j+i+jj+6].strip('\n').split()[-1]) * (1j)
            self.pol_x[j][5] = float(f1[10*j+i+jj+7].strip('\n').split()[-2]) + float(f1[10*j+i+jj+7].strip('\n').split()[-1]) * (1j)
            self.charge[0][j] = float(f1[10*j+i+jj+4].strip('\n').split()[-2]) + float(f1[10*j+i+jj+4].strip('\n').split()[-1]) * (1j)
        elif 'Dipmat_y' in f1[i]:  ## when induced charge in y direction
          jj = 0
          while True:
            jj = jj + 1
            if 'Hirshfeld fragment: 1' in f1[i+jj]:
              break
          jj = jj + 1
          for j in range(self.num):
            self.pol_y[j][0] = float(f1[10*j+i+jj+1].strip('\n').split()[-2]) + float(f1[10*j+i+jj+1].strip('\n').split()[-1]) * (1j)
            self.pol_y[j][1] = float(f1[10*j+i+jj+2].strip('\n').split()[-2]) + float(f1[10*j+i+jj+2].strip('\n').split()[-1]) * (1j)
            self.pol_y[j][2] = float(f1[10*j+i+jj+3].strip('\n').split()[-2]) + float(f1[10*j+i+jj+3].strip('\n').split()[-1]) * (1j)
            self.pol_y[j][3] = float(f1[10*j+i+jj+5].strip('\n').split()[-2]) + float(f1[10*j+i+jj+5].strip('\n').split()[-1]) * (1j) 
            if len(f1[10*j+i+jj+6].strip('\n').split()) >= 4: 
              self.pol_y[j][4] = float(f1[10*j+i+jj+6].strip('\n').split()[-2]) + float(f1[10*j+i+jj+6].strip('\n').split()[-1]) * (1j)
            elif len(f1[10*j+i+jj+6].strip('\n').split()) == 3:
              self.pol_y[j][4] = float(f1[10*j+i+jj+6].strip('\n').split()[-1].split('.')[0]) + float(f1[10*j+i+jj+6].strip('\n').split()[-1].split('.')[1][0:5])*0.00001 + ( float(f1[10*j+i+jj+6].strip('\n').split()[-1].split('.')[1][5:]) + float(f1[10*j+i+jj+6].strip('\n').split()[-1].split('.')[2])*0.00001 ) * (1j)
            else:
              print('err XXX')
            self.pol_y[j][5] = float(f1[10*j+i+jj+7].strip('\n').split()[-2]) + float(f1[10*j+i+jj+7].strip('\n').split()[-1]) * (1j)
            self.charge[1][j] = float(f1[10*j+i+jj+4].strip('\n').split()[-2]) + float(f1[10*j+i+jj+4].strip('\n').split()[-1]) * (1j)
        elif 'Dipmat_z' in f1[i]: ## when induced charge in z direction
          jj = 0
          while True:
            jj = jj + 1
            if 'Hirshfeld fragment: 1' in f1[i+jj]:
              break
          jj = jj + 1
          for j in range(self.num):
            self.pol_z[j][0] = float(f1[10*j+i+jj+1].strip('\n').split()[-2]) + float(f1[10*j+i+jj+1].strip('\n').split()[-1]) * (1j) 
            self.pol_z[j][1] = float(f1[10*j+i+jj+2].strip('\n').split()[-2]) + float(f1[10*j+i+jj+2].strip('\n').split()[-1]) * (1j)
            self.pol_z[j][2] = float(f1[10*j+i+jj+3].strip('\n').split()[-2]) + float(f1[10*j+i+jj+3].strip('\n').split()[-1]) * (1j)
            self.pol_z[j][3] = float(f1[10*j+i+jj+5].strip('\n').split()[-2]) + float(f1[10*j+i+jj+5].strip('\n').split()[-1]) * (1j)
            self.pol_z[j][4] = float(f1[10*j+i+jj+6].strip('\n').split()[-2]) + float(f1[10*j+i+jj+6].strip('\n').split()[-1]) * (1j)
            self.pol_z[j][5] = float(f1[10*j+i+jj+7].strip('\n').split()[-2]) + float(f1[10*j+i+jj+7].strip('\n').split()[-1]) * (1j)
            self.charge[2][j] = float(f1[10*j+i+jj+4].strip('\n').split()[-2]) + float(f1[10*j+i+jj+4].strip('\n').split()[-1]) * (1j)
    else:
      print('static_flag is wrong')

    self.poltotal = np.zeros((3, 3), dtype=np.complex_)
    self.pollocal = np.zeros((3, 3), dtype=np.complex_)
    for i in range(3):
      for j in range(self.num):
        self.poltotal[0][i] += self.pol_x[j][0+i] + self.pol_x[j][3+i]
        self.pollocal[0][i] += self.pol_x[j][0+i]
    for i in range(3):
      for j in range(self.num):
        self.poltotal[1][i] += self.pol_y[j][0+i] + self.pol_y[j][3+i]
        self.pollocal[1][i] += self.pol_y[j][0+i]
    for i in range(3):
      for j in range(self.num):
        self.poltotal[2][i] += self.pol_z[j][0+i] + self.pol_z[j][3+i]
        self.pollocal[2][i] += self.pol_z[j][0+i]



  def calc_polbond(self, p1, p2, component='all'):

    self.polatom = {}
    self.polbond = {}

    for i in range(len(self.atomnote)):
      if component == 'xx':
        self.polatom.update( { i: self.pol_x[i][0]})
      elif component == 'yy':
        self.polatom.update( { i: self.pol_y[i][1]})
      elif component == 'zz':
        self.polatom.update( { i: self.pol_z[i][2]})
      elif component == 'all':
        self.polatom.update( { i: (1.0 / 3.0) * (self.pol_x[i][0] + self.pol_y[i][1] + self.pol_z[i][2]) } )
  

    self.Q = []
    self.Q.append(calc_chargeflow(p1=p1, p2=p2, atomnote=self.atomnote, charge=self.charge[0], dis=self.dis, anglecos=self.anglecos[0])) 
    self.Q.append(calc_chargeflow(p1=p1, p2=p2, atomnote=self.atomnote, charge=self.charge[1], dis=self.dis, anglecos=self.anglecos[1])) 
    self.Q.append(calc_chargeflow(p1=p1, p2=p2, atomnote=self.atomnote, charge=self.charge[2], dis=self.dis, anglecos=self.anglecos[2])) 


    for i in range(len(self.atomnote)):
      for j in range(i):
        if component == 'xx':
          self.polbond.update( { (i, j) : self.Q[0][i][j] * ( self.coord_bohr[i][0] - self.coord_bohr[j][0] )  })
        if component == 'yy':
          self.polbond.update( { (i, j) : self.Q[1][i][j] * ( self.coord_bohr[i][1] - self.coord_bohr[j][1] )  })
        if component == 'zz':
          self.polbond.update( { (i, j) : self.Q[2][i][j] * ( self.coord_bohr[i][2] - self.coord_bohr[j][2] )  })
        if component == 'all':
          self.polbond.update( { (i, j) : (1.0 / 3.0) * ( self.Q[0][i][j] * ( self.coord_bohr[i][0] - self.coord_bohr[j][0] ) +  self.Q[1][i][j] * ( self.coord_bohr[i][1] - self.coord_bohr[j][1] ) + self.Q[2][i][j] * ( self.coord_bohr[i][2] - self.coord_bohr[j][2] )  )     })
  

#  def calc_pol3compo(self, mollist, clulist):
#    mol = 0.0
#    inter = 0.0
#    clu = 0.0
#    for k in range(len(self.atomnote)):
#      if self.atomnote[k] in clulist:
#        clu += self.polatom[k]
#      elif self.atomnote[k] in mollist:
#        mol += self.polatom[k]
#
#    for k in range(len(self.atomnote)):
#      for l in range(k):
#        if self.atomnote[k] in clulist and self.atomnote[l] in clulist:
#          clu += self.polbond[(k, l)]
#        elif self.atomnote[k] in clulist and self.atomnote[l] in mollist:
#          inter += self.polbond[(k, l)]
#        elif self.atomnote[l] in clulist and self.atomnote[k] in mollist:
#          inter += self.polbond[(k, l)]
#        elif self.atomnote[k] in mollist and self.atomnote[l] in mollist:
#          mol += self.polbond[(k, l)]
#
#    self.pol3compo = (mol, inter, clu)

#  def calc_Q3compo(self, mollist, clulist):
#    mol = 0.0
#    inter = 0.0
#    clu = 0.0
#
#    for k in range(len(self.atomnote)):
#      for l in range(k):
#        if self.atomnote[k] in clulist and self.atomnote[l] in clulist:
#          clu += self.Q[2][k, l]
#        elif self.atomnote[k] in clulist and self.atomnote[l] in mollist:
#          inter += self.Q[2][k, l]
#        elif self.atomnote[l] in clulist and self.atomnote[k] in mollist:
#          inter += self.Q[2][k, l]
#        elif self.atomnote[k] in mollist and self.atomnote[l] in mollist:
#          mol += self.Q[2][k, l]
#    self.Q3compo = (mol, inter, clu)
