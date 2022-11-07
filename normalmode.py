import numpy as np
import math
from decimal import *
from numpy.linalg import norm
from cmath import polar

A2B = 1.8897261328856432 ## constant converting angstrom to bohr


atomnote2mass = {
'H': 1.00797,
'C': 12.011,
'N': 14.0067,
'O': 15.9994,
'F': 18.998403,
'Mg': 24.305,
'Cl': 35.453,
'Br': 79.904,
'I' : 126.9045,
'S': 32.06,
'Ag': 107.868,
'Cu': 63.546,
'Au': 196.9665}

def mode_to_pymol(obj, mode, vectorwidth=0.1, scalevector=1.0, component='all', transparency=1.0, unitconvert=1.0):

  # Open filename for this mode
  fw = open('mode{0:.3f}.pymol'.format(mode), 'w')
  for i in range(len(obj.freq)):
    if abs(obj.freq[i] - mode) < 0.01:
      pymolmode = obj.normalmode[i] / ( A2B * norm(obj.normalmode[i] ))

  # Cycle over each atom
  for j in range(obj.coord.shape[0]):
    if component == 'all':
      print('cgo_modevec {0}{2: 11.7f},{3: 11.7f},{4: 11.7f}'
              '{1}, {0}{5: 11.7f},{6: 11.7f},{7: 11.7f}{1}, radius={8:4.2f}, transparency={9:2.2f}'.format(
              '[', ']', unitconvert*obj.coord[j][0], unitconvert*obj.coord[j][1], unitconvert*
              obj.coord[j][2], unitconvert*pymolmode[j][0]*scalevector,
              unitconvert*pymolmode[j][1]*scalevector, unitconvert*pymolmode[j][2]*scalevector,
              vectorwidth, transparency), file=fw)

    if component == 'x':
      print('cgo_modevec {0}{2: 11.7f},{3: 11.7f},{4: 11.7f}'
              '{1}, {0}{5: 11.7f},{6: 11.7f},{7: 11.7f}{1}, radius={8:4.2f}, transparency={9:2.2f}'.format(
              '[', ']', unitconvert*obj.coord[j][0], unitconvert*obj.coord[j][1], unitconvert*
              obj.coord[j][2], unitconvert*pymolmode[j][0]*scalevector,
              0.0, 0.0,
              vectorwidth, transparency), file=fw)
    if component == 'y':
      print('cgo_modevec {0}{2: 11.7f},{3: 11.7f},{4: 11.7f}'
              '{1}, {0}{5: 11.7f},{6: 11.7f},{7: 11.7f}{1}, radius={8:4.2f}, transparency={9:2.2f}'.format(
              '[', ']', unitconvert*obj.coord[j][0], unitconvert*obj.coord[j][1], unitconvert*
              obj.coord[j][2], 0.0, unitconvert*pymolmode[j][1]*scalevector,
              0.0,
              vectorwidth, transparency), file=fw)
    if component == 'z':
      print('cgo_modevec {0}{2: 11.7f},{3: 11.7f},{4: 11.7f}'
              '{1}, {0}{5: 11.7f},{6: 11.7f},{7: 11.7f}{1}, radius={8:4.2f}, transparency={9:2.2f}'.format(
              '[', ']', unitconvert*obj.coord[j][0], unitconvert*obj.coord[j][1], unitconvert*
              obj.coord[j][2], 0.0, 0.0, unitconvert*pymolmode[j][2]*scalevector,
              vectorwidth, transparency), file=fw)

  fw.close()


def calc_stepsize(obj, mode):
  for i in range(len(obj.freq)):
    if abs(mode - obj.freq[i]) < 0.01:
      return (norm(obj.mw_normalmode[i]) / norm(obj.normalmode[i]) * 0.01 ) 
'''cart_stepsize = 0.01 usually,  normalmodes due to digital precision are not strictly normalzied '''


def calc_displacedcoord(obj, mode):
  for i in range(len(obj.freq)):
    if abs(float(mode) - obj.freq[i]) < 0.01:
      displacedcoord_plus = obj.coord + (obj.normalmode[i] / ( A2B * norm(obj.normalmode[i] ))) * 0.01  ## convert from Bohr to Angstrom !
      displacedcoord_minus = obj.coord - (obj.normalmode[i] / ( A2B * norm(obj.normalmode[i]))) * 0.01 

  return displacedcoord_plus, displacedcoord_minus


def create_modefile(obj, mode, template):
  displacedcoord_plus, displacedcoord_minus = calc_displacedcoord(obj, mode)
  f = open(template)
  f1 = f.readlines()
  f.close()
  
#  g = open('mode'+'{0:.2f}'.format(mode)+'-p.run', 'w')
  g = open('mode'+'{0:.3f}'.format(mode)+'-p.run', 'w')
  for i in range(len(f1)):
    if 'atoms' in f1[i].strip('\n').lower():
      g.write(f1[i])
      for j in range(obj.num):
        g.write('{0:<3}    {1: .8f}    {2: .8f}    {3: .8f}\n'.format(obj.atomnote[j], displacedcoord_plus[j][0], displacedcoord_plus[j][1], displacedcoord_plus[j][2]))
    else:
      g.write(f1[i])
  g.close()

#  g = open('mode'+'{0:.2f}'.format(mode)+'-m.run', 'w')
  g = open('mode'+'{0:.3f}'.format(mode)+'-m.run', 'w')
  for i in range(len(f1)):
    if 'atoms' in f1[i].strip('\n').lower():
      g.write(f1[i])
      for j in range(obj.num):
        g.write('{0:<3}    {1: .8f}    {2: .8f}    {3: .8f}\n'.format(obj.atomnote[j], displacedcoord_minus[j][0], displacedcoord_minus[j][1], displacedcoord_minus[j][2]))
    else:
      g.write(f1[i])
  g.close()


def create_periodic_numdif_modefile(obj, mode, template, pfield=0.0514220674763, mfield=-0.0514220674763):
  displacedcoord_plus, displacedcoord_minus = calc_displacedcoord(obj, mode)

  f = open(template)
  f1 = f.readlines()
  f.close()
  
  g = open('mode'+'{0:.3f}'.format(mode)+'-vibpfieldp.run', 'w')
  for i in range(len(f1)):
    g.write(f1[i])
    if 'atoms' in f1[i].lower():
      for j in range(obj.num):
        g.write('{0:<3}    {1: .8f}    {2: .8f}    {3: .8f}\n'.format(obj.atomnote[j], displacedcoord_plus[j][0], displacedcoord_plus[j][1], displacedcoord_plus[j][2]))
    if 'electrostaticembedding' in f1[i].lower():
       g.write('ElectricField 0 0 {0:}\n'.format(pfield))
    if 'lattice' in f1[i].lower():
      for j in range(obj.vec.shape[0]):
        g.write('{0: .8f}    {1: .8f}    {2: .8f}\n'.format(obj.vec[j][0], obj.vec[j][1], obj.vec[j][2]))

  g.close()

  g = open('mode'+'{0:.3f}'.format(mode)+'-vibpfieldm.run', 'w')
  for i in range(len(f1)):
    g.write(f1[i])
    if 'atoms' in f1[i].lower():
      for j in range(obj.num):
        g.write('{0:<3}    {1: .8f}    {2: .8f}    {3: .8f}\n'.format(obj.atomnote[j], displacedcoord_plus[j][0], displacedcoord_plus[j][1], displacedcoord_plus[j][2]))
    if 'electrostaticembedding' in f1[i].lower():
       g.write('ElectricField 0 0 {0:}\n'.format(mfield))
#    if 'efield' in f1[i].lower():
#       g.write('unit a.u.\n')
#       g.write('ez {0: .4f}\n'.format(mfield))

    if 'lattice' in f1[i].lower():
      for j in range(obj.vec.shape[0]):
        g.write('{0: .8f}    {1: .8f}    {2: .8f}\n'.format(obj.vec[j][0], obj.vec[j][1], obj.vec[j][2]))
  g.close()

  
  g = open('mode'+'{0:.3f}'.format(mode)+'-vibmfieldp.run', 'w')
  for i in range(len(f1)):
    g.write(f1[i])
    if 'atoms' in f1[i].lower():
      for j in range(obj.num):
        g.write('{0:<3}    {1: .8f}    {2: .8f}    {3: .8f}\n'.format(obj.atomnote[j], displacedcoord_minus[j][0], displacedcoord_minus[j][1], displacedcoord_minus[j][2]))
    if 'electrostaticembedding' in f1[i].lower():
       g.write('ElectricField 0 0 {0:}\n'.format(pfield))
#    if 'efield' in f1[i].lower():
#       g.write('unit a.u.\n')
#       g.write('ez {0: .4f}\n'.format(pfield))

    if 'lattice' in f1[i].lower():
      for j in range(obj.vec.shape[0]):
        g.write('{0: .8f}    {1: .8f}    {2: .8f}\n'.format(obj.vec[j][0], obj.vec[j][1], obj.vec[j][2]))
  g.close()

  g = open('mode'+'{0:.3f}'.format(mode)+'-vibmfieldm.run', 'w')
  for i in range(len(f1)):
    g.write(f1[i])
    if 'atoms' in f1[i].lower():
      for j in range(obj.num):
        g.write('{0:<3}    {1: .8f}    {2: .8f}    {3: .8f}\n'.format(obj.atomnote[j], displacedcoord_minus[j][0], displacedcoord_minus[j][1], displacedcoord_minus[j][2]))
    if 'electrostaticembedding' in f1[i].lower():
       g.write('ElectricField 0 0 {0:}\n'.format(mfield))
#    if 'efield' in f1[i].lower():
#       g.write('unit a.u.\n')
#       g.write('ez {0: .4f}\n'.format(mfield))

    if 'lattice' in f1[i].lower():
      for j in range(obj.vec.shape[0]):
        g.write('{0: .8f}    {1: .8f}    {2: .8f}\n'.format(obj.vec[j][0], obj.vec[j][1], obj.vec[j][2]))
  g.close()


################
################

class normalmode(object):

  def __init__(self, freqout):

    f = open(freqout) ## freq output
    f1 = f.readlines()
    f.close()


    ## collect coordiantes ##
    self.coord = []
    self.atomnote = []
    self.freq = []
    for i in range(len(f1)):
      if 'Geometry' in f1[i] and '---' in f1[i-1] and '---' in f1[i+1]:
        ii = 0
        while True:
          if 'Index' in f1[i+ii] and 'Symbol' in f1[i+ii]:
            indicator = i + ii + 1
            break
          else:
            ii += 1

        while True:
          if len(f1[indicator]) < 3:
            break
          else:
            self.coord.append(list(map(float, f1[indicator].strip('\n').split()[2:5])))
            self.atomnote.append(f1[indicator].strip('\n').split()[1])
            indicator += 1

      elif 'Index   Frequency (cm-1)' in f1[i]:
        ii = 1
        while True:
          if len(f1[i+ii]) < 2:
            break
          self.freq.append(float(f1[i+ii].strip('\n').split()[1]))
          ii += 1

    self.freq = np.array(self.freq)
    self.coord = np.array(self.coord)

    self.num = self.coord.shape[0]
    self.num_freq = len(self.freq)

    self.mass = np.zeros(self.num)
    for i in range(len(self.atomnote)):
      self.mass[i] = atomnote2mass[self.atomnote[i]]
    self.rt_mass = np.diag(np.sqrt(self.mass))

    self.normalmode = np.zeros((len(self.freq), self.num, 3))
    ii = 0
    for i in range(len(f1)):
      if 'Displacements (x/y/z)' in f1[i]: ## not mass weighted, in Bohr
        for j in range(self.num):
          self.normalmode[ii][j][0] = float(f1[i+1+j].strip('\n').split()[-3])
          self.normalmode[ii][j][1] = float(f1[i+1+j].strip('\n').split()[-2])
          self.normalmode[ii][j][2] = float(f1[i+1+j].strip('\n').split()[-1])
        ii += 1

    self.mw_normalmode = np.zeros((self.num_freq, self.num, 3)) ##  mass-weighted normal modes
    for i in range(self.num_freq):
      self.mw_normalmode[i] = np.matmul(self.rt_mass, self.normalmode[i] / np.linalg.norm(self.normalmode[i]) )


class normalmode_mbh(object): ## mobile block hessian

  def __init__(self, freqout):

    f = open(freqout) ## freq output
    f1 = f.readlines()
    f.close()


    ## collect coordiantes ##
    self.coord = []
    self.atomnote = []
    self.freq = []
    for i in range(len(f1)):
      if 'Geometry' in f1[i] and '---' in f1[i-1] and '---' in f1[i+1]:
        ii = 5
        while True:
          self.coord.append(list(map(float, f1[i+ii].strip('\n').split()[2:5])))
          self.atomnote.append(f1[i+ii].strip('\n').split()[1])
          ii += 1
          if len(f1[i+ii]) < 3:
            break
      elif 'Block Normal Modes Frequencies' in f1[i]:
        ii = 3
        while True:
          if len(f1[i+ii]) < 3:
            break
          else:
            self.freq.append(float(f1[i+ii].strip('\n').split()[1]))
            ii += 1

    self.freq = np.array(self.freq)
    self.coord = np.array(self.coord)

    self.num = self.coord.shape[0]
    self.num_freq = len(self.freq)

    self.mass = np.zeros(self.num)
    for i in range(len(self.atomnote)):
      self.mass[i] = atomnote2mass[self.atomnote[i]]
    self.rt_mass = np.diag(np.sqrt(self.mass))

    self.normalmode = np.zeros((len(self.freq), self.num, 3))
    ii = 0

    for i in range(len(f1)):
      if 'Block Normal Modes (including rigid motions)' in f1[i]:
        indicator = i

    for i in range(indicator, len(f1)):
      if 'Index' in f1[i] and 'Frequency' in f1[i] and 'Intensity' in f1[i]: ## not mass weighted, in Bohr
        for j in range(self.num):
          self.normalmode[ii][j][0] = float(f1[i+1+j].strip('\n').split()[-3])
          self.normalmode[ii][j][1] = float(f1[i+1+j].strip('\n').split()[-2])
          self.normalmode[ii][j][2] = float(f1[i+1+j].strip('\n').split()[-1])
        ii += 1

    self.mw_normalmode = np.zeros((self.num_freq, self.num, 3)) ##  mass-weighted normal modes
    for i in range(self.num_freq):
      self.mw_normalmode[i] = np.matmul(self.rt_mass, self.normalmode[i] / np.linalg.norm(self.normalmode[i]) )

class normalmode_periodic():
  def __init__(self, filename):
    f = open(filename)
    f1 = f.readlines()
    f.close()

    self.vec = np.zeros((2, 3)) ## 2d periodic systems
    for i in range(len(f1)):
      if 'Lattice vectors' in f1[i]:
        for j in range(self.vec.shape[0]):
          self.vec[j][0] = float(f1[i+1+j].strip('\n').split()[-3])
          self.vec[j][1] = float(f1[i+1+j].strip('\n').split()[-2])
          self.vec[j][2] = float(f1[i+1+j].strip('\n').split()[-1])

    self.coord = []
    self.atomnote = []
    self.freq = []
    for i in range(1, len(f1)-1):
      if 'Geometry' in f1[i] and '---' in f1[i-1] and '---' in f1[i+1]:
        ii = 5
        while True:
          self.coord.append(list(map(float, f1[i+ii].strip('\n').split()[2:5])))
          self.atomnote.append(f1[i+ii].strip('\n').split()[1])
          ii += 1
          if len(f1[i+ii]) < 3:
            break
      if 'Index   Frequency (cm-1)' in f1[i]:
        ii = 1
        while True:
          if len(f1[i+ii]) < 2:
            break
          self.freq.append(float(f1[i+ii].strip('\n').split()[-1]))
          ii += 1
    self.coord = np.array(self.coord)
    self.num = self.coord.shape[0]
    self.normalmode = np.zeros((len(self.freq), self.num, 3))
    ii = 0
    for i in range(len(f1)):
      if 'Displacements (x/y/z)' in f1[i]: ## not mass weighted, in Bohr
        for j in range(self.num):
          self.normalmode[ii][j][0] = float(f1[i+1+j].strip('\n').split()[-3])
          self.normalmode[ii][j][1] = float(f1[i+1+j].strip('\n').split()[-2])
          self.normalmode[ii][j][2] = float(f1[i+1+j].strip('\n').split()[-1])
        ii += 1


    self.mass = np.zeros(self.num)
    for i in range(len(self.atomnote)):
      self.mass[i] = atomnote2mass[self.atomnote[i]]

    self.rt_mass = np.diag(np.sqrt(self.mass))
    self.mw_normalmode = np.zeros(self.normalmode.shape)

    for i in range(self.normalmode.shape[0]):
      self.mw_normalmode[i] = np.matmul(self.rt_mass, self.normalmode[i] / np.linalg.norm(self.normalmode[i]) )




