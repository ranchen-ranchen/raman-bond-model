from cmath import polar
from pymol import cmd
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
import argparse
import _pickle as cPickle

A2B = 1.8897261328856432

def calc_rgb(angle):
  if (-1*math.pi) <= angle <= math.pi:
    pass
  elif angle > math.pi:
    angle =  angle - 2*math.pi
  elif angle < -1*math.pi:
    angle = angle + 2*math.pi
  else:
    print('angle is not in the right range')

  p = (angle * 3.0) / math.pi
  if -3 <= p < -2:
    R = 1
    G = p + 3
    B = 0
  elif -2 <= p < -1:
    R = -1-p
    G = 1
    B = 0
  elif -1 <= p < 0:
    R = 0
    G = 1
    B = p + 1
  elif 0 <= p < 1:
    R = 0
    G = 1 - p
    B = 1
  elif 1 <= p < 2:
    R = p -1
    G = 0
    B = 1
  elif 2<=p <= 3:
    R = 1
    G = 0
    B = 3 - p

  return float(R), float(G), float(B)


def plotatombond(picklefile, numbond=70, scalefactor = 1):
  with open(picklefile, "rb") as input_file:
    data = cPickle.load(input_file)
#  data = pickle.load(open(picklefile, 'rb'))
  atomnote = data[0]
  coord = data[1]
  ramanatom = data[2]
  ramanbond = data[3]

  map_list = []
  for i in range(len(atomnote)):
    map_list.append('{0:}/{1:}'.format(i+1, atomnote[i]))
  print(map_list)

  ## adjust the sign of the total pol der ##
  bond_and_local = (sum(ramanatom.values()) + sum(ramanbond.values())).real
  if bond_and_local > 0:
    pass
  else:
    for ii in ramanbond:
      ramanbond[ii] = ramanbond[ii] * (-1)
    for ii in ramanatom:
      ramanatom[ii] = ramanatom[ii] * (-1)

#  for ii in ramanbond:
#    ramanbond[ii] = ramanbond[ii] * (-1)
#  for ii in ramanatom:
#    ramanatom[ii] = ramanatom[ii] * (-1)


  #########################################
  ## rank the ramanbond magnitude
  bond_magnitude = []
  for ii in ramanbond.values():
    bond_magnitude.append(polar(ii)[0])
  bond_magnitude.sort(reverse=True)
  if numbond < len(bond_magnitude):
    threshold = bond_magnitude[numbond-1]
  else:
    threshold = bond_magnitude[-1]

  ### plot bonds and atoms in pymol ##
  ## preparation ##
  preset.ball_and_stick(selection='all', mode=1)
  cmd.set('sphere_transparency', 0.1, 'all')
  for i in range(len(map_list)):
    cmd.set('sphere_scale',    '0.0000000001', selection=map_list[i])
  cmd.set_bond('stick_radius', '0.0000000001', 'all')
  R = 0.0
  G = 0.0
  B = 0.0

  jj = 0
  for ii in ramanbond:
    if polar(ramanbond[ii])[0] > threshold:
      aa = ii[0]
      bb = ii[1]
      atom_aa = map_list[aa]
      atom_bb = map_list[bb]
  
      dis = math.sqrt((coord[aa][0] - coord[bb][0])**2 + (coord[aa][1] - coord[bb][1])**2 + (coord[aa][2] - coord[bb][2])**2)
      r_bond = math.sqrt((abs(ramanbond[ii]/scalefactor) / (dis * math.pi * (A2B**3.0))))
  
      cmd.bond(atom_aa, atom_bb)
      cmd.select('bond'+str(jj+1), selection=atom_aa + ' '+atom_bb)
      cmd.set_bond('stick_radius', str(r_bond), 'bond'+str(jj+1))
  
      R, G, B = calc_rgb(polar(ramanbond[ii])[1])
  
      cmd.set_color('bondcolor'+str(jj+1), [R, G, B])
      cmd.set_bond('stick_color', 'bondcolor'+str(jj+1), 'bond'+str(jj+1))
      jj = jj + 1  
    else:
      pass


  R = 0.0
  G = 0.0
  B = 0.0
  
  jj = 0
  for ii in ramanatom:
    r_atom = ((abs(ramanatom[ii] / scalefactor )*3 / (4 * math.pi * (A2B**3.0)))**(0.3333333333333))

    cmd.set('sphere_scale', str(r_atom), selection=map_list[ii])

    R, G, B = calc_rgb(polar(ramanatom[ii])[1])

    cmd.set_color('atomcolor'+str(jj+1), [R, G, B])
    cmd.color('atomcolor'+str(jj+1), map_list[ii])
    jj = jj + 1

  cmd.set('ray_opaque_background', 'off')
  cmd.set('ray_shadow', 'off')

cmd.extend('plotatombond', plotatombond)

#		fig = plt.figure(figsize=(18, 13.5))
#		counter = range(50)
#		plt.rc('font', size=20)
#		plt.bar(counter[jj: 50], absrelative[jj:50], color=(0.2, 0.4, 0.6, 1.0))
#		plt.bar(counter[0: jj], absrelative[0:jj], color=(0.3, 0.4, 0.1, 1.0))
##		plt.hlines(y=threshold, xmin=0, xmax=50, color='g', linestyle='dotted')
#		plt.text(40, 0.4, 'first '+str(threshold)+' plotted')
#		plt.xlabel('Raman polarizability bonds, from largest to smallest')
#		plt.ylabel('Relative absolute values of bonds')
#		plt.show()
