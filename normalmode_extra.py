import numpy as np
from math import ceil


class normalmode_oldADF(object):

  def __init__(self, freqout):

    f = open(freqout) ## freq output
    f1 = f.readlines()
    f.close()

    ## collect atomic masses ##
    mass = [] ## a list for atomic masses
    for i in range(len(f1)):
      if 'Atomic Masses' in f1[i]:
        ii = 2
        while True:
          if f1[i+ii] == '\n':
            break
          else:
            mass.append(float(f1[i+ii].strip('\n').split()[-1]))
            ii = ii + 1
      else:
        pass
    mass = np.array(mass)
    self.num = len(mass)
    self.rt_mass = np.sqrt(mass)
    self.rt_mass = np.diag(self.rt_mass)

    ## collect coordiantes ##
    self.coord = np.zeros((self.num, 3))
    self.atomnote = []

    for i in range(len(f1)):
      if 'FRAGMENTS' in f1[i]:
    	  for j in range(self.num):
    		  self.coord[j][0] = float(f1[i-2-self.num+j].strip('\n').split()[2])
    		  self.coord[j][1] = float(f1[i-2-self.num+j].strip('\n').split()[3])
    		  self.coord[j][2] = float(f1[i-2-self.num+j].strip('\n').split()[4])
    		  self.atomnote.append(f1[i-2-self.num+j].strip('\n').split()[1])


#    ## !!! cannot use coord from this block because the sequence of atoms is changed in this block
#    for i in range(len(f1)):
#      if 'FRAGMENTS' in f1[i]:
#    	  for j in range(self.num):
#    		  self.coord[j][0] = float(f1[i+3+j].strip('\n').split()[4])
#    		  self.coord[j][1] = float(f1[i+3+j].strip('\n').split()[5])
#    		  self.coord[j][2] = float(f1[i+3+j].strip('\n').split()[6])
#    		  self.atomnote.append(f1[i+3+j].strip('\n').split()[1])
#      else:
#        pass

    ## collect frequencies and  cartesian coordinate changes (not mass-weighted) ##
    self.freq = [] ## a list for frequencies
    for i in range(len(f1)):
      if 'Vibrations and Normal Modes' in f1[i]:
        ii = 7
        while True:
          if f1[ii+i] == '\n':
            break
          else:
            for iii in range(len(f1[i+ii].strip('\n').split())):
              self.freq.append(float(f1[i+ii].strip('\n').split()[iii]))
            ii = ii + 1 + self.num + 3
      else:
        pass
    self.freq = np.array(self.freq)
    self.num_freq = len(self.freq) ## the number of frequencies or normal modes
    self.normalmode = np.zeros((self.num_freq, self.num, 3))
    for i in range(len(f1)):
      if 'Vibrations and Normal Modes' in f1[i]:
        ii = 9
        j = 0
        while True:
          if 'List of All Frequencies' in f1[i+ii]:
            break
          else:
            if (len(f1[i+ii].strip('\n').split()) - 1) == 9:
              for jj in range(3):
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][0] = float(f1[i+ii+iii].strip('\n').split()[3*jj+1])
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][1] = float(f1[i+ii+iii].strip('\n').split()[3*jj+2])
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][2] = float(f1[i+ii+iii].strip('\n').split()[3*jj+3])
              j = j + 3
            elif (len(f1[i+ii].strip('\n').split()) - 1) == 6:
              for jj in range(2):
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][0] = float(f1[i+ii+iii].strip('\n').split()[3*jj+1])
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][1] = float(f1[i+ii+iii].strip('\n').split()[3*jj+2])
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][2] = float(f1[i+ii+iii].strip('\n').split()[3*jj+3])
              j = j + 2
  
            elif (len(f1[i+ii].strip('\n').split()) - 1) == 3:
              for jj in range(1):
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][0] = float(f1[i+ii+iii].strip('\n').split()[3*jj+1])
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][1] = float(f1[i+ii+iii].strip('\n').split()[3*jj+2])
                for iii in range(self.num):
                  self.normalmode[j+jj][iii][2] = float(f1[i+ii+iii].strip('\n').split()[3*jj+3])
              j = j + 1
            else:
              print('err about locating normal modes block')
          ii = ii + self.num + 4
      else:
        pass

    self.mw_normalmode = np.zeros((self.num_freq, self.num, 3)) ##  mass-weighted normal modes
    for i in range(self.num_freq):
      self.mw_normalmode[i] = np.matmul(self.rt_mass, self.normalmode[i])


class normalmode_nwchem():
  def __init__(self, filename):
    f = open(filename)
    f1 = f.readlines()
    f.close()

    self.atomnote = []
    self.coord = []
    self.mass = []
    self.freq = []
    for i in range(len(f1)):
      if 'Geometry' in f1[i]:
        ii = 7
        while True:
          if len(f1[i+ii]) < 3:
            break 
          else:
            self.atomnote.append(f1[i+ii].split()[1])
            self.coord.append(list(map(float, f1[i+ii].strip('\n').split()[3:])))
            ii += 1
      elif 'Atomic Mass' in f1[i]:
        ii = 3
        while True:    
          if len(f1[i+ii]) < 3:
            break
          else:
            self.mass.append(float(f1[i+ii].strip('\n').split()[-1]))
            ii += 1

    self.coord = np.array(self.coord)
    self.num = self.coord.shape[0]
    self.mass = np.array(self.mass)

    self.normalmode = np.zeros((self.num * 3, self.num, 3))

    for i in range(len(f1)):
      if 'Normal Eigenvalue' in f1[i] and 'Projected Derivative Dipole Momeent' in f1[i]:
        ii = 3
        while True:
          if '-----'  in len(f1[i+ii]):
            break
          else:
            self.freq.append(float(f1[i+ii].split()[1]))

      if 'NORMAL MODE EIGENVECTORS IN CARTESIAN COORDINATES' in f1[i] and 'Projected Frequencies' in f1[i+2]:
        ii = 8
        num_block = ceil(self.num * 3 / 6)
        for j in range(num_block):
          for k in range(len(f1[i+ii+j*(self.num*3 + 5)].split()) - 1):
            temp = []
            for l in range(self.num*3):
              temp.append(f1[i+ii+j*(self.num*3+5) + l].strip('\n').split()[k+1])
            self.normalmode[j * 6 + k] = np.array(temp).reshape(self.num, 3)
 
    self.freq = np.array(self.freq)






   




 








      


