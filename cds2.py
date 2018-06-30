#!/usr/bin/env python

#Inteded for RR cubes. Takes as input a filename and one frame numbers. 
# Displays a median-subtracted image.

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pdb
import sys
import time
   
args=sys.argv

if len(args) != 3: 
    print
    print "Usage: python cds.py myfilename.fits framenum "
    print "Displays a median_subtracted frame of the image."
    print
    #return
else:
    filename=str(args[1])
    f1 = int(args[2])
    #f2 = int(args[3])


#def cds(filename, f1, f2):
img = pyfits.getdata(filename)
img = img[-100 :] #exclude first half
avg = np.median(img, axis=0)
cds = img[f1] - avg
if np.median(cds) < 0: cds *= -1

print "stddev is:", np.std(cds, ddof=1)

mymin=np.sort(cds.flatten())[0.01*np.size(cds)]
mymax=np.sort(cds.flatten())[0.9999*np.size(cds)]

plt.imshow(cds, interpolation='none', vmin=mymin, vmax=mymax)
#plt.imshow(cds, interpolation='none', norm=LogNorm(), vmin=2000, vmax=mymax)
plt.title(filename + ' CDS ' + " - " + str(f1))
plt.colorbar()
plt.show()

stddevs=np.array([])
for i in range(np.shape(img)[0]):
    stddevs=np.append(stddevs, np.std(img[i]-avg, ddof=1) )

print "Average RFI is:", np.median(stddevs)
time.sleep(5)
