import pyfits
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.interpolation
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import register_translation
import pdb
import os
import glob
import random
from matplotlib.colors import LogNorm

#Reads in the shifts produced in alignmulti. Figures out inter-cube shifts.
# Shifts images to align them. This is a minimalist version
# of tiptilt4.py.
#Run intershifts to get intermediate shifts. Then align(). 


def intershifts(dataset='blah'):
    if dataset=='blah':
        print "Please choose dataset='onsky' or dataset='internal'"
        return

    if dataset.lower()=='onsky':
        stub = 'pbimage_12*csv'
        dir = '/media/data/20170911/saphira/processed/'

    elif dataset.lower()=='internal':
        stub = 'pbimage_04*csv'
        dir = '/media/data/20180413/saphira/processed/'

    elif dataset.lower()=='internal2':
        stub = 'pbimage_03*csv'
        dir = '/media/data/20180413/saphira/processed/'

    elif dataset.lower()=='small':
	stub = '20170531*pbimage*csv'
        dir = '/media/data/20170531/saphira/processed/'

    elif dataset.lower()=='small2':
	stub = '20170815*pbimage*csv'
        dir = '/media/data/20170815/saphira/processed/'

    else:
        print "bad!"
        return

    ldir = 'align_offsets/'	
    filelist = sorted(glob.glob(ldir+stub))
    intermeds = np.zeros((2, len(filelist)-1))

    for i in range(len(filelist)-1):
        print str(int(round(float(i) / len(filelist)*100.))) + '% done.'
        c1 = pyfits.getdata(dir + fntofits(filelist[i])) #.replace('_offsets.csv', '.fits').replace(ldir, ''))
        c2 = pyfits.getdata(dir + fntofits(filelist[i+1])) #.replace('_offsets.csv', '.fits').replace(ldir, ''))

        intermeds[:,i] = register_translation(c1[-1], c2[0], 1000)[0]
        if (abs(intermeds[0,i]) > 0.5) or (abs(intermeds[1,i]) > 0.5):
            print intermeds[:,i]
            print dir + filelist[i].replace('_offsets.csv', '.fits')
            print dir + filelist[i+1].replace('_offsets.csv', '.fits')

    np.savez('align_offsets/intermedshifts_'+dataset+'.npz', filelist, intermeds, dir)

def align(dataset='blah'):
    junk = np.load('align_offsets/intermedshifts_'+dataset+'.npz')
    filelist = junk['arr_0']
    intermeds = junk['arr_1']
    dir = str(junk['arr_2']) #otherwise it is a np.ndarray

    x = os.system('echo 1 > '+dir+'delete.txt')
    if x != 0:
        print "You don't have permission to write to", dir
        return
    else:
        os.system('rm '+ dir + 'delete.txt')
        print "You have the permissions needed."

    allshifts = np.zeros((2,0))
    for i in range(len(filelist)):
        shifts = np.loadtxt(filelist[i], delimiter=',')

        allshifts = np.append(allshifts, shifts, 1)

        if i != np.shape(intermeds)[1]: #intermeds is 1 shorter than filelist
            allshifts = np.append(allshifts, np.reshape(intermeds[:,i], (2,1)), 1)

    if dataset=='onsky': #compensate for period that detector was blacked out
        allshifts[:, 86040 : 101000] = 0
        
    x_shift_arr_filtered = gaussian_filter(allshifts[1,:], 1)
    y_shift_arr_filtered = gaussian_filter(allshifts[0,:], 1)
    x_shift_arr_cum = np.cumsum(x_shift_arr_filtered)
    y_shift_arr_cum = np.cumsum(y_shift_arr_filtered)

    #doesn't work for huge datasets
    #coeffs_x = np.polyfit(range(len(x_shift_arr_cum)), x_shift_arr_cum, 3)
    #coeffs_y = np.polyfit(range(len(y_shift_arr_cum)), y_shift_arr_cum, 3)
    #fit_x = np.poly1d(coeffs_x)
    #fit_y = np.poly1d(coeffs_y)
    #driftfit_x = fit_x(range(len(x_shift_arr_cum)))
    #driftfit_y = fit_y(range(len(y_shift_arr_cum)))
    driftfit_x = gaussian_filter(x_shift_arr_cum, 2000)
    driftfit_y = gaussian_filter(y_shift_arr_cum, 2000)
        
    plt.figure(1, figsize=(10,10), dpi=100)
    plt.subplot(211)
    #plt.plot(shifts[0,:]-0.03, label='Unfiltered Y')
    #plt.plot(shifts[1,:]+0.03, label='Unfiltered X')
    #plt.plot(x_shift_arr_filtered + 0.03, label='Filtered X')
    #plt.plot(y_shift_arr_filtered - 0.03, label='Unfiltered Y')
    #plt.title("Shift from one frame to next")
    #plt.legend()

    plt.plot(x_shift_arr_cum, label='X shift')
    plt.plot(y_shift_arr_cum, label='Y shift')
    plt.plot(driftfit_x, label='X fit')
    plt.plot(driftfit_y, label='Y fit')
    plt.title('Cumulative Pixel Shift')
    plt.legend()

    x_shift_arr_cum -= driftfit_x
    y_shift_arr_cum -= driftfit_y

    plt.subplot(212)
    plt.plot(x_shift_arr_cum, label='X shift')
    plt.plot(y_shift_arr_cum, label='Y shift')
    plt.title('Drift-corrected Cumulative Pixel Shift')
    plt.legend()
    plt.show()

    shift_ind = 0
    for i in range(len(filelist)):
        img = pyfits.getdata(dir+fntofits(filelist[i])) #.replace('_offsets.csv', '.fits'))
        for z in range(len(img)):
            if shift_ind%500==0:
                print float(shift_ind) / (len(filelist) * len(img)) * 100., "% aligned."

            if (i==0) & (z==0): 
                continue #everything is shifted relative to the first image

            img[z] = scipy.ndimage.interpolation.shift(img[z], 
                                                       [1*y_shift_arr_cum[shift_ind],\
                                                        1*x_shift_arr_cum[shift_ind]],\
                                                       mode='wrap')
            shift_ind += 1

        if 1: #Save image?
            newfilename = fntofits(filelist[i]).replace('.fits', '_aligned_cc.fits' )
            
            if '12:16:30' in newfilename: #filter change during 20170531 data
                img = img[4000:]

            print "Saving image as "+dir + newfilename
            pyfits.writeto(dir + newfilename, img, clobber='true') #save file
            print "Saved."


def check(dataset='blah'):

    junk = np.load('align_offsets/intermedshifts_'+dataset+'.npz')
    filelist = junk['arr_0']
    intermeds = junk['arr_1']
    dir = str(junk['arr_2']) #otherwise it is a np.ndarray

    n_files = 3
    print "len filelist", len(filelist)

    for i in range(n_files):
        index = random.randint(0, len(filelist)-1)
        print "index:", index

        if i==0:
            cubeb = pyfits.getdata(dir+fntofits(filelist[index]))
            cubea = pyfits.getdata(dir+fntofits(filelist[index]).replace('.fits', '_aligned_cc.fits'))

        else:
            cubeb = np.append(cubeb, pyfits.getdata(dir+fntofits(filelist[index])) , 0)
            cubea = np.append(cubea, pyfits.getdata(dir+fntofits(filelist[index]).replace('.fits', '_aligned_cc.fits')) , 0)

    #flatten cube
    cubeb = np.mean(cubeb, 0)
    cubea = np.mean(cubea, 0)

    print "stddev before:", np.std(cubeb, ddof=1)
    print "stddev after:", np.std(cubea, ddof=1)
    print "max before", np.max(cubeb)
    print "max after", np.max(cubea)

    
    plt.figure(1, figsize=(10,5), dpi=100)
    plt.subplot(121)
    plt.imshow(cubeb, interpolation='none', vmin=10, vmax=65e3, norm=LogNorm())
    plt.title('before')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(cubea, interpolation='none', vmin=10, vmax=65e3, norm=LogNorm())
    plt.title('after')
    plt.colorbar()

    plt.show()


def fntofits(fn):
    fn = fn.replace('_offsets.csv', '.fits')
    fn = fn.replace('align_offsets/', '')
    fn = fn[fn.find('pbimage') : ] #strip off date at beginning
    return fn
