import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb
import glob
from multiprocessing import Pool
from skimage.feature import register_translation
#import pdb
#import os.path

#Identifies tip/tilt using a cross-correlation. This is a minimalist version
# of tiptilt4.py. Only records offsets. Doesn't shift images. Use tiptilt4 for 
# that.

def getalignments(filelist):
    print
    print len(filelist)
    #print filelist
    print
    #return
    
    if type(filelist) == str: #if just a string has been given
        filelist = [filelist] #make it a list


    #all_shifts = np.zeros((2, 0))
    for j in range(len(filelist)):

        img = pyfits.getdata(filelist[j])
        shifts = np.zeros((2, len(img)-1))
    
        for i in range(len(img)-1):
            if i%(len(img)/100)==0:
                print (float(i) + j*len(img) ) / (len(img) * len(filelist)) * 100.,  "% analyzed."
            shifts[:,i] = register_translation(img[i], img[i+1], 1000)[0]
            
        #all_shifts = np.append(all_shifts, shifts, 1)
        
        dateloc = filelist[j].find('2017')
        date = filelist[j][dateloc:dateloc+8]
        
        filename = filelist[j][filelist[j].find('pbimage') : ]
        filename = filename.replace('.fits', '_offsets.csv')
        np.savetxt('align_offsets/' + date + '_' + filename, shifts, delimiter=",")


def main(dataset='blah'):
    n_threads = 10

    if dataset=='blah':
        print "Please specify dataset='internal' or dataset='onsky'"
        return
    if dataset=='internal':
        print "Using internal source data."
        datapath='/media/data/20180413/saphira/processed/pbimage_04*_p.fits'
    if dataset=='internal2':
	print "Using unsaturated internal source data."
	datapath='/media/data/20180413/saphira/processed/pbimage_03*_p.fits'
    elif dataset=='onsky':
        print "Using on-sky data."
        datapath='/media/data/20170911/saphira/processed/pbimage*fits'
    elif dataset=='custom':
        if 0: #fix bug
            filelist = ['/media/data/20170911/saphira/processed/pbimage_12:39:11.567950898_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:17.579298576_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:23.888036649_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:29.896397533_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:35.907139942_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:41.945508319_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:47.976047287_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:53.984434050_p.fits',
                        '/media/data/20170911/saphira/processed/pbimage_12:39:59.993963846_p.fits']
        if 0:
            filelist = ['/media/data/20170531/saphira/processed/pbimage_12:16:30.383522433_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:16:36.414117331_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:16:42.513128841_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:16:48.555151016_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:16:54.807961320_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:17:00.825902718_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:17:06.835488780_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:17:12.892986502_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:17:18.982505901_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:17:25.063482820_p.fits',
                        '/media/data/20170531/saphira/processed/pbimage_12:17:31.072491455_p.fits']
        if 1:
            filelist = ['/media/data/20170815/saphira/processed/pbimage_14:13:49.748328688_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:13:56.732148193_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:04.178947149_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:11.211186873_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:18.232345016_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:25.245495970_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:32.248706014_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:39.246021859_p.fits',
                        '/media/data/20170815/saphira/processed/pbimage_14:14:46.216407905_p.fits']

    else:
        print "Please specify dataset='internal' or dataset='onsky'"
        return

    if dataset.lower() != 'custom':
        filelist = sorted(glob.glob(datapath))


    thelist = []
    for i in range(n_threads):
        thelist.append(filelist[int(round(float(len(filelist)) / n_threads * i)) :
                                int(round(float(len(filelist)) / n_threads * (i+1.)))])

    #if np.size(thelist) != len(filelist):
    #    print "there's some sort of indexing error."
    #    pdb.set_trace()
    #    return

    pool = Pool(processes=n_threads)
    #results = 
    pool.map(getalignments, thelist)
