#Creates a plot showing how the saphira settles with time.

import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb

def makeplot():

    dir = '/home/scexao/pizzabox/SCExAO/pbserver.main/saphcamdarks/'

    files = [
        '20170909225111_320_256.fits',
        '20170909230014_320_256.fits',
        '20170909231028_320_256.fits',
        '20170909232037_320_256.fits',
        '20170909233051_320_256.fits',
        '20170909234102_320_256.fits',
        '20170909235114_320_256.fits',
        '20170910000125_320_256.fits',
        '20170910001136_320_256.fits',
        '20170910002147_320_256.fits',
        '20170910003159_320_256.fits',
        '20170910004211_320_256.fits',
        '20170910005222_320_256.fits',
        '20170910010234_320_256.fits',
        '20170910011244_320_256.fits',
        '20170910012257_320_256.fits',
        '20170910013306_320_256.fits',
        '20170910014319_320_256.fits',
        '20170910015328_320_256.fits',
        '20170910020342_320_256.fits',
        '20170910021353_320_256.fits',
        '20170910022404_320_256.fits',
        '20170910023414_320_256.fits',
        '20170910024425_320_256.fits',
        '20170910025434_320_256.fits',
        '20170910030443_320_256.fits',
        '20170910031456_320_256.fits',
        '20170910032509_320_256.fits',
        '20170910033521_320_256.fits',
        '20170910034529_320_256.fits',
        '20170910035539_320_256.fits',
        '20170910040550_320_256.fits',
        '20170910041602_320_256.fits',
        '20170910042613_320_256.fits',
        '20170910043622_320_256.fits',
        '20170910044631_320_256.fits',
        '20170910045641_320_256.fits',
        '20170910050652_320_256.fits']

    junk = '''
    files = [
        '20170909225111_320_256.fits',
        '20170909230014_320_256.fits',
        '20170909231028_320_256.fits',
        '20170909232037_320_256.fits',
        '20170909233051_320_256.fits',
        '20170909234102_320_256.fits',
        '20170909235114_320_256.fits',
        '20170910000125_320_256.fits',
        '20170910001136_320_256.fits',
        '20170910002147_320_256.fits',
        '20170910003159_320_256.fits',
        '20170910004211_320_256.fits',
        '20170910005222_320_256.fits',
        '20170910010234_320_256.fits',
        '20170910011244_320_256.fits',
        '20170910012257_320_256.fits',
        '20170910013306_320_256.fits',
        '20170910014319_320_256.fits',
        '20170910015328_320_256.fits',
        '20170910020342_320_256.fits',
        '20170910021353_320_256.fits',
        '20170910022404_320_256.fits',
        '20170910023414_320_256.fits',
        '20170910024425_320_256.fits',
        '20170910025434_320_256.fits',
        '20170910030443_320_256.fits',
        '20170910031456_320_256.fits',
        '20170910032509_320_256.fits',
        '20170910033521_320_256.fits',
        '20170910034529_320_256.fits',
        '20170910035539_320_256.fits',
        '20170910040550_320_256.fits',
        '20170910041602_320_256.fits',
        '20170910042613_320_256.fits',
        '20170910043622_320_256.fits',
        '20170910044631_320_256.fits',
        '20170910045641_320_256.fits',
        '20170910050652_320_256.fits',
        '20170910051705_320_256.fits',
        '20170910052716_320_256.fits',
        '20170910053728_320_256.fits',
        '20170910054739_320_256.fits',
        '20170910055748_320_256.fits',
        '20170910060759_320_256.fits',
        '20170910061809_320_256.fits',
        '20170910062821_320_256.fits',
        '20170910063830_320_256.fits',
        '20170910064842_320_256.fits',
        '20170910065854_320_256.fits',
        '20170910070905_320_256.fits',
        '20170910071918_320_256.fits',
        '20170910072929_320_256.fits',
        '20170910073939_320_256.fits',
        '20170910074953_320_256.fits',
        '20170910080003_320_256.fits',
        '20170910081014_320_256.fits',
        '20170910082025_320_256.fits',
        '20170910083037_320_256.fits',
        '20170910084050_320_256.fits',
        '20170910085059_320_256.fits',
        '20170910090110_320_256.fits',
        '20170910091120_320_256.fits',
        '20170910092131_320_256.fits',
        '20170910093143_320_256.fits',
        '20170910094153_320_256.fits',
        '20170910095205_320_256.fits',
        '20170910100219_320_256.fits',
        '20170910101231_320_256.fits',
        '20170910102242_320_256.fits',
        '20170910103253_320_256.fits',
        '20170910104303_320_256.fits',
        '20170910105314_320_256.fits',
        '20170910110327_320_256.fits',
        '20170910111338_320_256.fits',
        '20170910112349_320_256.fits',
        '20170910113400_320_256.fits',
        '20170910114409_320_256.fits',
        '20170910115420_320_256.fits',
        '20170910120432_320_256.fits']

    files = ['20170910121755_128_128.fits',
             '20170910122758_128_128.fits',
             '20170910123801_128_128.fits',
             '20170910124804_128_128.fits',
             '20170910125806_128_128.fits',
             '20170910130809_128_128.fits',
             '20170910131812_128_128.fits',
             '20170910132814_128_128.fits',
             '20170910133817_128_128.fits',
             '20170910134819_128_128.fits',
             '20170910135822_128_128.fits',
             '20170910140825_128_128.fits',
             '20170910141828_128_128.fits',
             '20170910142830_128_128.fits',
             '20170910143833_128_128.fits',
             '20170910144836_128_128.fits',
             '20170910145839_128_128.fits',
             '20170910150842_128_128.fits',
             '20170910151844_128_128.fits',
             '20170910152847_128_128.fits',
             '20170910153850_128_128.fits',
             '20170910154852_128_128.fits',
             '20170910155855_128_128.fits',
             '20170910160857_128_128.fits',
             '20170910161900_128_128.fits',
             '20170910162902_128_128.fits',
             '20170910163905_128_128.fits',
             '20170910164908_128_128.fits',
             '20170910165910_128_128.fits',
             '20170910170913_128_128.fits',
             '20170910171915_128_128.fits',
             '20170910172918_128_128.fits']
    '''

    xaxis = (np.arange(len(files)-1)*10.+10.)/60.

    stddevs = np.zeros(len(files)-1)

    for i in range(len(files)-1):
        #pdb.set_trace()
        print "Working on file", i+1, "of", len(files)-1
        cds = pyfits.getdata(dir+files[i+1])-pyfits.getdata(dir+files[i])

        stddevs[i] = np.std(cds, ddof=1)

        #if (i<10) | (i%20==0) | ( (stddevs[i] > 6.9) & (i>20) ):
        #if 1:
        if i==14:#5:
            mymax = np.sort(cds.flatten())[np.size(cds)*0.999]
            mymin = np.sort(cds.flatten())[np.size(cds)*0.001]
            #print vmin, vmax
            #pdb.set_trace()
            plt.imshow(cds, interpolation='none', vmin=mymin, vmax=mymax)
            plt.colorbar(shrink=0.8)
            title = files[i+1][8:10]  + ':' + \
                    files[i+1][10:12] + ':' + \
                    files[i+1][12:14] + ' - ' + \
                    files[i][8:10]  + ':' + \
                    files[i][10:12] + ':' + \
                    files[i][12:14] 
            title = 'Difference between frames collected 10 minutes \n apart after 2.5 hours of readouts'

            plt.title(title+ ', stddev=' + str(stddevs[i])[:4]+' ADU', \
                      fontsize=20)
            #plt.tight_layout()
            #plt.show()
            plt.savefig('../figures/settling_2.5hr.png', bbox_inches='tight')
            plt.close()
            
            #pyfits.writeto(title.replace(' ', '') + '.fits', cds, clobber=True)


    plt.plot(xaxis, stddevs, 'o')
    #indicate the two points used for images
    plt.plot(xaxis[14], stddevs[14], 'ro')
    plt.plot(xaxis[5], stddevs[5], 'ro')

    
    plt.title('SAPHIRA Settling', fontsize=20)
    plt.xlabel('Time since readout began (hours)', fontsize=18)
    plt.ylabel('Standard deviation (ADU) of difference \n between adjacent average frames', fontsize=18)
    plt.xlim((0,6))
    plt.show()

