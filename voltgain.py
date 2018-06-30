#Forked from gaincheck.py on 20180420. Computes a volt gain.

import numpy as np
import pyfits
import matplotlib.pyplot as plt
import pdb

def main(save=False, set='blah'):
    if set=='blah':
        print "Please choose set='before' or set='after'"
    if set =='before':
        dir = 'voltgain_20180419/'
        im_names=['vg_prv=3.5633-8-0.fits',
                  'vg_prv=3.5828-4-0.fits',
                  'vg_prv=3.6022-3-0.fits',
                  'vg_prv=3.6228-7-0.fits',
                  'vg_prv=3.5731-6-0.fits',
                  'vg_prv=3.5925-2-0.fits',
                  'vg_prv=3.6131-5-0.fits']
        filename = '../figures/SCExAO Volt Gain no Preamp'
        nchars = 6

    else:
        dir = 'voltgain_20180527/'
        im_names = ['vg_20180527_prv=3.560-2-0.fits',
                    'vg_20180527_prv=3.569-3-0.fits',
                    'vg_20180527_prv=3.579_2-1-0.fits',
                    'vg_20180527_prv=3.589-12-0.fits',
                    'vg_20180527_prv=3.598-11-0.fits',
                    'vg_20180527_prv=3.609-2-0.fits',
                    'vg_20180527_prv=3.6191-1-0.fits']
        filename = '../figures/SCExAO Volt Gain with Preamp'
        nchars = 5


    v = np.zeros(len(im_names))
    for i in range(len(im_names)):
        v[i] = float(im_names[i][im_names[i].find('=')+1 : 
                                 im_names[i].find('=')+nchars +1])

    adus= np.zeros(len(im_names))
    for i in range(len(im_names)):
        img = pyfits.getdata(dir+im_names[i])

        adus[i] = np.median(img[ -500 :])

        if 0: #check linearity
            meds = np.zeros(len(img))
            for z in range(len(img)):
                meds[z] = np.median(img[z])
            plt.plot(meds,'o')
            plt.plot([0, len(img)], [adus[i], adus[i]])
            plt.title(im_names[i])
            plt.ylim(np.min(meds), np.max(meds[2:]))
            plt.show()

    coeffs=np.polyfit(v, adus, 1)
    p = np.poly1d(coeffs)

    #plt.figure(1, figsize=(9, 6))
    plt.plot(v, adus, 'o')
    plt.plot(v, p(v), '-', label='y = '+str(coeffs[0])[:7]+' x + '+str(coeffs[1])[:7] )

    plt.suptitle('SCExAO SAPHIRA Voltage Gain, '+ \
                 str((coeffs[0]*1e-6)**-1)[:5]+r'$\mu$'+"V/ADU", \
                 fontsize=16)
    plt.title('Rev3 Pizza Box, High Gain, Mk14 SAPHIRA M06715-34 on ME1000', \
              fontsize=14)
    plt.legend(fontsize=14)
    plt.xlabel('PRV [V]', fontsize=14)
    plt.ylabel('ADUs', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    #plt.show()

    if save==True:
        print "Will save figure now with the following filename. c to continue."
        print filename+'.png'
        pdb.set_trace()
        plt.savefig(filename+'.png', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    np.savez(filename, filename, v, adus, coeffs, p(v))
    #pdb.set_trace()
