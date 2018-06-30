#updated 6/5/18
#Forked from speckelives4. Updated for RR mode. Subtracts
# adjacent images to see
# how long it takes for speckles to change by ___ fraction of core
# flux.

import pyfits
import matplotlib.pyplot as plt
import numpy as np
#import scipy.ndimage.interpolation
import pdb
from multiprocessing import Pool

#def speckles(fullfn='blah', save=False, plottitle=''):
def speckles(params):
    fullfn=params[0]
    save=params[1]
    plottitle=params[2]

    #if fullfn='blah', then it uses the filenames defined below.

    filename = 'pbimage_12:17:12.892986502_p_aligned.fits'
    #pbimage_12:31:14.268904945_p.fits' #dark
    #'pbimage_02:11:42.583337933_p.fits'
    #'pbimage_02:10:23.570269224_p_aligned.fits'
    #'pbimage_14:14:32.248706014_p_aligned.fits'
    #pbimage_10:24:35.852766640_p_aligned.fits'
    #pbimage_10:24:57.047762508_p_aligned.fits'
    #'pbimage_12:25:45.673185625_p_aligned.fits'
    #pbimage_12:20:51.778010423_p_aligned.fits
    
    dir = '/media/data/20170531/saphira/processed/'
    #'/media/data/20170902/saphira/processed/'
    #'/media/data/20170815/saphira/processed/'

    #

    #/media/data/20170815/saphira/processed/'
    #pbimage_14:14:32.248706014_p_aligned.fits'

    if fullfn=='blah':
        fullfn = dir+filename

    #voltage='-6' #determines which dark to use
    #darkmode='F' #Are we running on dark images or data images?
    aomode='T' #Does the data have AO corrections?
    framerate = 1680. #Hz

    print "Framerate is assumed to be", framerate, "Hz."
    #print "Voltage is assumed to be ", voltage
    if aomode != 'F':
        print "Working with AO on."
    else:
        print "Working with no AO."

    #0.0107''/pixel plate scale
    #lambda/D in units of pixels
    lambda_D_pix = 1.63e-6 / 8.2 * 206265. / .0107 
    lambda_D_min = 4 #inner radius of annulus
    lambda_D_max = 10 #inner radius of annulus

    img = pyfits.getdata(fullfn).astype('float32')#[:1000]

    #if the 95th percentile pixel is < 10*sddev of darker half of image, assume it's a dark frame
    if np.sort(img[0].flatten())[0.98*np.size(img[0])] < \
       10 * np.std(np.sort(img[0].flatten())[:0.5*np.size(img[0])], ddof=1):
        darkmode = 'T'
    else:
        darkmode = 'F'

    im_avg_orig = np.sum(img, 0) / len(img)

    if darkmode == 'F':
        print "Working on light data."

        #find center of image
        if aomode != 'F':
            im_avg = im_avg_orig - np.sort(im_avg_orig.flatten())[np.size(im_avg_orig)-30]
            im_avg[np.where(im_avg < 0)] = 0
        else:
            im_avg = im_avg_orig - np.median(im_avg_orig)
            im_avg[np.where(im_avg < 0)] = 0
        
        x0 = np.sum(np.sum(im_avg, 0)*range(np.shape(im_avg)[1])) / np.sum(im_avg)
        y0 = np.sum(np.sum(im_avg, 1)*range(np.shape(im_avg)[0])) / np.sum(im_avg)

        if not save:
            plt.imshow(im_avg_orig, interpolation='none')
            print x0, y0
            plt.plot([x0], [y0], 'x')

            theta = np.arange(0, 2.*np.pi, 0.01)
            x = np.cos(theta)
            y = np.sin(theta)
            plt.plot(lambda_D_min*lambda_D_pix*x + x0, lambda_D_min*lambda_D_pix*y + y0, 'r-')
            plt.plot(lambda_D_max*lambda_D_pix*x + x0, lambda_D_max*lambda_D_pix*y + y0, 'r-')

            plt.show()

    else: #if it is dark data
        print "Operating on dark data."
        x0 = np.shape(img)[2] / 2
        y0 = np.shape(img)[1] / 2

        im_avg = np.sum(img, 0) / len(img)
        #psf_core = np.max(im_avg) #used to scale flux later

    #scale the counts to fraction of PSF core. But the core is saturated,
    # so use the astrorid speckles
    if fullfn == '/media/data/20170531/saphira/processed/' + \
       'pbimage_12:25:45.673185625_p_aligned.fits': #no ao
        astrogrid_flux=np.max(im_avg_orig)
    elif fullfn == '/media/data/20170531/saphira/processed/' + \
         'pbimage_12:31:14.268904945_p.fits': #dark dataset
        astrogrid_flux=2844
    elif fullfn == '/media/data/20170902/saphira/processed/'+\
         'pbimage_02:10:23.570269224_p_aligned.fits': #internal source
        astrogrid_flux=1
    else:
        astrogrid_flux = photocal(img, x0, y0)

    if '20170531' in fullfn:
        contrast = 5.09e-3
    elif '20170813' in fullfn:
        contrast = 2.58e-3
    elif '20170815' in fullfn:
        contrast = 2.58e-3
    elif (('20170902' in fullfn) or 
          ('pbimage_12:25:45.673185625_p_aligned.fits' in fullfn)): 
        #not performed on this day or noao
        contrast = 1 
    else:
        print "you need to define the astrogrid speckle brightness."
        pdb.set_trace()
    psf_core = astrogrid_flux / contrast
    print "If not saturated, the PSF core would be", psf_core, "ADU."
    im_avg_orig /= psf_core
    img /= psf_core

    #calculate mask around center
    mask = np.zeros(np.shape(im_avg))
    mask[:,:] = False
    for x in range(np.shape(mask)[1]):
        for y in range(np.shape(mask)[0]):
            r = np.sqrt((x-x0)**2 + (y-y0)**2)
            #select annulus between 2 and 8 lambda/D
            if ((r < lambda_D_max*lambda_D_pix) & (r>lambda_D_min*lambda_D_pix)): 
                mask[y,x] = True

    print "Number of pixels in annulus:", np.sum(mask)
    if aomode=='F':
        mask[:,:12] = False

    maskloc = np.where(mask == True)

    med_1 = np.median(np.sort(im_avg_orig[maskloc].flatten()) \
                      [0.2*np.size(im_avg_orig[maskloc]) : 0.3*np.size(im_avg_orig[maskloc])] )
    med_2 = np.median(np.sort(im_avg_orig[maskloc].flatten()) \
                      [0.4*np.size(im_avg_orig[maskloc]) : 0.5*np.size(im_avg_orig[maskloc])] )
    med_3 = np.median(np.sort(im_avg_orig[maskloc].flatten()) \
                      [0.6*np.size(im_avg_orig[maskloc]) : 0.7*np.size(im_avg_orig[maskloc])] )
    med_4 = np.median(np.sort(im_avg_orig[maskloc].flatten()) \
                      [0.8*np.size(im_avg_orig[maskloc]) : 0.9*np.size(im_avg_orig[maskloc])] )

    separations = np.arange(260)+1
    #separations = np.append(np.arange(50)+1, np.arange(52,260,4))
    stddev_storage_1 = np.zeros(len(separations)) #20-30% brightness
    stddev_storage_2 = np.zeros(len(separations)) #40-50% brightness
    stddev_storage_3 = np.zeros(len(separations)) #60-70% brightness
    stddev_storage_4 = np.zeros(len(separations)) #80-90% brightness
    n_entries = np.zeros(len(separations)) #how many data points are in each stddev_storage bin?
 
    frames = range(np.shape(img)[0]-np.max(separations))
     
    #calculate stddev of pixels within mask for various frame deltas
    for i in frames:
        if i%(round(np.max(frames)/100.))==0: #update status
            print str(int(round(float(i)/np.max(frames)*100.)))+"% complete."
            
        sorted = np.sort((img[i][maskloc]).flatten())
        n_elements = len(sorted)

        darkloc_1 = np.where( (img[i][maskloc] > sorted[0.2 * n_elements]) &
                              (img[i][maskloc] < sorted[0.3 * n_elements]) )
        darkloc_2 = np.where( (img[i][maskloc] > sorted[0.4 * n_elements]) &
                              (img[i][maskloc] < sorted[0.5 * n_elements]) )
        darkloc_3 = np.where( (img[i][maskloc] > sorted[0.6 * n_elements]) &
                              (img[i][maskloc] < sorted[0.7 * n_elements]) )
        darkloc_4 = np.where( (img[i][maskloc] > sorted[0.8 * n_elements]) &
                              (img[i][maskloc] < sorted[0.9 * n_elements]) )

        #darkloc_2 = np.where(img[i][maskloc] <
        #                   np.sort(img[i][maskloc])[0.8*len(img[i][maskloc])])

        for j in range(len(separations)):
            sep = separations[j]
            
            ccds = img[i] - img[i+sep]
            stddev_storage_1[j] += np.std(ccds[maskloc][darkloc_1], ddof=1)
            stddev_storage_2[j] += np.std(ccds[maskloc][darkloc_2], ddof=1)
            stddev_storage_3[j] += np.std(ccds[maskloc][darkloc_3], ddof=1)
            stddev_storage_4[j] += np.std(ccds[maskloc][darkloc_4], ddof=1)

            n_entries[j] += 1

            if 0:#sep == 1:
                plt.figure(num=1, figsize=(10, 5), dpi=100) 
                plt.subplot(121)
                plt.imshow(ccds, interpolation='none', vmin=-15, vmax=15)
                plt.title((np.std(ccds[maskloc], ddof=1)))

                plt.subplot(122)
                plt.imshow(img[i] - img[i+sep+1], interpolation='none',
                           vmin=-15, vmax=15)
                plt.title((np.std((img[i] - img[i+sep+1])[maskloc], ddof=1)))
                plt.show()

    stddev_storage_1 /= n_entries
    stddev_storage_2 /= n_entries
    stddev_storage_3 /= n_entries
    stddev_storage_4 /= n_entries
    #stddev_storage_1 /= psf_core #rescale to fraction of PSF brightness  
    #stddev_storage_2 /= psf_core #rescale to fraction of PSF brightness
   
    time_axis = separations/framerate

    #if darkmode == 'T':
    #    np.savetxt('noise'+voltage+'.txt', [np.median(stddev_storage_1), 
    #                                        np.median(stddev_storage_2),
    #                                        np.median(stddev_storage_3),
    #                                        np.median(stddev_storage_4)])
    #    print "dark data saved as", 'noise'+voltage+'.txt'
    #else:
     #   print
     #   noise1, noise2, noise3, noise4 = np.loadtxt('noise'+voltage+'.txt')
     #   print "referencing dark data from: ", 'noise'+voltage+'.txt'

     #   #subtract noise in quadrature
     #   stddev_storage_1 = np.sqrt(stddev_storage_1**2 - noise1**2) 
     #   stddev_storage_2 = np.sqrt(stddev_storage_2**2 - noise2**2)
     #   stddev_storage_3 = np.sqrt(stddev_storage_3**2 - noise3**2)
     #   stddev_storage_4 = np.sqrt(stddev_storage_4**2 - noise4**2)

    #fit a polynomial to the variance data
    coeffs_1 = np.polyfit(time_axis[:4], stddev_storage_1[:4], 1) #linear fit
    coeffs_2 = np.polyfit(time_axis[:4], stddev_storage_2[:4], 1) #linear fit
    coeffs_3 = np.polyfit(time_axis[:4], stddev_storage_3[:4], 1) #linear fit
    coeffs_4 = np.polyfit(time_axis[:4], stddev_storage_4[:4], 1) #linear fit

    fit_1 = np.poly1d(coeffs_1)
    fit_2 = np.poly1d(coeffs_2)
    fit_3 = np.poly1d(coeffs_3)
    fit_4 = np.poly1d(coeffs_4)

    noisefit_1 = fit_1(time_axis)
    noisefit_2 = fit_2(time_axis)
    noisefit_3 = fit_3(time_axis)
    noisefit_4 = fit_4(time_axis)

    print "10-20% median:", med_1, "fit coefficients:", coeffs_1
    print "30-40% median:", med_2, "fit coefficients:", coeffs_2
    print "50-60% median:", med_3, "fit coefficients:", coeffs_3
    print "70-80% median:", med_4, "fit coefficients:", coeffs_4

    plt.figure(num=1, figsize=(12, 5), dpi=100) 
    plt.plot(time_axis, stddev_storage_1, 'o-', label='20-30% brightness pixels')
    #plt.plot(time_axis[:4], noisefit_1[:4])
    plt.plot(time_axis, stddev_storage_2, 'o-', label='40-50% brightness pixels')
    #plt.plot(time_axis[:4], noisefit_2[:4])
    plt.plot(time_axis, stddev_storage_3, 'o-', label='60-70% brightness pixels')
    #plt.plot(time_axis[:4], noisefit_3[:4])
    plt.plot(time_axis, stddev_storage_4, 'o-', label='80-90% brightness pixels')
    #plt.plot(time_axis[:4], noisefit_4[:4])
    plt.legend(loc=0, fontsize=16) #1 for upper right

    #plt.plot(time_axis[:10], stddev_storage_1[:10], 'go') #show which points are being fit to
    #plt.plot(time_axis[:10], stddev_storage_2[:10], 'mo') #show which points are being fit to

    #plt.plot(time_axis[:10], noisefit_1[:10], 'g-')
    #plt.plot(time_axis[:10], noisefit_2[:10], 'm-')
    
    if plottitle=='':
        #Title plot
        if ('12:15' in filename) or ('12:16' in filename) or ('12:17' in filename) or \
           ('10:24' in filename) or ('14:14' in filename):
            mytitle="AO188 + Extreme AO, "
        elif ('12:19' in filename) or ('12:20' in filename) or ('12:21' in filename):
            mytitle="AO188 Only, "
        elif ('12:25' in filename) or ('12:26' in filename):
            mytitle="No AO Correction, "
        elif darkmode != 'T':
            mytitle = raw_input("Enter a plot title. ")    

        if '20170531' in fullfn:
            mytitle+= '20170531, '
        elif '20170813' in fullfn:
            mytitle+= '20170813, '
        elif '20170815' in fullfn:
            mytitle+= '20170815, '

        if darkmode != 'T':
            mytitle += '50nm filter bandwidth'

        if darkmode == 'T':
            mytitle="Unilluminated Data"
    else:
        mytitle=plottitle

    plt.title(mytitle, fontsize=18)#+', Median PSF Core = '+str(int(round(psf_core)))+' ADU')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Seconds', fontsize=16)
    if '02:10:23.570269224_p_aligned' in fullfn:
        plt.ylabel('Standard deviaton (Arb. Units)', fontsize=16)
    else:
        plt.ylabel('(Standard deviation) / (PSF core intensity)', fontsize=16)

    #plt.ylim(0, np.max(stddev_storage_2) + 0.1*np.ptp(stddev_storage_2) )
    plt.xlim(0, 0.1)#np.max(separations)/framerate)
    
    #plt.text(time_axis[19] + .006, stddev_storage_1[19]+200,'10% dimmest pixels: y = ' + str(coeffs_1[0])[:6]+'x^2 + ' + str(coeffs_1[1])[:6] + 'x + ' + str(coeffs_1[2])[:4], color='g')
    #plt.text(time_axis[19] + .006, stddev_storage_2[19]+200, '80% dimmest pixels: y = ' + str(coeffs_2[0])[:6]+'x^2 + ' + str(coeffs_2[1])[:6] + 'x + ' + str(coeffs_2[2])[:4], color='m')
    #plt.text(time_axis[19] + .006, stddev_storage_1[19]+200,'10% dimmest pixels', color='g')
    #plt.text(time_axis[19] + .006, stddev_storage_2[19]+200,'80% dimmest pixels', color='m')

    if save:
        plt.savefig('../figures/sl_'+mytitle.replace(' ', '')+'.png', bbox_inches='tight', dpi=150)
        plt.clf()
        print "Wrote", '../figures/sl_'+mytitle.replace(' ', '')+'.png'
    else:
        plt.show()


def photocal(image, x0, y0):
    #prints brightness of speckles
    img = np.mean(image, 0) #collapse in z dimension
    
    sep = 36 #x or y separation in pixels of spots from center
    #specklelocs = np.zeros((2,4))
    brightnesses = np.zeros(4)
    #crop out astrogrid speckles
    for i in range(4):
        if i==0:
            #Q1
            crop = img[y0 + sep - 5 : y0 + sep + 5 ,
                       x0 + sep - 5 : x0 + sep + 5 ]
        elif i==1: 
            #Q2
            crop = img[y0 + sep - 5 : y0 + sep + 5 ,
                       x0 - sep - 5 : x0 - sep + 5 ]
        elif i==2:
            #Q3
            crop = img[y0 - sep - 5 : y0 - sep + 5 ,
                       x0 - sep - 5 : x0 - sep + 5 ]
        else:
            #Q4
            crop = img[y0 - sep - 5 : y0 - sep + 5 ,
                       x0 + sep - 5 : x0 + sep + 5 ]

        #Compute centroid to find exact location of speckle
        #crop -= np.sort(crop.flatten())[np.size(crop)-9]
        #crop[np.where(crop < 0)] = 0
        
        #x1 = np.sum(np.sum(crop, 0)*range(np.shape(crop)[1])) / np.sum(crop)
        #y1 = np.sum(np.sum(crop, 1)*range(np.shape(crop)[0])) / np.sum(crop)
        
        #specklelocs[0,i] = y1
        #specklelocs[1,i] = x1
        
        if 0:
            plt.imshow(crop, interpolation='none')
            plt.title("quadrant "+ str(i+1))
            plt.colorbar()
            plt.show()
        
        brightnesses[i] = np.max(crop)

    print "quadrants:", np.arange(4)+1 
    print "astrogrid brightnesses:", brightnesses
    avg_brightness = np.mean(brightnesses)
    print "mean astrogrid brightness", avg_brightness
    print
        
    return avg_brightness


def makeall():
    #20170531 exao
    speckles(fullfn='/media/data/20170531/saphira/processed/pbimage_12:17:12.892986502_p_aligned.fits', save=True, plottitle='AO188 + Extreme AO, 20170531')

    #20170531 ao188
    speckles(fullfn='/media/data/20170531/saphira/processed/pbimage_12:20:51.778010423_p_aligned.fits', save=True, plottitle='AO188 Only, 20170531')

    #20170531 noao
    speckles(fullfn='/media/data/20170531/saphira/processed/pbimage_12:25:45.673185625_p_aligned.fits', save=True, plottitle='No AO, 20170531')

    #20170531 dark
    speckles(fullfn='/media/data/20170531/saphira/processed/pbimage_12:31:14.268904945_p.fits', save=True, plottitle='Unilluminated Data, 20170531')


    #20170813
    speckles(fullfn='/media/data/20170813/saphira/processed/pbimage_10:24:35.852766640_p_aligned.fits', save=True, plottitle='AO188 + Extreme AO, 20170813')

    #20180815
    speckles(fullfn='/media/data/20170815/saphira/processed/pbimage_14:14:32.248706014_p_aligned.fits', save=True, plottitle='AO188 + Extreme AO, 20170815')

    #'pbimage_02:11:42.583337933_p.fits'
    #'pbimage_02:10:23.570269224_p_aligned.fits'
    #pbimage_10:24:57.047762508_p_aligned.fits'


def makeallfast():
              #20180531 exao
    params = [['/media/data/20170531/saphira/processed/pbimage_12:17:12.892986502_p_aligned.fits', \
               True, 'AO188 + Extreme AO, 20170531'],
              #20170531 ao188
              ['/media/data/20170531/saphira/processed/pbimage_12:20:51.778010423_p_aligned.fits', \
              True, 'AO188 Only, 20170531'],
              #20170531 noao
              ['/media/data/20170531/saphira/processed/pbimage_12:25:45.673185625_p_aligned.fits', \
              True, 'No AO, 20170531'],
             #20170531 dark
              ['/media/data/20170531/saphira/processed/pbimage_12:31:14.268904945_p.fits', \
              True, 'Unilluminated Data, 20170531'],
              ['/media/data/20170902/saphira/processed/pbimage_02:10:23.570269224_p_aligned.fits', \
              True, 'SCExAO Internal Source, 20170902'],
    #          #20170813
              ['/media/data/20170813/saphira/processed/pbimage_10:24:35.852766640_p_aligned.fits', \
               True, 'AO188 + Extreme AO, 20170813'],
    #          #20180815
              ['/media/data/20170815/saphira/processed/pbimage_14:14:32.248706014_p_aligned.fits', \
               True, 'AO188 + Extreme AO, 20170815']]

    #params=[set1, set2, set3, set4, set5, set6]

    #pdb.set_trace()

    pool = Pool(processes=len(params))
    pool.map(speckles, params)
