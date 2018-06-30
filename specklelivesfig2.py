import pyfits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

#Generates the PSF comparison plot in the 2018 PASP speckle lives paper

def main(save=False):

    filename_aligned = 'pbimage_12:17:12.892986502_p_aligned.fits'
    filename_unaligned = 'pbimage_12:17:12.892986502_p.fits'
    dir = '/media/data/20170531/saphira/processed/'

    lambda_D_pix = 1.63e-6 / 8.2 * 206265. / .0107 
    lambda_D_min = 4 #inner radius of annulus
    lambda_D_max = 10 #inner radius of annulus

    img_u=pyfits.getdata(dir+filename_unaligned).astype('float32')#[:1000]
    img = pyfits.getdata(dir+filename_aligned).astype('float32')#[:1000]

    im_avg_orig = np.sum(img, 0) / len(img)
    im_avg_unaligned = np.sum(img_u, 0) / len(img_u)

    im_avg = im_avg_orig - np.median(im_avg_orig)
    im_avg[np.where(im_avg < 0)] = 0
        
    x0 = np.sum(np.sum(im_avg, 0)*range(np.shape(im_avg)[1])) / np.sum(im_avg)
    y0 = np.sum(np.sum(im_avg, 1)*range(np.shape(im_avg)[0])) / np.sum(im_avg)

    theta = np.arange(0, 2.*np.pi, 0.01)
    x = np.cos(theta)
    y = np.sin(theta)
    mymin=350
    mymax=65e3

    plt.figure(figsize=(15, 5), dpi=100)
    plt.suptitle("20170531 ExAO PSF and Analysis Region", fontsize=20)
###First Image
    plt.subplot(131)
    plt.imshow(im_avg_unaligned, interpolation='none', \
               norm=LogNorm(), vmin=mymin, vmax=mymax)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=16) 
    plt.xlim(0,127)
    plt.ylim(0,127)
    plt.title("Mean Cube (Unaligned)", fontsize=18)
    plt.tick_params(axis='both', labelsize=16)

###Second Image
    plt.subplot(132)
    plt.imshow(im_avg_orig, interpolation='none', \
               norm=LogNorm(), vmin=mymin, vmax=mymax)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=16) 
    plt.plot(lambda_D_min*lambda_D_pix*x + x0, lambda_D_min*lambda_D_pix*y + y0, \
             'r-', linewidth=2)
    plt.plot(lambda_D_max*lambda_D_pix*x + x0, lambda_D_max*lambda_D_pix*y + y0, \
             'r-', linewidth=2)
    plt.xlim(0,127)
    plt.ylim(0,127)
    plt.title("Mean Cube (Aligned)", fontsize=18)
    plt.tick_params(axis='both', labelsize=16)


###Third Image
    plt.subplot(133)
    plt.imshow(img[5000], interpolation='none', \
               norm=LogNorm(), vmin=mymin, vmax=mymax)
    cb = plt.colorbar(shrink=0.8)
    cb.ax.tick_params(labelsize=16) 
    plt.plot(lambda_D_min*lambda_D_pix*x + x0, lambda_D_min*lambda_D_pix*y + y0, \
             'r-', linewidth=2)
    plt.plot(lambda_D_max*lambda_D_pix*x + x0, lambda_D_max*lambda_D_pix*y + y0, \
             'r-', linewidth=2)
    plt.xlim(0,127)
    plt.ylim(0,127)
    plt.title("Example Single Frame", fontsize=18)
    plt.tick_params(axis='both', labelsize=16)

    plt.subplots_adjust(left=0.03, bottom=0.02, right=1.00, top=0.97, wspace=0.04)
    if save:
        mytitle = '../figures/sl_psfcomparison.png'
        plt.savefig(mytitle, bbox_inches='tight', dpi=150)
        plt.clf()
        print "Wrote", mytitle
    else:
        plt.show()
