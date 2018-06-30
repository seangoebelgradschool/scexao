import matplotlib.pyplot as plt
import numpy as np
import pyfits
import pdb
import glob
from matplotlib.colors import LogNorm
#import pickle
import random
from scipy.optimize import curve_fit
from scipy import exp
import matplotlib
from multiprocessing import Pool
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.unicode'] = True

#like corrcoeffs, but doesn't read all the images into a cube. This uses less ram.

def main(restore=False, dataset='dataset', savefig=False, multithread=1, allseps=True):

    #check if user specified which dataset to use
    dataset = dataset.lower()
    if (dataset == 'dataset'):# & \
       #(dataset != 'onsky') & \
       #(dataset != 'small') & \
       #(dataset != 'small2') & \
       #(dataset != 'test'):
        print "dataset options: 'internal', 'onsky', 'small', 'small2', 'test'"
        print "Other options include:"
        print " restore=True or restore=False"
        print " savefig=True or savefig=False"
        print " allseps=True or allseps=False"
        return
        
    if dataset=='internal': 
        #suffix='_internal.npy'
        plttitle="Pearson's Correlation Coefficients for Saturated Internal Source"
        datapath='/media/data/20180413/saphira/processed/pbimage_04*aligned.fits'
    elif dataset=='internal2':
        plttitle="Pearson's Correlation Coefficients for Unsaturated Internal Source"
        datapath='/media/data/20180413/saphira/processed/pbimage_03*aligned_cc.fits'
    elif dataset=='onsky': 
        #suffix='_onsky.npy'
        plttitle="Pearson's Correlation Coefficients for On-Sky Data"
        datapath='/media/data/20170911/saphira/processed/pbimage*aligned.fits'
    elif dataset=='small': 
        plttitle="Pearson's Correlation Coefficients for 20170531 Data"
        datapath='/media/data/20170531/saphira/processed/pbimage*aligned_cc.fits'
    elif dataset=='small2': 
        plttitle="Pearson's Correlation Coefficients for 20170815 Data"
        datapath='/media/data/20170815/saphira/processed/pbimage*aligned_cc.fits'
    elif dataset=='test':
        plttitle = 'Should be 1 everywhere'
        datapath = 'sameframe.fits'
    else:
        print "BAD"
        return

    if restore != True:
        print "Calculating data, not restoring it."

        filelist = sorted(glob.glob(datapath))
        print "filelist:"
        print np.transpose(filelist)

        print "Identifying annulus."

        n_files_avg = 5 #number of files to average when finding the psf core

        for i in np.linspace(0, len(filelist)-1, n_files_avg):
            if i==0:
                cube = pyfits.getdata(filelist[int(i)])
            else:
                #cube += pyfits.getdata(filelist[int(round(i))]
		cube = np.append(cube, pyfits.getdata(filelist[int(round(i))]), 0)

        #cube /= n_files_avg
        avgcube = np.mean(cube, 0)
        im = np.copy(avgcube)

        n_brightest = 12
        threshold = np.sort(avgcube.flatten())[np.size(avgcube)-n_brightest] #29th brightest pixel
        avgcube -= threshold
        avgcube[avgcube<0] = 0

        x0 = np.sum(np.sum(avgcube, 0)*range(np.shape(avgcube)[1])) / \
            np.sum(avgcube)
        y0 = np.sum(np.sum(avgcube, 1)*range(np.shape(avgcube)[0])) / \
            np.sum(avgcube)

        #0.0107 is arcseconds per pixel
        lambda_D_pix = 1.63e-6 / 8.2 * 206265. / .0107 #lambda/D in pixels
        rmin = 3 * lambda_D_pix
        rmax = 10 * lambda_D_pix

        #make annulus
        annulus = np.zeros(np.shape(im))
        for y in range(np.shape(annulus)[0]):
            for x in range(np.shape(annulus)[1]):
                r2 = (x-x0)**2 + (y-y0)**2
                if (r2 > rmin**2) & (r2 < rmax**2):
                    annulus[y,x] = 1

        #check region of interest
        if 1:
            plt.imshow(im, interpolation='none', vmin=50, vmax=5e4, norm=LogNorm())
            plt.plot([x0], [y0], 'kx', markersize=15)

            theta = np.linspace(0, 2*np.pi, 100)
            plt.plot(x0 + rmin * np.cos(theta), y0 + rmin * np.sin(theta))
            plt.plot(x0 + rmax * np.cos(theta), y0 + rmax * np.sin(theta))

            plt.colorbar()
            plt.tight_layout()
            plt.show()

            plt.imshow(annulus, interpolation='none')
            plt.show()

        print "Done."
        print "Reading in data."
        loc = np.where(annulus == 1)
        global size_selection
        size_selection = np.shape(loc)[1] #number of pixels in annulus

        global separations
        if allseps: #calculate the correlatoin coefficient for a range of timescales?
            separations = np.append(range(1, 199), range(200, 1680*5, 8))
            separations = np.append(separations, range(1680*6, 1680*360, 1680))
        else: #only the first 0.15 second?
            separations = np.append(range(1, 199), range(200, 1680, 4))
        #save separations
        np.save('corr_coeff_separations_'+dataset+'.npy', separations)

        #np.append is really slow, so let's create a huge cube and then
        # delete the extra space at the end.
        #Assume no more than 1e4 frames per cube
        global img2, stds
        img2 = np.float32(np.zeros((len(filelist)*1e4 , size_selection)))
        stds = np.zeros(len(filelist)*1e4)

        z_index=0 #how many frames have I processed?
        for i in range(len(filelist)):
            img = pyfits.getdata(filelist[i])

            print ' '+ str(int(round(float(i)/len(filelist)*100.))) + '% complete'

            for z in range(len(img)):
                img2[z_index] = img[z][loc]
                img2[z_index] -= np.mean(img2[z_index])
                stds[z_index] = np.std(img2[z_index])

                z_index += 1

        #delete excess space
        img2 = img2[:z_index]
        stds = stds[:z_index]
        print "img2 type (should be float32):", type(img2[30,30])
        framenos = range(0, len(img2)-100, 1)
        random.shuffle(framenos) #so it samples randomly in time
        
        if multithread == 1:
            index='' #what code number is it?
            pcc([index, dataset, framenos])

        else:
            thelist = []
            for i in range(multithread):
                mylist = [i, dataset]
                mylist.append(framenos[int(round(float(len(framenos))/multithread*i)): 
                                       int(round(float(len(framenos))/multithread*(i+1.)))])
                thelist.append(mylist)
            pool = Pool(processes=multithread)
            pool.map(pcc, thelist)

            mergematrices(dataset) #combine the results of the multithreading

        flattenmatrix(dataset) #flatten the matrix and save it

    else: #restore data from text file

        flattened = np.load('corr_coeff_matrix_'+dataset+'_flat.npy')
        separations = np.load('corr_coeff_separations_'+dataset+'.npy')

        #plot stuff
        plt.figure(figsize=(8,8), dpi=100)
 
        for i in range(3):
            plt.subplot(3,1,i+1)
            plt.plot(separations/1680., flattened, '*')

            if i==0:
                plt.xlim(-1,360)
                if dataset=='onsky':
                    plt.ylim(0.155, 0.26)
                elif dataset=='internal':
                    plt.ylim(0.975, 0.990)
                plt.title(plttitle, fontsize=16)

                #perform linear fit
                loc = np.where((separations/1680 > 20) & (separations/1680 < 350))
                coeffs = np.polyfit((separations[loc])/1680., flattened[loc], 1)
                fit = np.poly1d(coeffs)
 
                if dataset != 'zinternal2':
                    plt.plot((separations[loc])/1680., fit(separations[loc]/1680), 'r-', 
                             label=r'$\rho (t)='+
                             str(np.float16(coeffs[0])).replace('e', r'\times 10^{') \
                             + '} t+' + \
                             str(coeffs[1])[:5] + '$')
                    plt.legend()
                    print "linear fit coeffs:", coeffs

            elif i==1:
                plt.xlim(-0.01,5)
                if dataset=='onsky':
                    plt.ylim(0.192,0.22)
                elif dataset=='internal':
                    plt.ylim(0.985, 0.9895)
                plt.ylabel("Pearson's Correlation Coefficient", fontsize=14)

                loc = np.where((separations/1680. > 0.01) & (separations/1680. < 5))
                popt,pcov = curve_fit(expdec, (separations[loc])/1680., flattened[loc], 
                                      p0=[0.006, 0.5, 0.5])

                if dataset != 'zinternal2':
                    plt.plot((separations[loc])/1680., 
                             expdec((separations[loc])/1680., popt[0], popt[1], popt[2]),
                             'r-',
                             label=r'$\rho (t) = ' + str(popt[0])[:7] + \
                             r'e^{-t/'+str(popt[1])[:5]+'} + '+str(popt[2])[:5]+'$')
                    plt.legend()

                    print "coeffs: Lambda, tau, rho_0:", popt
                
            else:
                plt.xlim(0,0.1)
                plt.xlabel('Seconds', fontsize=14)
                if dataset=='onsky':
                    plt.ylim(0.20,0.26)
                elif dataset=='internal':
                    plt.ylim(0.987, 0.991)

        plt.subplots_adjust(top=0.96, bottom=0.07, left=0.10, right=0.97, hspace=0.15)
        
        outtitle='corrcoeffs_'+dataset+'.png'
        if savefig:
            plt.savefig(outtitle, dpi=150, bbox_inches='tight')
            plt.clf()
            print "Wrote", outtitle
        else:
            plt.show()

def expdec(t,a,tau, rho_0):
    #return np.array(a*exp(-(x-x0)**2/(2*sigma**2))) + x_0
    #expon = -1. * tau / np.array(t)
    #expon[expon > 200] = 200
    #Dexpon[expon < -200] = -200
    return a*np.array(exp(-1. * np.array(t) / tau)) + rho_0

def pcc(params):
    #Compute's pearson's correlation coefficients.
    #has to be called by another main() because it needs a bunch of global 
    # variables defined there
    
    index = params[0]
    dataset = params[1]
    framenos = params[2]
    print "Index is:", index
    print "first few frames:", framenos[:20]
    print "Analyzing data now."
    if index!='':
        index = str(index)+'_'
    matrix = np.float16(np.zeros((len(framenos), len(separations)))) #i, sep
    matrix -= 99 #makes it easier to identify unfilled areas

    for no in range(len(framenos)):
        i = framenos[no]

        if no%100 == 0:
            print ' '+ str(round(float(no)/len(framenos)*100.*1000.)/1000.)+ "% done."

        for s in range(len(separations)):
            sep = separations[s]

            if i+sep >= len(img2): 
            #    print "skipping because", i+sep, len(img2)
            #    pdb.set_trace()
                continue #skip to next iteration

            if dataset=='onsky': 
                #exclude period in which saphira was blanked off
                if (i > (1e4*8+6000)) & (i < (1e4*10+900)):
                    continue
                if (i+sep > (1e4*8+6000)) & (i+sep < (1e4*10+900)):
                    continue

            #clear definition
            #rho = np.sum((sel1 - np.mean(sel1)) * (sel2 - np.mean(sel2)) ) / \
            #             (np.std(sel1) * np.std(sel2) * np.size(sel1))

            #efficient definition
            rho = np.sum(img2[i]  * img2[i+sep]) / \
                         (stds[i] * stds[i+sep] * size_selection)

            matrix[no, s] = rho

        #save matrix
        if (no % 10000 == 0) & (no > 0):
            print "Saving."
            print "matrix type (should be float16):", type(matrix[30,30])
            np.save('corr_coeff_matrix_'+index+dataset+'.npy', matrix)
            print "Saved."
            print

    print "Finished! Making final save."
    #print "matrix type:", type(matrix[30,30])
    #default is 64-bit. Let's cut the size in half.
    np.save('corr_coeff_matrix_'+index+dataset+'.npy', matrix)
    print "Saved."
    print


def mergematrices(dataset):
    matrixlist = glob.glob('corr_coeff_matrix_*_'+dataset+'.npy')
    for i in range(len(matrixlist)):
        print "reading matrix", i+1, "of", len(matrixlist)
        if i ==0:
            matrix = np.load(matrixlist[i])
        else:
            #partial = np.load(matrixlist[i])
            #loc = np.where(partial != -99)
            #matrix[loc] = partial[loc]
            matrix = np.append(matrix, np.load(matrixlist[i]), 0)

    print "shape matrix:", np.shape(matrix)
    print "Finished merging matrices! Making final save."
    np.save('corr_coeff_matrix_'+dataset+'.npy', np.float16(matrix))
    print "Saved."
    print


def flattenmatrix(dataset):
    print "Loading data to flatten the matrix..."
    matrix = np.load('corr_coeff_matrix_'+dataset+'.npy')
    separations = np.load('corr_coeff_separations_'+dataset+'.npy')
    print "Done."

    print "Removing empty data, flattening matrix."
    flattened = np.zeros(len(separations))
    for i in range(len(flattened)):
        #if matrix is float16 then the mean crashes for the internal data source
        flattened[i] = np.mean((matrix[:,i])[np.where(matrix[:,i] != -99)].astype('float32'))
    print "Done."

    flat_filename = 'corr_coeff_matrix_'+dataset+'_flat.npy'
    np.save(flat_filename, flattened)
    print "Saved ", flat_filename

def makeotherplot(dataset='blah', savefig=False):
    if dataset=='blah':
        print "Please choose dataset='onsky' or dataset='internal' or dataset='small'"
        return

    flattened = np.load('corr_coeff_matrix_'+dataset+'_flat.npy')
    separations = np.load('corr_coeff_separations_'+dataset+'.npy')

    #plot stuff
    #plt.figure(figsize=(8,8), dpi=100)

    for i in range(3):
        #plt.subplot(3,1,i+1)
        plt.plot(separations/1680., flattened, 'o')
        plt.xlabel('Seconds', fontsize=14)
        
 
        plt.xlim(0,0.1)
        if dataset=='onsky':
            plt.ylim(0.20,0.26)
        elif dataset=='internal':
            plt.ylim(0.987, 0.991)

    #plt.subplots_adjust(top=0.96, bottom=0.07, left=0.10, right=0.97, hspace=0.29)
        
    outtitle='corrcoeffs_'+dataset+'.png'
    if savefig:
        plt.savefig(outtitle, dpi=150, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()

def plotinternal(savefig=False):

    #plot interal sources next to each other
    plt.figure(figsize=(11,8), dpi=100)
    plt.suptitle("Pearson's Correlation Coefficients for Internal Source", fontsize=16)
 
    for j in range(2):
        if j == 1: 
            dataset='internal'
            plttitle="PSF Saturated"
        else: 
            dataset='internal2'
            plttitle="PSF Unsaturated"

        flattened = np.load('corr_coeff_matrix_'+dataset+'_flat.npy')
        separations = np.load('corr_coeff_separations_'+dataset+'.npy')

        for i in range(3):
            plt.subplot(3, 2, i*2 + j + 1)
            plt.plot(separations/1680., flattened)

            if i==0:
                plt.xlim(-1,360)
                if dataset=='internal':
                    plt.ylim(0.975, 0.990)
                plt.title(plttitle, fontsize=14)

                #perform linear fit
                loc = np.where((separations/1680 > 20) & (separations/1680 < 350))
                coeffs = np.polyfit((separations[loc])/1680., flattened[loc], 1)
                fit = np.poly1d(coeffs)
 
                if dataset != 'zinternal2':
                    plt.plot((separations[loc])/1680., fit(separations[loc]/1680), 'r-', 
                             label=r'$\rho (t)='+
                             str(np.float16(coeffs[0])).replace('e', r'\times 10^{') \
                             + '} t+' + \
                             str(coeffs[1])[:5] + '$')
                    plt.legend()
                    print "linear fit coeffs:", coeffs

            elif i==1:
                plt.xlim(-0.01,5)
                if dataset=='onsky':
                    plt.ylim(0.192,0.22)
                elif dataset=='internal':
                    plt.ylim(0.985, 0.9895)
                if j==0: plt.ylabel("Pearson's Correlation Coefficient", fontsize=14)

                loc = np.where((separations/1680. > 0.01) & (separations/1680. < 5))
                popt,pcov = curve_fit(expdec, (separations[loc])/1680., flattened[loc], 
                                      p0=[0.006, 0.5, 0.5])

                if dataset != 'zinternal2':
                    plt.plot((separations[loc])/1680., 
                             expdec((separations[loc])/1680., popt[0], popt[1], popt[2]),
                             'r-',
                             label=r'$\rho (t) = ' + str(popt[0])[:7] + \
                             r'e^{-t/'+str(popt[1])[:5]+'} + '+str(popt[2])[:5]+'$')
                    plt.legend()

                    print "coeffs: Lambda, tau, rho_0:", popt
                
            else:
                plt.xlim(0,0.1)
                #plt.xlabel('Seconds', fontsize=14)
                if dataset=='onsky':
                    plt.ylim(0.20,0.26)
                elif dataset=='internal':
                    plt.ylim(0.987, 0.991)

    plt.subplots_adjust(top=0.91, bottom=0.07, left=0.08, right=0.98, hspace=0.15, wspace=0.16)
    plt.text(-0.01, 0.9863, 'Seconds', ha='center', va='center', fontsize=14)

    outtitle='corrcoeffs_internal_combined.png'
    if savefig:
        plt.savefig(outtitle, dpi=150, bbox_inches='tight')
        plt.clf()
        print "Wrote", outtitle
    else:
        plt.show()
