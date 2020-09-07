import numpy as np
import tables
import matplotlib.pyplot as plt
import time
import os
import warnings
import pickle
import sys
from sklearn.neighbors import KernelDensity


def gaus(m,s,a,x):
    return a*np.exp(-(x-m)**2 / (2.* s**2) )

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, kernel='gaussian', **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def get_min_max_der(chans,pdf5):
    # follow the procedure of gausspy to find maxima and generalize to minima 
    # https://github.com/gausspy/gausspy/blob/master/doc/GaussPy.pdf
    dydx = np.gradient(pdf5/pdf5.max())
    d2ydx = np.gradient(dydx)
    d3ydx = np.gradient(d2ydx)
    d4ydx = np.gradient(d3ydx)
    
    #----Find minima
    ## Find where first derivative changes sign
    asign = np.sign(dydx)
    indices_signchange = np.zeros_like(asign)
    signchange = np.where(asign[1:]-asign[:-1] !=0)[0]+1
    indices_signchange[signchange] = 1    
    # Find sign of second derivative
    a2sign = np.sign(d2ydx)
    # Minima are where dydx changes sign and d2ydx is positive
    minima_d1 = np.logical_and(indices_signchange, a2sign >0)
    #----

    ## Find where third derivative changes sign (more robust than looking for == 0)
    asign = np.sign(d3ydx)
    indices_signchange = np.zeros_like(asign)
    signchange = np.where(asign[1:]-asign[:-1] !=0)[0]+1
    indices_signchange[signchange] = 1
    
    # Find sign of second derivative
    a2sign = np.sign(d2ydx)
    # Find sign of 4rth derivative
    a4sign = np.sign(d4ydx)
    
    # Maxima of the second derivative are where d3ydx changes sign, d2ydx is positive and d4ydx is negative
    max_of_d2 = np.logical_and(indices_signchange, np.logical_and(a2sign > 0, a4sign < 0))
    
    # Maxima of the function
    preliminary_maxima = np.logical_and(indices_signchange, np.logical_and(a2sign < 0, a4sign > 0))

    # Sometimes it finds maxima at the edges of the domain. Remove these edge effects
    chans_from_edge_to_discard = 3
    maxima = np.logical_and(preliminary_maxima, np.logical_and(np.round(chans) < np.round(chans.max())-chans_from_edge_to_discard, \
                            np.round(chans) > np.round(chans.min())+chans_from_edge_to_discard))
    
    # Minima or points of max second derivative
    minima = np.logical_or(max_of_d2, minima_d1)

    # Now we don't want gaps in between the minima points (with no max in between them)
    # Loop over each element in the minimum points and keep only the true minima 
    # (not max of second derivative) if no max in between
    minlocations = np.where(minima == True)[0]
    
    for ii in range(len(minlocations)-1):
        # check if there is a max between this min location and the next
        if np.any(maxima[minlocations[ii]:minlocations[ii+1]]):
            continue
        # if this and next point don't have a max in between, then keep only the min 
        # (not the max of 2nd derivative type point)
        else:
            # is this a minimum (1st derivative check)?
            tokeep = minima_d1[minlocations[ii]]==True
            # if this is a max_of_d2 type point, then remove it from the minima list
            if tokeep == False:
                minima[minlocations[ii]] = False
            # now check the next point
            tokeep = minima_d1[minlocations[ii+1]]==True
            if tokeep == False:
                minima[minlocations[ii+1]] = False

    return minima,maxima


# Assign extent of each maximum (from previous min to next)

def neighboring_minima(minima, maxima, pdf,chans ):
    '''
    Find the left and right edge of each peak. 

    Input: minima - 1d boolean array of len same as chans. True at location 
                of a local minimum of the pdf or maximum of the pdf's second derivative
           maxima - 1d boolean array like minima, but True indicates location of
                a local maximum of the pdf
           pdf - 1d array (float) - the values of the PDF
           chans - 1d array (float) - the values of the velocity channels

    Output: rightborder - 1d boolean array like minima. True at location of a right edge of a peak
            leftborder  - 1d boolean array like minima. True at location of a left edge of a peak
            cleanmaxima - 1d boolean array like minima. True at location of a maximum that corresponds to a robust peak

    '''
    assert len(minima) == len(chans)
    assert len(maxima) == len(chans)
    assert len(minima) == len(pdf)

    # Just find the nearest min to the left and right of each peak
    # Normalize the pdf
    pdf = pdf/pdf.max()
    chans_minima = chans[minima]
    chans_maxima = chans[maxima]
    
    leftborder = np.zeros_like(chans_maxima) # array to store the left border of each maximum
    rightborder = np.zeros_like(chans_maxima) # array to store the right border of each maximum

    for ii,maxim in enumerate(chans_maxima):
        
        # find any zeros to either side of the peak
        zeros_left_from_this_peak = np.logical_and(chans<maxim, pdf < 1e-4)
        zeros_right_from_this_peak = np.logical_and(chans>maxim, pdf < 1e-4)
        
        # find minima towards either side of peak
        mins_left_from_this_peak = chans_minima[chans_minima < maxim]
        mins_right_from_this_peak = chans_minima[chans_minima > maxim]
        
        # Make sure channels are positive
        assert maxim>0
        
        # Find nearest zeros
        # If there are no zeros, make the nearest zero be nan
        if len(chans[zeros_left_from_this_peak]) == 0:
            nearest_zero_left = np.nan
        else:
            nearest_zero_left = chans[zeros_left_from_this_peak][np.argmin(maxim-chans[zeros_left_from_this_peak])]
        if len(chans[zeros_right_from_this_peak]) == 0:
            nearest_zero_right = np.nan
        else:
            nearest_zero_right = chans[zeros_right_from_this_peak][np.argmin(chans[zeros_right_from_this_peak]- maxim)]
            
        # Find nearest mins
        # If there are no mins, make nearest min nan
        if len(mins_left_from_this_peak) ==0:
            nearest_min_left = np.nan
        else:
            nearest_min_left = mins_left_from_this_peak[np.argmin(maxim - mins_left_from_this_peak)]
        if len(mins_right_from_this_peak) == 0:
            nearest_min_right = np.nan
        else:
            nearest_min_right = mins_right_from_this_peak[np.argmin(mins_right_from_this_peak - maxim)]
        
        # On either side, choose whichever is nearest: min or zero
        choices = np.array([nearest_min_left, nearest_zero_left])
        tochoose = np.argmin(maxim - choices)
        # if one is nan, pick the other
        if np.any(np.isnan(choices)):
            
            if len(choices[choices==choices])==1: # if at least one element is non-nan
                leftborder[ii] = choices[choices==choices][0]
            else:
                leftborder[ii] = chans[chans<maxim][np.argmin(pdf[chans<maxim])]

        else:
            leftborder[ii] = choices[tochoose]
        
        choices = np.array([nearest_min_right, nearest_zero_right])
        
        tochoose = np.argmin(choices - maxim)
        if np.any(np.isnan(choices)):
            if len(choices[choices==choices])==1: # if at least one element is non-nan
                rightborder[ii] = choices[choices==choices][0]
            else:# there are no zeros or mins to the right of this peak
                # just assign the value where the pdf is min to the right (not zero)
                rightborder[ii] = chans[chans>maxim][np.argmin(pdf[chans>maxim])]
        else:
            rightborder[ii] = choices[tochoose]
        
    # Make sure that no consecutive peaks are left without a border between them (zero or min)
    mask_max = np.ones_like(chans_maxima).astype(bool)
    cleanmaxima = maxima.copy()
    
    # Loop over maxima
    for ii in range(len(chans_maxima)-1):
        # If there are two consecutive maxima, we need to keep only one
        if not rightborder[ii] < chans_maxima[ii+1]:
            # Sometimes you can have two maxima identified without a min in between.
            # For this, assume both belong to the same underlying peak
            # Then mask the one where the pdf is lower
            pdfcheck = np.array([pdf[maxima][ii],pdf[maxima][ii+1]])
            max_to_mask = np.array([ii,ii+1])[np.argmin(pdfcheck)]
            
            # which index in the channel array does this correspond to?
            index = np.where(chans == chans[maxima][max_to_mask])[0][0]
            # This will not be a max any more
            cleanmaxima[index] = False
            mask_max[max_to_mask] = False
    # Delete right and left border of the bad maxima
    rightborder = rightborder[mask_max]
    leftborder = leftborder[mask_max]
            
    return rightborder, leftborder, cleanmaxima


def run_cloud_identification(SFILENAME, sdir, datafile, RMSCUT = 0, SIGMACUT = 15, BANDWIDTH =3, \
                             RETURN_ARRS=False, NCHANS = 933, REMOVE_NOISE = True):
    '''
    Identifies clouds as peaks of the PDF of Gaussian component velocities.
    Input:
        SFILENAME: name of file to save output (string)
        sdir: path to folder to save output (string)
        datafile: full path + name of file in which Gaussian components are saved groupped by big pixel (output of collect function)
        RMSCUT: Gaussian components with peak amplitude less than this value will be discarded (float, unit: K) 
        SIGMACUT: GCs with sigma less than this value will be discarded (unit: channels)
        BANDWIDTH: KDE bandwidth (int, unit: channels)
        NCHANS: number of channels in original spectra (int)
        REMOVE_NOISE: If True, searches for output of clean_noise_squares function, to remove GCs associated with stray radiation - use False for GASS footprint (bool)
        
    Output:
        creates file sdir+SFILENAME which stores a dictionary with all clouds in each big pixel and their properties
    '''
    # Load the file that contains Gaussian components (GCs) by pixel (output of function 'collect' in preprocess_functions)
    GC_per_pixel = pickle.load(open(datafile,'r'))
    pixids = GC_per_pixel.keys()

    # This dict will hold all clouds in all NSIDE 32 pixels
    clouds_per_sqdeg= {}

    # Make arrays 
    allpixels = []
    alll = []
    allb = []
    allcloudmean = []
    allcloudmeanstd = []

    #print 'Converting peak amplitude by dividing with sqrt(2pi)'

    for ii,pixid in enumerate(pixids):
        sys.stdout.write("\r'Pixel number %d of %d'" %(ii,len(pixids)-1))
        sys.stdout.flush()
        
        # Get pixel data
        means_uncut, means_kms_uncut, sigmas_uncut, sigmas_kms_uncut, ampskmsK_uncut, hpxind_uncut, amps_uncut_norm = \
            GC_per_pixel[pixid]['means'], GC_per_pixel[pixid]['means_kms'], \
            GC_per_pixel[pixid]['sigmas'], GC_per_pixel[pixid]['sigmas_kms'],\
            GC_per_pixel[pixid]['line_integral_kmsK'], GC_per_pixel[pixid]['hpxindices'], GC_per_pixel[pixid]['peak_amp']

        # The amplitude of the decomposition is currently A = peak/sqrt(2pi). Convert to correct peak amplitude.
        amps_uncut = amps_uncut_norm*np.sqrt(2*np.pi)

        # If you've run the clean_from_noise_boxes method, then there will be a mask with 
        # GCs to be removed due to stray radiation
        # Read to mask out 'bad' gaussians that have been affected by the noise squares
        if REMOVE_NOISE:
            mask = GC_per_pixel[pixid]['mask_bad']
            # Make cuts and include this mask
            cuts = np.logical_and(mask, np.logical_and(np.logical_and(sigmas_uncut < SIGMACUT, amps_uncut > RMSCUT),\
	    	    np.logical_and(means_uncut>=0, means_uncut<NCHANS)))
           
        else:
            cuts = np.logical_and(np.logical_and(sigmas_uncut < SIGMACUT, amps_uncut > RMSCUT),\
	    		    np.logical_and(means_uncut>=0, means_uncut<NCHANS))
            
            if ii == 1:
                warnings.warn('WARNING: REMOVE_NOISE is set to False. Uncorrected stray radiation may still exist - run clean_from_noise_boxes.')

        means = means_uncut[cuts]
        means_kms = means_kms_uncut[cuts]
        sigmas = sigmas_uncut[cuts]
        sigmas_kms = sigmas_kms_uncut[cuts]
        amps = amps_uncut[cuts]
        ampskmsK = ampskmsK_uncut[cuts]
        hpxind = hpxind_uncut[cuts]

        # Intersect many points within the x axis. This is used only to evaluate the PDF, not construct it.
        # Add padding on either side to avoid edge effects in the derivatives
        pad = 10
        # remember means is in channels not km/s or m/s
        chans = np.arange(means.min()-pad, means.max()+pad, 0.1) # make the evaluated pdf smooth
        
        # Find pdf with KDE gaussian kernel
        pdf = kde_sklearn(means, chans, bandwidth=BANDWIDTH)

        # Get locations of minima and maxima in the pdf
        minima, maxima = get_min_max_der(chans,pdf)

        # Assign minima as edges of maxima
        try:
            r,l, newmaxima = neighboring_minima(minima, maxima, pdf, chans )
            # mask out the maxima that are not well separated by mins/zeros
            maxima = newmaxima
        except:

            plt.clf()
            plt.plot(chans,pdf)
            plt.scatter(chans[minima],np.zeros_like(chans[minima]),c='k')
            plt.scatter(chans[maxima],np.zeros_like(chans[maxima]),c='r')
            for mm in chans[maxima]:
                plt.axvline(mm)
            plt.show()
            print 'Skipping pixid',pixid, 'due to error - Something wrong with minmax determination!'#minima, maxima
            print len(minima), len(maxima)
            raise

        N_gaussians = np.zeros_like(chans[maxima])

        clouds = []

        clouds_per_sqdeg[pixid] = {}

        # Loop over clouds (peaks) in this NSIDE32 pixel
        for i,maxim in enumerate(chans[maxima]):
            cloud = {}
            #cloud['pdf'] = pdf # keep the pixel's pdf - this will be the same for all clouds in pixel pixid
            #cloud['pdfchans'] = chans
            # Assign all gaussians with mean within peak range to this peak/cloud
            within_peak = np.logical_and(means<r[i],means>l[i])
            # Count how many Gaussians this cloud (peak) has
            N_gaussians[i] = len(means[within_peak])

            if len(means[within_peak])>0:
                cloud['means'] = means[within_peak]
                cloud['sigmas'] = sigmas[within_peak]
                #cloud['means_kms'] = means_kms[within_peak]
                #cloud['sigmas_kms'] = sigmas_kms[within_peak]
                cloud['amps'] = amps[within_peak]
    	        cloud['line_integral_kmsK'] = ampskmsK[within_peak]
                cloud['hpxind'] = hpxind[within_peak]

                # Find the intensity of the cloud by summing the contribution of all its Gaussians 
                # and integrating over all velocties
                # First transpose arrays so that they are columns and can be fed into the gaus function
                # Each column will correspond to the same velocity and each row will refer to one Gaussian
                #gaussian_array = gaus( np.reshape(cloud['means_kms'],(1,len(cloud['means']))).T, \
                #                       np.reshape(cloud['sigmas_kms'],(1,len(cloud['means']))).T, \
                #                       np.reshape(cloud['amps'],(1,len(cloud['means']))).T, \
                #                       np.arange(vmin,vmax,dv))

                #cloud['intensities'] = gaussian_array.sum(axis = 1).T
                #cloud['cloudintensity'] = cloud['intensities'].sum()#/len() # NOT CORRECT! Must find average integrated intensity within this big pixel
            	
                
                # Find the average Gaussian mean and its std
                cloudmean = np.mean(cloud['means'])
                cloudmeanstd = np.std(cloud['means'])

                clouds.append(cloud)

            if RETURN_ARRS:
                # Find unique NSIDE 1024 pixels that have gaussians within this cloud
                unique_pixels_with_cloud, indices = np.unique(cloud['hpxind'], return_index = True)

                allpixels+=list(unique_pixels_with_cloud)

                gal_l,gal_b = hp.pix2ang(1024,unique_pixels_with_cloud.astype('int'),nest=False,lonlat = True)

                alll+=list(gal_l)
                allb+=list(gal_b)
                allcloudmean+=list(np.ones_like(unique_pixels_with_cloud)*cloudmean)
                allcloudmeanstd +=list(np.ones_like(unique_pixels_with_cloud)*cloudmeanstd)

        clouds_per_sqdeg[pixid]['pdf'] = pdf # keep the pixel's pdf - this will be the same for all clouds in pixel pixid
        clouds_per_sqdeg[pixid]['pdfchans'] = chans # keep the channels used to make pdf
        clouds_per_sqdeg[pixid]['clouds'] = clouds # keep cloud info for this pixel

    pickle.dump(clouds_per_sqdeg, open(sdir+SFILENAME, 'w'))
    
    if RETURN_ARRS:
        return alll,allb,allcloudmean,allcloudmeanstd,allpixels
    else:
        return
