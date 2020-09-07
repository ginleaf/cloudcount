import numpy as np
import tables
import matplotlib.pyplot as plt
import os
import time
import healpy as hp
import sys
import pickle
from astropy.coordinates import SkyCoord

def nside2order(nside):
    '''
    Replace hp.nside2norder for old version of healpy
    '''
    return len("{0:b}".format(nside)) - 1


def collect(nside_low, nside_high, nside_low_pix_array_NEST, gaussdecompfile, savedir, savename):

    ''' Collect parameters of all Gaussians in a pixel of given NSIDE
    Input
        nside_low: NSIDE of the larger pixels (integer)
        nside_high: NSIDE of the Gaussian decomposition (default 1024) (integer)
        nside_low_pix_array_NEST: 1d array of healpix pixel indices. Each element 
            is a pixel of NSIDE nside_high (NESTED ordering), within which we 
            wish to collect all the Gaussians.
        gaussdecompfile: filename of the file that contains the Gaussian decomposition
        savedit: path to save output (string)

    Output
        Saves 4 files for every nside_low_pix_array_NEST pixel, one containing 
        the Gaussian mean velocities, one the amplitudes, one the sigmas, and one the healpix indices.  
        
    '''
    # If the output folder does not exist, create it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    store = tables.open_file(gaussdecompfile) # Run with python 3.5
    table = store.root.gaussdec

    # Get all healpix indices of the Gaussian decomposition
    hpxindices = table.cols.hpxindex[:]
    # Bring all full-res indices to nested ordering
    hpxindices_nest = hp.ring2nest(nside_high, hpxindices)

    # Read table columns as numpy arrays to speed up the later steps
    tca = table.cols.peak_amplitude[:]
    tcp = table.cols.line_integral_kmsK[:]
    tcc = table.cols.center_c[:]
    tcs = table.cols.sigma_c[:]
    tcv = table.cols.center_kms[:]
    tcsk = table.cols.sigma_kms[:]

    steps = nside2order(nside_high) - nside2order(nside_low)
    if steps <= 0:
        raise ValueError('nside_high must be larger than nside_low.')

    # Calculate the lower-res pixel for each high-res pixel
    idx_rightshifted = np.right_shift(hpxindices_nest, 2 * steps)

    # Create a dictionary that will store the Gaussian components per big pixel
    GCs_per_pixel = {}

    for pp, pixid in enumerate(nside_low_pix_array_NEST):

        sys.stdout.write("\r'Pixel number %d of %d'" %(pp,len(nside_low_pix_array_NEST)-1))
        sys.stdout.flush()

        # Build a boolean mask: points to all small pixels that are within this big pixel
        mask = (idx_rightshifted == pixid)

        # Now get parameters of Gaussian components    
        allamps = tca[mask]
        allampskmsK = tcp[mask]

        allmeans = tcc[mask]
        allmeanskms = tcv[mask]

        allsigmas = tcs[mask]
        allsigmaskms = tcsk[mask]
        
        allpixids = hpxindices[mask]

        GCs_per_pixel[pixid] = {}
        GCs_per_pixel[pixid]['sigmas'] = allsigmas
        GCs_per_pixel[pixid]['sigmas_kms'] = allsigmaskms
        GCs_per_pixel[pixid]['peak_amp'] = allamps
        GCs_per_pixel[pixid]['line_integral_kmsK'] = allampskmsK
        GCs_per_pixel[pixid]['means'] = allmeans
        GCs_per_pixel[pixid]['means_kms'] = allmeanskms
        GCs_per_pixel[pixid]['hpxindices'] = allpixids

    pickle.dump(GCs_per_pixel, open(savedir+'/'+savename, 'w'))

    store.close()

    return


def chan2velo(channel, CRPIX3 = 466, CDELT3 = 1288.21496):
    """
    Convert Channel to LSR velocity in m/s. Default CRPIX3 and CDELT3 are from the HI4PI fits header
    """
    return (channel - CRPIX3) * CDELT3


def clean_from_noise_squares(gcfilename, noisefilename, sdir, bigpixnside, nside_high, CRPIX3, CDELT3):

    # What is the angular resolution of these pixels
    reso = hp.nside2resol(bigpixnside, arcmin = True)/60.

    # Mehod to transform channels to velocity (for noise squares)
    vel = chan2velo #Velo(header)

    # Read noise boxes
    box_info = np.genfromtxt(noisefilename, delimiter = ',', usecols = (1,2,3,5))

    # Create a map of all pixels that have a noise box associated with them
    allboxes = np.zeros(hp.nside2npix(nside_high))
    # Create a dictionary that contains for each box the pixels that are in it
    pixels_per_box = {}

    pixindices = hp.ang2pix(nside_high, box_info[:,1],box_info[:,2], lonlat = True, nest = False)
    box_center_vectors = hp.ang2vec( box_info[:,1],box_info[:,2], lonlat = True)
    box_centers = []

    for ii, box_center in enumerate(box_center_vectors):
        # Get the box size
        bsize = box_info[ii][3]
        
        if bsize >=4.5: # for a whole box
            radius = 3.5 # make it large so that corners are inside
        else: # for a half box
            radius = 2
            
        # Find all pixels within a circle approximating the box size
        pixels_in_this_box = hp.query_disc(nside_high, box_center_vectors[ii], np.radians(radius), nest = False)

        # Mark the pixels associated with this box
        allboxes[pixels_in_this_box] +=1
        
        pixels_per_box[ii] = pixels_in_this_box

    box_centers = SkyCoord(frame='galactic', l = box_info[:,1], b = box_info[:,2], unit = 'deg')


    # Load the data groupped by big pixel
    GC_per_pixel = pickle.load(open(gcfilename,'r'))

    pixids = GC_per_pixel.keys()

    # For each big pixel, Clean array of gaussians from those that are affected by noise boxes

    # Update the dictionary to include a mask of GCs that should not be used
    for jj,pixid in enumerate(pixids): # pixids are in NESTED ordering while hpxindices are in RING
    	
        sys.stdout.write("\r'Pixel number %d of %d'" %(jj,len(pixids)-1))
        sys.stdout.flush()

        # Load pixel data
        means, means_kms, sigmas_kms, hpxind = \
                GC_per_pixel[pixid]['means'], GC_per_pixel[pixid]['means_kms'], \
                GC_per_pixel[pixid]['sigmas_kms'],\
                GC_per_pixel[pixid]['hpxindices']

        # Define an array of indices to be masked out (contaminated by noise boxes)
        indices_noise = np.ones_like(means_kms).astype('bool')
        
        
        # Find central coords of this pixel 
        l,b = hp.pix2ang(bigpixnside, pixid, lonlat = True, nest = True)
        pixcen = SkyCoord(frame='galactic', l= l, b= b, unit = 'deg')
        
        # Find boxes that have centers near enough so that they could affect the pixel
        boxinds_to_check = np.where(pixcen.separation(box_centers).deg < (np.sqrt(2.)*3.+np.sqrt(2.)*reso/2))[0]
        
        # Loop over gaussians
        
        for ii in range(len(means_kms)):
            
            # Check if there is no box at this location
            if allboxes[int(hpxind[ii])] == 0:
                continue
                
            else:
                # We need to remove the gaussian if it is associated with the noise
                # To find which channels are affected, 
                # loop over all boxes and check if pixel is associated with them
                for kk in boxinds_to_check:
                    # Check if this gaussian is in a pixel within this noise box
                    if len(pixels_per_box[kk][pixels_per_box[kk] == int(hpxind[ii])]) == 1:
                        
                        # Convert to velocity for comparison (the channel axis of the decomp is offset from the data axis)
                        this_box_vel = vel(box_info[kk][0]-1, CRPIX3, CDELT3) # 
                        
                        # Check if this gaussian's velocity is consistent with the channel where the noisebox appears
                        if np.absolute(means_kms[ii]-this_box_vel/1.e3) < 0.5*sigmas_kms[ii]:
                            indices_noise[ii] = False

        # Write array of masked indices for future use
        GC_per_pixel[pixid]['mask_bad'] = indices_noise
        
    # Save the output
    pickle.dump(GC_per_pixel, open(gcfilename,'w'))
    
    return
    





