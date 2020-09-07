import numpy as np
import healpy as hp
import main
import preprocess_functions

# HI4PI
CRPIX3_hi4pi= 466
CDELT3_hi4pi = 1288.21496


def chan2velo(channel, CRPIX3 = CRPIX3_hi4pi, CDELT3 = CDELT3_hi4pi):
    """
    Convert Channel to LSR velocity in m/s
    """
    return (channel - CRPIX3) * CDELT3


def make_arrays_vcut(clouds_per_sq_deg, nside_low = 64, Ngauscut = 20, \
                     NHcut = 0, keeplow = False, cmin = 0, cmax = 933,\
                     compute_centroid = False, modify_cloudlist = False):
    '''
    Create maps of Nclouds, NH, apply cuts to cloud properties, gather cloud properties in arrays for analysis
    Input: clouds_per_sq_deg is a dictionary with keys corresponding to each superpixel. 
           The object stored in each key is a list of dictionaries. Each dict summarizes the properties of a single cloud.
    Keywords: 
    cmin, cmax = minimum and maximum vel channel to be included (float/integer in the range 0,933). 
                 Any cloud with centroid velocity outside this range will be discarded.
    nside_low = Nside of the analysis (Corresponding to the choice of superpixel size). 
                Must match Nside chosen to create clouds_per_sq_deg. (int)

    Ngauscut = Threshold on number of Gaussians of a cloud. Clouds with less than this number of Gaussians
                are discarded. (int)
    NHcut = Threshold on cloud column density. Clouds with less NH will be discarded (if keeplow = False)
            if keeplow = True then cloud with NH > NHcut will be discarded (float, units cm^{-2})
    compute_centroid = If True, computes the full cloud spectrum by evaluating the Gaussian components,
                        and finds centroid velocity, skewness, number of maxima in cloud spectrum. 
                        Increases runtime. (bool)
    modify_cloudlist = If True adds the attributes 'centroid','skew','nmaxima' to each cloud in clouds_per_sq_deg. 
                        Should be used with compute_centroid = True. (bool)
                
    '''
    if modify_cloudlist == True:
        assert modify_cloudlist==compute_centroid
        
    assert cmin < cmax
    assert cmin >= 0
    assert cmax <=933

    
    # there is a set of bad pixels in the HI4PI data at this location. do not count these pixels
    mask_l,mask_b = 208.98, -19.35
    vec  = hp.ang2vec(mask_l, mask_b, lonlat=True)
    badpixels = hp.query_disc(nside_low, vec, np.radians(0.25), inclusive = True, nest = True )
    
    # Initialize maps at the superpixel resolution (low res)
    Neff_map_lowres = np.zeros(hp.nside2npix(nside_low))+hp.pixelfunc.UNSEEN
    Nclouds_map_lowres = np.zeros(hp.nside2npix(nside_low))+hp.pixelfunc.UNSEEN
    NH_map_lowres = np.zeros(hp.nside2npix(nside_low))+hp.pixelfunc.UNSEEN
    
    # Initialize maps at the native resolution of the HI4PI data
    Nclouds_map_highres = np.zeros(hp.nside2npix(1024))+hp.pixelfunc.UNSEEN
    NH_map_highres = np.zeros(hp.nside2npix(1024))+hp.pixelfunc.UNSEEN

    # Initialize arrays to store cloud properties after implementing cuts
    # Each entry in these 1d lists corresponds to 1 cloud
    superpixel_array = np.array([]) # keeps value of superpixel (NESTED) corresponding to each cloud
    cloud_NHs = np.array([]) # keeps column density of clouds (low res)
    cloud_mean_vels = np.array([]) # keeps centroid velocity of clouds 
    cloud_mean_sigmas = np.array([]) # keeps mean standard deviation of the Gaussians in each cloud
    Nmaxima = np.array([]) # counts how many peaks in mean spectrum of a cloud
    Ngaus = np.array([]) # how many gaussians in each cloud
    Npix = np.array([]) # how many high-res (Nside 1024) pixels each cloud occupies
    skewnessmeanspec = np.array([]) # keeps value of skewness (difference between centroid and peak velocity) of each cloud spectrum

    # Initialize 1d array for storing the number of clouds per superpixel (after implementing cuts on cloud properties)
    Nclouds = [] # Each entry corresponds to a unique (low-res) superpixel
    # 1d array to store the number of gaussians per superpixel
    Ngaus_per_pixel = np.array([])
    
    
    # Keep track of progress
    print 'Total of', len(clouds_per_sq_deg.keys())
    
    # Loop over superpixels
    for kk,key in enumerate(clouds_per_sq_deg.keys()):
        
        # Keep track of progress
        print '{0}\r'.format(kk), 
        
        # skip the bad pixels in the dataset
        if key in badpixels:
            continue
        
        clouds = clouds_per_sq_deg[key]['clouds']
        # Initialize the cloud count for this pixel
        N=0
        # Count the number of Gaussians that were used for the PDF in this pixel
        Ngaus_per_pix = 0
        
        # Make array of cloud NH (at the superpixel level) for calculating NH-weighted Neff
        NH_per_cloud_arr = np.array([])
        
        # Loop over clouds
        for cloud in clouds:
            
            # Define new attributes to the cloud (will help future analyses)
            if modify_cloudlist == True:
                cloud['centroid'] = np.nan # velocity centroid (in channels)
                cloud['skew'] = np.nan # velocity centroid - peak of spectrum (in channels)
                cloud['nmaxima'] = np.nan # number of maxima in the cloud spectrum
                cloud['std'] = np.nan # Standard deviation of cloud spectrum
            
            # Add the number of gaussians of this cloud to the count for this superpixel
            Ngaus_per_pix += len(cloud['means'])
            
            # Skip the cloud if it does not have enough Gaussians associated with it
            if len(cloud['means']) < Ngauscut:
                continue
                
            else:
                
                # Create array of channels for computing the cloud spectrum later on.
                # Since max sigma of gaussians is ~14 channels, safe to evaluate spectrum in this range:
                # this is just for calculating centroid, not the column density
                chans = np.arange(cloud['means'].min()-40,cloud['means'].max()+40)
                
                # Compute vel centroid and skewness of each cloud spectrum - takes time
                if compute_centroid:
                    
                    # Create an array with spectra from each gaussian in the cloud
                    gaussian_array = main.gaus( np.reshape(cloud['means'],(1,len(cloud['means']))).T, \
                                   np.reshape(cloud['sigmas'],(1,len(cloud['means']))).T, \
                                   np.reshape(cloud['amps'],(1,len(cloud['means']))).T, \
                                   chans)
                    # Compute the mean spectrum (sum signal over all Gaussians)
                    meanspec = np.average(gaussian_array, axis = 0).T
                    # Get the centroid velocity (expectation value) of the mean spectrum
                    mean_of_spec = np.sum(meanspec*chans)/np.sum(meanspec)
                    # Get the peak velocity
                    peakvel = chans[np.argmax(meanspec)]
                    # Count number of maxima in the cloud spectrum
                    maxs = main.get_min_max_der(chans,meanspec)[1]
                    # Compute measure of skewness of the spectrum: difference between cenroid and peak
                    skew = mean_of_spec-peakvel
                    # Get variance (second moment) of cloud spectrum
                    var = np.sum(meanspec*(chans - mean_of_spec)**2)/np.sum(meanspec)

                    
                    cloud['centroid'] = mean_of_spec
                    cloud['skew'] = skew
                    cloud['nmaxima'] = chans[maxs].shape[0]
                    cloud['std'] = np.sqrt(var)
                
                # If you are not computing it, read the centroid value from the cloud list
                else:
                    mean_of_spec = cloud['centroid']
                    skew = cloud['skew']
                    maxs = cloud['nmaxima']
                    std = cloud['std']
                    
                # Apply cut on centroid velocity
                if mean_of_spec < cmin or mean_of_spec > cmax:
                    continue
                    
                elif mean_of_spec != mean_of_spec:
                    continue
                        
                # Calculate column density of this cloud (averaged within the superpixel)
                nh = 1.823*1e18*cloud['line_integral_kmsK'].sum()/len(np.unique(cloud['hpxind']))
                
                # Apply cut on column density
                if NHcut!=0:
                    if keeplow:
                        if nh > NHcut: continue
                    else:
                        if nh < NHcut: continue
                 
                # ----- At this point the cloud has passed all quality cuts ----
                # We now store its properties for further analysis
                
                # Add 1 to the number of clouds in this superpixel
                N+=1
                
                # Calculate column density of each Nside 1024 pixel
                for hpx in np.unique(cloud['hpxind']):
                    # sum the signal from all Gaussians in the same hpx pixel in this cloud
                    nh_highres = 1.823*1e18*cloud['line_integral_kmsK'][cloud['hpxind'] == hpx].sum()
                    
                    if NH_map_highres[hpx] == hp.pixelfunc.UNSEEN:
                        NH_map_highres[hpx] = nh_highres
                        Nclouds_map_highres[hpx] = 1
                    else:
                        NH_map_highres[hpx] += nh_highres
                        Nclouds_map_highres[hpx] += 1
                
                # Add cloud properties to the output
                skewnessmeanspec = np.append(skewnessmeanspec, skew)   
                Nmaxima = np.append(Nmaxima,maxs)
                cloud_NHs = np.append(cloud_NHs, nh)
                cloud_mean_sigmas = np.append(cloud_mean_sigmas, np.mean(cloud['sigmas']))
                superpixel_array = np.append(superpixel_array, key)
                cloud_mean_vels = np.append(cloud_mean_vels, mean_of_spec) # Keep velocity
                
                # Gather all NH of clouds in this superpixel
                NH_per_cloud_arr = np.append(NH_per_cloud_arr,nh)
                
                # Store number of Gaussians and number of pixels of this cloud
                Ngaus = np.append(Ngaus, len(cloud['means']))
                Npix = np.append(Npix, len(np.unique(cloud['hpxind'])))

                # Keep the value of NH from HI signal within clouds
                if NH_map_lowres[key] <= -1.6375e30:
                    NH_map_lowres[key] = nh
                else:
                    NH_map_lowres[key] += nh

        
        # Keep Number of clouds in this pixel  
        Nclouds.append(N)
        # if no cloud fulfills criteria, reset N to UNSEEN
        if N == 0:
            N = hp.pixelfunc.UNSEEN
        Nclouds_map_lowres[key] = N
        
        # Calculate Neff
        if len(NH_per_cloud_arr) > 0:
            # Find NH of cloud with max NH in this pixel
            maxNH = np.max(NH_per_cloud_arr)
            Neff_map_lowres[key] = np.sum(NH_per_cloud_arr/maxNH)
        else:
            # there is no cloud in this pixel, Neff = 0
            Neff_map_lowres[key] = 0.
        
        Ngaus_per_pixel = np.append(Ngaus_per_pixel,Ngaus_per_pix)

    
    if modify_cloudlist:
        return  Nclouds_map_lowres, Nclouds_map_highres, NH_map_lowres, NH_map_highres,\
           superpixel_array, cloud_NHs, cloud_mean_vels, cloud_mean_sigmas, Nclouds, Ngaus, Npix, \
            Ngaus_per_pixel, skewnessmeanspec, Neff_map_lowres, skewnessmeanspec, Nmaxima, clouds_per_sq_deg
    
    else:
        return Nclouds_map_lowres, Nclouds_map_highres, NH_map_lowres, NH_map_highres,\
           superpixel_array, cloud_NHs, cloud_mean_vels, cloud_mean_sigmas, Nclouds, Ngaus, Npix, \
           Ngaus_per_pixel, skewnessmeanspec, Neff_map_lowres, skewnessmeanspec, Nmaxima
