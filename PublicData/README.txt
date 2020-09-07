Acknowledgement:

Publications that make use of these datasets should acknowledge the published version of:
Panopoulou & Lenz, 2020, Maps of the number of HI clouds along the line of sight at high galactic latitude
https://ui.adsabs.harvard.edu/abs/2020arXiv200400647P/abstract

The code used to generate these datasets is available at https://github.com/ginleaf/cloudcount

Abstract:

Characterizing the structure of the Galactic Interstellar Medium (ISM) in three dimensions is of high importance for accurate modeling of dust emission as a foreground to the Cosmic Microwave Background (CMB). At high Galactic latitude, where the total dust content is low, accurate maps of the 3D structure of the ISM are lacking. We develop a method to quantify the complexity of the distribution of dust along the line of sight with the use of HI line emission. The method relies on a Gaussian decomposition of the HI spectra to disentangle the emission from overlapping components in velocity. We use this information to create maps of the number of clouds along the line of sight. We apply the method to: (a) the high-galactic latitude sky and (b) the region targeted by the BICEP/Keck experiment. In the North Galactic Cap we find on average 3 clouds per 0.2 square degree pixel, while in the South the number falls to 2.5. The statistics of the number of clouds are affected by Intermediate-Velocity Clouds (IVCs), primarily in the North. IVCs produce detectable features in the dust emission measured by Planck. We investigate the complexity of HI spectra in the BICEP/Keck region and find evidence for the existence of multiple components along the line of sight. The data and software are made publicly available, and can be used to inform CMB foreground modeling and 3D dust mapping.

Data contents:

For Polar Caps regions (defined in Section 4.1 of the paper)

1) 2 HEALPix fits files (nside 128, Galactic coordinates, Nested indexing scheme) containing a binary table with maps. 
* Ncloud_map.fits contains 2 maps (stored as columns in the binary table): first column holds the number of clouds, second holds the column-density-weihted number of clouds (eq.4). 
* NHI_cloud_maps.fits contains 3 maps (stored as columns in the binary table): first column is the column density from clouds within the velocity range [-70,70] km/s, second column is the column density of LVCs and third is the column density of IVCs.

2) HDF5 file containing 6 datasets (equivalent to columns in a binary table) each holding 477711 values that correspond to each of the 477711 clouds identified by the method: Ngaussians = number of Gaussians belonging to each cloud, Nmaxima = Number of peaks in cloud spectrum, centroids = velocity centroid of cloud spectrum (km/s), std = second moment of cloud spectrum (km/s), cloudNH = column density of the cloud (cm^-2), hpxpix = healpix pixel where the cloud is located (Nside 128, nested indexing)

The same types of files are provided separately for the BICEP/Keck region (Section 4.2 in the paper).


