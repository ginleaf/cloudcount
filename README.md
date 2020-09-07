# cloudcount
HI cloud identification method

- *Version:* 1.0.1
- *Author:* Gina Panopoulou
========

Purpose
=======
`cloudcount' uses a Gaussian decomposition of HI line emission data to identify kinematicaly distinct peaks in the spectra (`clouds'). 

Preferred citation method
=========================

Please cite our paper if you use `cloudcount` for your projects.

@ARTICLE{2020arXiv200400647P,
       author = {{Panopoulou}, G.~V. and {Lenz}, D.},
        title = "{Maps of the number of HI clouds along the line of sight at high galactic latitude}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - Solar and Stellar Astrophysics},
         year = 2020,
        month = apr,
          eid = {arXiv:2004.00647},
        pages = {arXiv:2004.00647},
archivePrefix = {arXiv},
       eprint = {2004.00647},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200400647P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

Dependencies
======
The program has been tested to work with the following package versions:
- python 2.7
- healpy 1.12.9
- astropy 2.0.9
- pytables 3.4.4
- numpy
If you want to run the notebooks yourself, you will also need the Jupyter
server and matplotlib.

Where to start
======
If you wish to run the method on a selected patch of sky, follow the notebook `Running_the_method' (or it's .py version)
If you wish to use the data products presented in the paper, follow the notebook `PublicData/DataUsageTutorial' (or it's .py version)
