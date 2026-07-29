import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

import sys
sys.path.append("..")

from pyvesta import Spectra

#filename to spectrum
filename = ''


SpectraList = Spectra.SpectraList.load(filename)

SNR_limit = 10

for specnr in range(SpectraList.nr_of_spectra()):
    fig, axs = plt.subplots(2, figsize=(16,9), sharex=True)

    fluxlist  = np.array([])
    errorlist = np.array([])
    
    for ordernr in range(SpectraList[specnr].nr_of_orders()):     
        order = SpectraList[specnr][ordernr]

        inds = np.where(order.flux / order.errors > SNR_limit)[0]
        
        if isinstance(SpectraList[specnr][ordernr], Spectra.SpectralOrder):
            axs[0].plot(order.wave[inds], order.flux[inds])
            #axs[0].plot(order.wave, median_filter(order.flux, size=1000))
            #axs[0].scatter(order.wave, order.flux)
            axs[1].plot(order.wave[inds], order.errors[inds])
            axs[0].set_xlabel(r'$\lambda$ in $\AA$')
            axs[1].set_xlabel(r'$\lambda$ in $\AA$')
        else:
            axs[0].plot(order.pixels[inds], order.flux[inds])
            axs[1].plot(order.pixels[inds], order.errors[inds])
            axs[0].set_xlabel('Pixel')
            axs[1].set_xlabel('Pixel') 
        
        fluxlist  = np.append(fluxlist, order.flux)
        errorlist = np.append(errorlist, order.errors)
        
        
    axs[0].set_ylabel('Flux in a.u.')
    axs[0].set_ylim((-0.5 * np.nanmedian(fluxlist), 2 * np.nanmedian(fluxlist)))
    axs[1].set_ylabel('Errors in a.u.')
    axs[1].set_ylim((-0.5 * np.nanmedian(errorlist), 2 * np.nanmedian(errorlist)))
    fig.suptitle('spectrum {} of file {}'.format(specnr, filename))
        
    axs[0].grid()
    axs[1].grid()
        
    plt.tight_layout()
     
    plt.show()
