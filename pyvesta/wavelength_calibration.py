######################################
# Created  2025/10/16
# 
# Author: Lukas Stock
#
# This file contains routines for the wavelength calibration of echelle spectrographs using ThAr reference spectra
#####################################

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import optimize, interpolate, ndimage, signal
import logging
import itertools
from multiprocessing import Pool

from pyvesta import ThAr_Peak_detection
from pyvesta import Spectra
from pyvesta import FitFunctions
from pyvesta import datashare
from pyvesta import plot_utilities

def init_pools(reduction_parameters, instrument, camera, current_filename):
    datashare.reduction_parameters = reduction_parameters
    datashare.instrument           = instrument
    datashare.camera               = camera
    datashare.current_filename     = current_filename

def _normalize_coordinates(coordinates, min_value, max_value):
    assert min_value < max_value

    return 2. * (coordinates - min_value)/(max_value - min_value) - 1.

def n_Edlen(w):
    """
    # Calculate the refractive index of air
    # source https://iopscience.iop.org/article/10.1088/0026-1394/2/2/002
    #
    # :param w: float or numpy array of floats, wavelengths in angstrom to calculate refractive index at
    #
    # :return n, float or numpy array of floats, has same shape as w, refractive index of air at wavelengths w
    """
    sigma2 = np.power(1e4 / w, 2.)
    n = 1 + 1e-8 * (8342.13 + 2406030 / (130-sigma2) + 15997/(38.9-sigma2))
    return n

def ToVacuum(w):
    """
    # Convert air wavelengths to vacuum wavelengths
    #
    # :param w: float or numpy array of floats, wavelengths in angstrom to convert
    #
    # :return w', float or numpy array of floats, has same shape as w, vacuum wavelengths
    """
    return w / n_Edlen(w)

def ToAir(w):
    """
    # Convert vacuum wavelengths to air wavelengths
    #
    # :param w: float or numpy array of floats, wavelengths in angstrom to convert
    #
    # :return w', float or numpy array of floats, has same shape as w, air wavelengths
    """
    return w * n_Edlen(w)


#FIXME: Originally V2, change in rest of code
def eval_wave_coeffs(coeffs, m0, normed_ordernr, normed_pix, ordernr):
    """
    #   Evaluate wavelengths using chebychev polynomials
    #
    #   :param coeffs: 2D numpy array, coefficient matrix of chebychev fit
    #   :param m0: int, physical diffraction order of first detected order
    #   :param normed_ordernr: float or numpy array of floats, normed order number(s) to evaluate wavelengths at
    #   :param normed_pix: float or numpy array of floats, normed pixel(s) to evaluate wavelengths at
    #   :param ordernr: int or numpy array of ints, software order(s) to evaluate wavelengths at
    """

    #shortcut
    eval2d = np.polynomial.chebyshev.chebval2d

    m0 = float(m0)

    #create array of normed_ordernr if it is a scalar
    if isinstance(normed_ordernr, float) or isinstance(normed_ordernr, int):
        order_array = np.ones_like(normed_pix) * normed_ordernr
    else:
        order_array = normed_ordernr

    #evaluate wavelength
    return (1./np.add(m0,ordernr)) * (eval2d(order_array, normed_pix, coeffs))


def find_order_overlap(order1, order2, peaks1, peaks2, scale_range=(-1e5, 1e5), nord=4, rel_height_diff=0.25, pixel_tol=5):
    """
    #    Find polynomial chebychev coefficients, which map order1 over order2
    #    Return those coefficients and a list of all found matched lines
    #
    #   :param order1: SpectralOrder object, which contains the "blue" spectral order
    #   :param order2: SpectralOrder object, which contains the "red" spectral order
    #   :param peaks1: list of integers, indices of ThAr peaks in order1
    #   :param peaks2: list of integers, indices of ThAr peaks in order2

    #   :param scale_range: touple of floats, (min_range, max_range), minimal and maximal values for polynomial interpolation coefficients, default (-1e5, 1e5)
    #   :param nord: int, degree of polynomial interpolation (default 4)
    #   :param rel_height_diff: float, maximal accepted relative height difference to match peaks. Default is 0.25, but you might to increase this value if no flat normalization is applied
    #   :param pixel_tol: int, maximal difference between original and interpolation to detect lines (default 5)
    #   :return best_coeffs: numpy.ndarray, coefficients of interpolation between order1 and order2
    #   :return found_matches: touple (matched_peaks1, matched_peaks2), each entry is a onedimensional numpy.ndarray of pixels with ThAr peaks for which a match was found
    #
    """

    #shortcuts
    #Prefer chebyshev polynomials over normal polynomials
    vander  = np.polynomial.chebyshev.chebvander
    polyval = np.polynomial.chebyshev.chebval

    #copy orders to avoid changes at original
    order1 = order1.copy()
    order2 = order2.copy()

    #substract background
    #order1.flux -= fit_undergroud_thar(order1.flux, nord=2, chunk_width=100)
    #order2.flux -= fit_undergroud_thar(order2.flux, nord=2, chunk_width=100)

    #get height of peaks
    order1_peakheights = order1.flux[np.around(peaks1).astype(int)]
    order2_peakheights = order2.flux[np.around(peaks2).astype(int)]

    """
    print('')

    fig, axs = plt.subplots(2)

    #as wavelengths are propto 1/m is wavelength(m) > wavelength(m+1)
    axs[1].plot(order1.pixels, order1.flux, linewidth=0.75)
    axs[1].scatter(peaks1, order1_peakheights, s=1)

    axs[0].plot(order2.pixels, order2.flux, linewidth=0.75)
    axs[0].scatter(peaks2, order2_peakheights, s=1)

    max_flux = np.max((np.stack((order1.flux, order2.flux))))

    axs[0].set_ylim([-2, 1.1* max_flux])
    axs[1].set_ylim([-2, 1.1* max_flux])

    fig.suptitle('Order overlaps')

    fig.tight_layout()
    plt.show()
    """


    #create matrix from heights
    height_matrix1 = order1_peakheights.reshape((-1,1)) * np.ones(shape=(len(peaks1),len(peaks2)))
    height_matrix2 = np.ones(shape=(len(peaks1),len(peaks2))) * order2_peakheights

    #compare heights
    height_correlations = np.abs(height_matrix1 - height_matrix2) < rel_height_diff * np.mean([height_matrix1, height_matrix2], axis=0)

    #throw away lines which have to other lines with comparable height
    lines_with_parters = np.where(np.sum(height_correlations, axis=1) > 0)[0]

    #filter peaks with matching partner
    valid_peaks1 = peaks1[lines_with_parters]
    height_correlations = height_correlations[lines_with_parters]

    #if not enough peaks were found to interpolate return None
    if len(valid_peaks1) < nord+2 or len(peaks2) < nord +2:
        return None, None

    #create vandermode matrix for valid_peaks1
    comb_vander   = vander(valid_peaks1, deg=nord)

    #get closest blue line interpolator
    blue_interpolator = interpolate.interp1d(peaks2, peaks2, kind='nearest', bounds_error=False, fill_value=(np.min(peaks2), np.max(peaks2)))

    valid_peaks1, peaks2 = np.array(valid_peaks1), np.array(peaks2)

    #create all posible combinations
    combinations1 = itertools.combinations(np.arange(len(valid_peaks1)), nord+2)

    #create vandermode matrix for pixel range
    pix_range = np.arange(np.min(valid_peaks1), np.max(valid_peaks1)+1, 10)
    pix_vander = vander(pix_range, deg=nord)

    match_count = []
    all_coeffs  = []

    #go through all combinations
    for indices in combinations1:
        comb_peaks1 = valid_peaks1[list(indices)]

        #invert vandermode matrix
        inv_vander = np.linalg.pinv(comb_vander[list(indices)])

        #get possible blue matches
        possible_lines2 = [peaks2[height_correlations[list(indices)][i]] for i in range(len(indices))]


        #only allow strictly increasing lines
        comb_product = np.array(list(itertools.product(*possible_lines2)))
        comb_product = comb_product[np.all(comb_product[:, :-1] < comb_product[:, 1:], axis=1)]

        possible_lines2 = comb_product.reshape(-1, len(possible_lines2), 1)

        if len(possible_lines2) > 0:

            #create interpolated pixel range
            coeffs = np.matmul(inv_vander, possible_lines2)

            #ensure higher order coeffs are positive -> to double matching
            #coeffs[:, 2:, 0] = np.abs(coeffs[:, 2:, 0])

            new_pix_range = np.matmul(pix_vander, coeffs)
            #ensure to allow only coeffs with strictly increasing results

            coeffs = coeffs[np.all(new_pix_range[:, :-1] < new_pix_range[:, 1:], axis=1)[:,0],:]

            #positions of mapped peaks
            lines_mapped = np.matmul(comb_vander, coeffs)

            #count peaks with matching positions
            number_of_matched_peaks = np.count_nonzero(np.isclose(lines_mapped, blue_interpolator(lines_mapped),\
                                                                  atol=pixel_tol), axis=1).flatten()

            #only use coefficients which are within allowed range
            coeffs_in_range = np.logical_and(coeffs[:, 1, 0] > min(scale_range), \
                                             coeffs[:, 1, 0] < max(scale_range))

            coeffs, number_of_matched_peaks = coeffs[coeffs_in_range], number_of_matched_peaks[coeffs_in_range]


            #if there are matched peaks add them to the list
            if len(number_of_matched_peaks) > 0:
                match_count.append(np.max(number_of_matched_peaks))
                all_coeffs.append(coeffs[np.argmax(number_of_matched_peaks)])

    #if multiple sets of coefficients have some matched lines, use the one with the most matches
    if len(match_count) > 0:
        best_coeffs = all_coeffs[np.argmax(match_count)]
    else:
        return None, None

    #flatten coefficients
    best_coeffs = np.array(best_coeffs).flatten()

    #evaluate interpolated positions
    transf_peaks1 = polyval(valid_peaks1, best_coeffs)
    transf_peaks2 = blue_interpolator(transf_peaks1)

    #check which lines are matched well enough
    is_close = np.isclose(transf_peaks2, transf_peaks1, atol=pixel_tol)

    #if enough lines are matched: find best coefficients by fitting
    if np.count_nonzero(is_close) >= nord +1:
        #best_coeffs = np.polynomial.polynomial.polyfit(valid_peaks1[is_close], transf_peaks2[is_close], nord)
        best_coeffs = np.polynomial.chebyshev.chebfit(valid_peaks1[is_close], transf_peaks2[is_close], nord)

    #evaluate again
    #transf_peaks1 = np.polynomial.polynomial.polyval(valid_peaks1, best_coeffs)
    transf_peaks1 = polyval(valid_peaks1, best_coeffs)
    transf_peaks2 = blue_interpolator(transf_peaks1)

    #check if fit is valid. If one order is completely within the other order, the red order is more blue or the blue order is more red, the fit wasn't successful
    new_red_pixels = polyval(order1.pixels, best_coeffs)

    min_pix_red  = np.min(new_red_pixels)
    max_pix_red  = np.max(new_red_pixels)
    min_pix_blue = np.min(order2.pixels)
    max_pix_blue = np.max(order2.pixels)

    #check for bad fit
    if  (min_pix_red < min_pix_blue) or (max_pix_red < max_pix_blue) or \
        (min_pix_red < 0)            or (max_pix_red > (max_pix_blue + 3 * len(order2.pixels))):
        return None, None

    #find matches again
    is_close = np.isclose(transf_peaks2, transf_peaks1, atol=pixel_tol)

    found_matches = (valid_peaks1[is_close], transf_peaks2[is_close])

    #return best coefficients and matched lines
    return best_coeffs, found_matches


def fit_undergroud_thar(flux, nord=2, chunk_width=100):
    """
    # Fit underground of ThAr spectrum. Build median of big chunks and fit those, as median should be not affected by emission lines, as there is usually more background than emission lines.
    # TODO: this still needs to be improved!
    #
    # :param flux: 1D numpy array, ThAr flux
    # :param nord: int, polynomial degree of underground fit (default 2)
    # :param chunk_width: int, size of chunks to build median in pixels (default 100)
    #
    # :return underground: 1D numpy array, same size as flux, underground fit of flux
    """
    medians = []
    mids   = []

    #minimal and maximal index to build median
    min_index = 0
    max_index = np.min((min_index + chunk_width, len(flux) - 1))

    while min_index < len(flux) -1:
        #build median
        medians.append(np.nanmedian(flux[min_index:max_index]))
        #mid of chunk
        mids.append((max_index - min_index)/2.)

        #go to next chunk
        min_index = max_index
        max_index = np.min((min_index + chunk_width, len(flux) - 1))

    #fit chunks
    fit = np.polynomial.chebyshev.chebfit(mids, medians, deg=nord)#, w= 1./np.square(tilt_errs[use_inds]))

    #evaluate fit
    return np.polynomial.chebyshev.chebval(np.arange(len(flux)), fit)


def coeffs_from_overlaps(m0, Overlaps, min_pix, max_pix, min_order, max_order, nord, npix):
    """
    # Find 2D polynomial coefficients that map order and pixels to wavelengths using known overlaps between orders
    # The results of this method will be still off by a constant factor (the so called global scale), which needs to be determined later
    # Strongly inspired by https://github.com/gmbrandt/xwavecal
    #
    # :param m0: int, lowest physical order in the spectrum, that corresponds to software order 0
    # :param Overlaps: Overlaps object, contains the informations about the overlaps
    # :param min_pix: int, minimal pixel value, used for normalization
    # :param max_pix: int, maximal pixel value, used for normalization
    # :param min_order: int, minimal software order, used for normalization
    # :param max_order: int, maximal software order, used for normalization
    # :param nord: int, polynomial degree of fit in order (cross-dispersion) direction
    # :param npix: int, polynomial degree of fit in pixel (dispersion) direction
    #
    # :return coeffs_new: 2-D np.ndarray, shape (nord +1, npix), contains the polynomial coefficients of the wavelength solution
    # :return res: float, residuals of fit
    # :return used_overlaps: Overlaps object, overlaps used for coefficient determination
    # :return not_used_overlaps: Overlaps object, overlaps rejected and not used for coefficient determination
    """
    #shortcut
    eval2d = np.polynomial.chebyshev.chebval2d

    cheb_matrix = []

    # get all overlap pixels and orders
    all_red_pixels, all_blue_pixels = Overlaps.allPixels()
    all_red_orders, all_blue_orders = Overlaps.allOrders()

    # normalize coordinates, so that pixels and orders are in range [0.1]
    normed_red_pixel  = _normalize_coordinates(all_red_pixels , min_pix, max_pix)
    normed_blue_pixel = _normalize_coordinates(all_blue_pixels, min_pix, max_pix)
    normed_red_orders  = _normalize_coordinates(all_red_orders , min_order, max_order)
    normed_blue_orders = _normalize_coordinates(all_blue_orders, min_order, max_order)

    # build
    red_array , red_b  = make_coeff_matrix(normed_red_pixel , normed_red_orders , all_red_orders , m0, nord, npix)
    blue_array, blue_b = make_coeff_matrix(normed_blue_pixel, normed_blue_orders, all_blue_orders, m0, nord, npix)

    array = red_array - blue_array
    b     = red_b     - blue_b

    coeffs, res, rank, s, good_overlap_indices = fit_coeffs(array, b, fit_overlaps=True, min_size=2 * (npix+1)*(nord+1))

    all_overlap_indices = np.arange(len(all_red_pixels))


    used_overlaps     = Overlaps.subsample(good_overlap_indices)
    not_used_overlaps = Overlaps.subsample(all_overlap_indices[~np.isin(all_overlap_indices, good_overlap_indices)])

    #print(residuals)

    coeffs_new = np.zeros((nord+1, npix+1))
    coeffs_new[0,0] = 1
    coeffs_new[:, 1:] = coeffs.reshape((nord+1, npix))

    return coeffs_new, res, used_overlaps, not_used_overlaps

def fit_coeffs(array, b, fit_overlaps=True, min_size=-1):
    good_indices = np.arange(len(b))

    if min_size < 0:
        min_size = len(b)//2

    continue_fitting = True
    while continue_fitting:
        coeffs, res, rank, s = np.linalg.lstsq(array, b, rcond=None)

        residuals = np.dot(array, coeffs) + (-1.) * b
        rms       = np.sqrt(np.sum(np.square(residuals)))

        if array.shape[0] < min_size or (not np.any(np.abs(residuals) > 3 * np.median(np.abs(residuals)))):
            continue_fitting = False
        else:
            bad_peak_index = np.argmax(np.abs(residuals))

            array = np.delete(array, bad_peak_index, axis=0)
            b     = np.delete(b    , bad_peak_index, axis=0)

            good_indices = np.delete(good_indices, bad_peak_index)


    return coeffs, res, rank, s, good_indices


def make_coeff_matrix(normed_pixel, normed_orders, orders, m0, nord, npix, overlap_fit=True):
    """
    # Make 2D coefficient matricies for wavelength solution determination
    # As we only need to calculate the cofficients of the fit, we can precalculate the polynomial values at all orders and pixels and only multiply those with the coefficients
    # The matrix columns are sorted (0, 0), (0,1), ... , (0, npix), (1, 0), ... , (nord, npix)
    #
    #
    #
    #
    #
    #
    """

    #shortcut
    eval2d = np.polynomial.chebyshev.chebval2d

    order_divisor = 1. / np.add(m0, orders).astype(np.float64)

    columns = ()

    for n1 in range(nord+1):
        for n2 in range(1, npix+1):
            coeff_array = np.zeros((n1+1, n2+1))
            coeff_array[n1, n2] = 1
            columns+= (order_divisor * eval2d(normed_orders, normed_pixel, coeff_array), )

    array = np.column_stack(columns)

    if overlap_fit:
        b     = -1. * order_divisor.reshape(-1,1)
    else:
        b = np.zeros_like(normed_pixel, dtype=np.float64)

    return array, b

def getGlobalScale(reference_filename, ThAr_Peaks, WaveSolution, npools = None, niter_max=20):
    """
    # Calculate Global Scale of Wavelength solution. This is a global prefactor neded to convert the wavelenght solution to actual physical wavelengths
    #
    # :param reference_filename: str, filename of the reference file, needs to contain a list of all ThAr line wavelengths, one wavelength per line
    # :param ThAr_Peaks: ThArPeaks object, contains the information about all found ThAr peaks
    # :param WaveSolution: WavelengthSolution object, contains the wavelength solution from overlaps, but still needs global scale
    # :param niter_max: int, maximal number of fitting (default 20)
    #
    # :return best_k: float, global scale of wavelength solution
    """

    #load reference
    reference = np.genfromtxt(reference_filename)

    #sort reference
    reference = np.sort(reference)

    #interpolator to always use closest reference value
    reference_interpolator = interpolate.interp1d(reference, reference, kind='nearest', bounds_error=False, \
                                                  fill_value=(np.min(reference), np.max(reference)))

    #median wavelength steps in reference
    median_reference_steps = np.median(reference[1:] - reference[:-1])

    #create list with wavelengths of all peaks
    all_wave_peaks = []

    for ordernr in ThAr_Peaks.getOrders():
        Peaks = np.array(ThAr_Peaks[ordernr])

        Wave_Peaks = WaveSolution.eval_wavelengths(ordernr, Peaks)

        for p in Wave_Peaks:
            all_wave_peaks.append(p)

    all_wave_peaks = np.array(all_wave_peaks)
    all_wave_peaks = all_wave_peaks[all_wave_peaks > 0] #negative wavelengths are not physical

    #compare medians of reference and wavelength solution to get a first guess of global scale
    median_reference = np.nanmedian(reference)
    median_peaks     = np.nanmedian(all_wave_peaks)

    first_guess = median_reference / median_peaks

    # step size of brute force search
    brute_force_steps = first_guess * median_reference_steps / (median_reference * 10)

    #minimal and maximal limit of search
    #lower_limit  = first_guess / 1.25
    #higher_limit = first_guess * 1.25

    lower_limit  = np.min(reference) / median_peaks
    higher_limit = np.max(reference) / median_peaks


    fit_peaks = all_wave_peaks.copy()

    k_range = np.arange(lower_limit, higher_limit, brute_force_steps)

    #do brute force search

    if npools is None:
        npools = datashare.reduction_parameters.npools

    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
        try:
            args = [[all_wave_peaks, k, reference_interpolator, niter_max] for k in k_range]

            results = pool.map(_sigmaclip_globalscale, args)
        finally:
            pool.close()
            pool.join()

    residuals = []
    for r in results:
        residuals.append(r[0])

    #fit linear model to residuals to get background. Also set minimum of residuals to zero to allow a more stable fit
    residuals = np.array(residuals)

    residuals -= np.min(residuals)

    #best result so far is brute force iteration with least residuals
    best_k = k_range[np.argmin(residuals)]

    min_k = 0.999 * best_k
    max_k = 1.001 * best_k

    #second brute force run with smoother resolution:
    k_range = np.linspace(min_k, max_k, 1000)
    #do brute force search
    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
        try:
            args = [[all_wave_peaks, k, reference_interpolator, niter_max] for k in k_range]

            results = pool.map(_sigmaclip_globalscale, args)
        finally:
            pool.close()
            pool.join()


    fine_residuals = []
    use_inds       = []
    for r in results:
        fine_residuals.append(r[0])
        use_inds.append(r[1])

    # Also set minimum of residuals to zero to allow a more stable fit. No background substraction here as this is a very small region
    fine_residuals = np.array(fine_residuals)
    fine_residuals -= np.min(fine_residuals)

    #best result so far is brute force iteration with least residuals
    best_k    = k_range[np.argmin(fine_residuals)]
    best_inds = use_inds[np.argmin(fine_residuals)]

    min_k = 0.999 * best_k
    max_k = 1.001 * best_k


    fine_residual_peak_inds, _ = signal.find_peaks(fine_residuals)
    k_peaks = k_range[fine_residual_peak_inds]


    #get next previous and next peak in residuals. Fit only between those peaks (the current 'valley')
    try:
        min_k = np.max(k_peaks[k_peaks < best_k])
        max_k = np.min(k_peaks[k_peaks > best_k])
    except:
        min_k = 0.999 * best_k
        max_k = 1.001 * best_k

    #plot, if requested
    if datashare.reduction_parameters.plot_WaveGlobalScale:
        x = np.arange(lower_limit, higher_limit, brute_force_steps)
        #inds = np.where(np.logical_and(x > 200000, x < 260000))
        inds = np.arange(len(x))

        fig, ax = plt.subplots()

        ax.plot(x[inds], residuals[inds])
        fig.suptitle('Global Scale search for m0={}'.format(WaveSolution.m0))
        ax.set_xlabel('Global scale')
        ax.set_ylabel('Normed residuals')
        fig.tight_layout()


        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_GlobalScaleSearch_m0{}.png".format(WaveSolution.m0))
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('') #needed to show plot in Jupyter Notebook
            plt.show()

        plt.close()

    previous_guess = best_k

    #matches of all found peaks
    all_list_matches = reference_interpolator(previous_guess * all_wave_peaks)

    fit_peaks = all_wave_peaks.copy()

    # fit functions for least square optimization
    def fit_func(K, wavelengths, list_matches):
        res =  residuals_func(K, wavelengths, list_matches)
        res =  np.sum(np.square(res))/len(wavelengths)
        return res

    def residuals_func(K, wavelengths, list_matches):
        return (K * wavelengths - list_matches)


    unscaled_wavelengths = all_wave_peaks[best_inds]
    list_matches         = reference_interpolator(previous_guess * unscaled_wavelengths)

    # find residuals mimimum with least squares, do sigma clipping
    cont_fitting = True
    nfit = 0
    while cont_fitting:
        nfit += 1

        # fit via least squares
        result = optimize.minimize(fit_func, previous_guess, args=(unscaled_wavelengths, list_matches), bounds=[(min_k, max_k)]) #method='Nelder-Mead')

        # if fit went well: set guess to current result
        if result.success:
            previous_guess = float(result.x[0])

        residuals = np.abs(residuals_func(previous_guess, unscaled_wavelengths, list_matches))
        #rms       = np.sqrt(np.sum(np.square(residuals))/len(residuals))
        rms       = np.median(residuals)

        # do sigma clipping
        # break if to few peaks, no outliners anymore or reached maximum number of iterations
        if (len(fit_peaks) < len(all_wave_peaks)/2.) or (not np.any(residuals > 5 * rms)) or nfit >= niter_max:
            cont_fitting = False
            best_k       = previous_guess
        else:
            #update reference interpolator
            all_list_matches = reference_interpolator(previous_guess * all_wave_peaks)

            # remove bad indices
            all_residuals    = np.abs(residuals_func(previous_guess, all_wave_peaks, all_list_matches))

            good_indices = np.where(all_residuals <= 5 * rms)
            unscaled_wavelengths = all_wave_peaks[good_indices]
            list_matches = all_list_matches[good_indices]

    return best_k


def _sigmaclip_globalscale(args):
    all_wave_peaks, k, reference_interpolator, niter_max = args

    #do sigma clipping
    fit_peaks = all_wave_peaks.copy()

    use_inds = np.arange(len(fit_peaks))

    cont_fitting = True
    nfit         = 0

    #sigma clipping
    while cont_fitting:
        #residuals
        res = np.abs(k * fit_peaks[use_inds] - reference_interpolator(k * fit_peaks[use_inds]))
        #rms = np.sqrt(np.sum(np.square(residuals)))/len(residuals)
        rms = np.median(res)

        #print(residuals, rms)

        nfit += 1

        #break if not enough peaks anymore, no outliners anymore or reached niter_max
        if (len(use_inds) < len(all_wave_peaks)/2.) or (not np.any(res > 5 * rms)) or nfit > niter_max:
            return rms, use_inds
        else:
            #remove index with highest residual
            bad_indices = np.argsort(res)[-1:]
            use_inds = np.delete(use_inds, bad_indices)

def final_wavelength_fit(reference, ThAr_Peaks, WaveSolution, mjd, niter_max=20, pixshift=0, ordershift=0, minlines= None, fromOverlaps=False):
    """
    # Create final wavelength solution from initial wavelength solution
    # Wavelengths are interpolated using Chebychev polynomials
    #
    # :param reference: list or str, list of reference wavelengths or filename of reference file
    # :param ThAr_Peaks: ThArPeaks object, contains the information about all found ThAr peaks
    # :param WaveSolution: WavelengthSolution object, contains the wavelength solution from overlaps or a reference solution from another ThAr frame
    # :param mjd: float, MJD of exposure (mid of exposure)
    # :param niter_max: int, maximal number of fitting (default 20)
    # :param pixshift: int, shift pixels by this value (e.g. when transfering one wavelength solution at a different time to this one. Thermal instabilities will cause these temperature shifts). Default 0
    # :param ordershift: int, shift orders by this value (e.g. when using a wavelength fron another observation which had more or less detected orders). Default 0
    # :param minlines: minimal lines to fit. If None, will use 10 * (nord + 1) * (npix + 1) (Default None)
    # :param fromOverlaps: boolean, whether this method uses a wavelength solution from order overlaps. This leads to using other polynomial degrees as specified in the instruments object (default False)
    #
    # :return FinalWaveSolution: FinalWavelengthSolution object, contains the final wavelength solution
    """
    # Final wavelength fit

    #load reference list if reference is filename
    if type(reference) is str:
        reference = np.genfromtxt(reference)

    # make sure reference list is sorted
    reference = np.sort(np.unique(reference))

    #interpolator to find closest reference wavelength
    reference_interpolator = interpolate.interp1d(reference, reference, kind='nearest', bounds_error=False, \
                                                  fill_value=(np.min(reference), np.max(reference)))

    #median distance between two reference points
    median_reference_steps = np.median(reference[1:] - reference[:-1])

    #get peaks positions from overlap detection
    all_peaks, all_orders = ThAr_Peaks.allPeaks()

    # use parameters from initial wavelength solution
    #min_order = WaveSolution.min_order
    #max_order = WaveSolution.max_order
    min_order = np.min(all_orders)
    max_order = np.max(all_orders)
    min_pix   = WaveSolution.min_pix
    max_pix   = WaveSolution.max_pix
    m0        = WaveSolution.m0

    wav_tol   = datashare.instrument.wav_tol    #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

    #polynomial degrees to start at and to end at in pixel (dispersion) and order (cross-dispersion) direction
    if fromOverlaps:
        nord_init  = datashare.instrument.nord_init_final_fromoverlaps
        npix_init  = datashare.instrument.npix_init_final_fromoverlaps
        nord_final = datashare.instrument.nord_final_fromoverlaps
        npix_final = datashare.instrument.npix_final_fromoverlaps
    else:
        nord_init  = datashare.instrument.nord_init_final
        npix_init  = datashare.instrument.npix_init_final
        nord_final = datashare.instrument.nord_final
        npix_final = datashare.instrument.npix_final

    #minimal number of lines for fitting
    if minlines is None:
        minlines = 10 * (nord_final+1) * (npix_final+1)

    #normalize coordinates
    norm_peaks  = _normalize_coordinates(all_peaks, min_pix, max_pix)
    norm_orders = _normalize_coordinates(all_orders, min_order, max_order)

    #apply ordershift if given
    if ordershift != 0:
        m0        += ordershift
        min_order += ordershift
        max_order += ordershift

        #normalize again
        norm_orders = _normalize_coordinates(all_orders, min_order, max_order)

    #evalate wavelengths of initial wavelength solution
    if pixshift == 0 or np.isnan(pixshift) or pixshift is None:
        start_wavs = eval_wave_coeffs(WaveSolution.coeffs, m0, norm_orders, norm_peaks, all_orders)
    else:
        #apply pixel shift if given
        min_pix_shifted = min_pix + pixshift
        max_pix_shifted = max_pix + pixshift

        #normalize again
        norm_peaks_shifted  = _normalize_coordinates(all_peaks, min_pix_shifted, max_pix_shifted)

        #evaluate wavelengths
        start_wavs = eval_wave_coeffs(WaveSolution.coeffs, m0, norm_orders, norm_peaks_shifted, all_orders)


    #find peaks that have a close representative in the reference
    list_matches = reference_interpolator(start_wavs)
    is_close = np.asarray(np.abs(list_matches - start_wavs) < wav_tol)

    # start at middle order, then add orders from above and below
    middle_order = np.around(np.mean((max_order, min_order))).astype(int)



    #indices of peaks in middle order
    start_inds = np.array(np.logical_and(np.asarray(all_orders == middle_order), is_close).nonzero()).flatten()

    for i in range(1, datashare.instrument.final_wavelength_fit_number_orders_to_start//2 + 1):
        #add indicies of lower order
        new_order = middle_order - i
        new_inds = np.array(np.logical_and(np.asarray(all_orders == new_order), is_close).nonzero()).flatten()

        start_inds = np.concatenate((start_inds, new_inds))

        #add indicies of higher order
        new_order = middle_order + i
        new_inds = np.array(np.logical_and(np.asarray(all_orders == new_order), is_close).nonzero()).flatten()
        start_inds = np.concatenate((start_inds, new_inds))

    #if not enough peaks just use all peaks in that order, without the need of a close reference
    if len(start_inds) < 2 * (nord_init+1)*(npix_init+1):
        start_inds = np.where(all_orders == middle_order)[0]

        for i in range(1, datashare.instrument.final_wavelength_fit_number_orders_to_start//2 + 1):
            #add indicies of lower order
            new_order = middle_order - i
            new_inds = np.where(all_orders == new_order)[0]
            start_inds = np.concatenate((start_inds, new_inds))

            #add indicies of higher order
            new_order = middle_order + i
            new_inds = np.where(all_orders == new_order)[0]
            start_inds = np.concatenate((start_inds, new_inds))

    inds = np.sort(np.unique(start_inds))

    #fit functions
    #wavelength residuals, should be minimized. Nomrlalized by number of indices and summed
    def fit_func(coeffs, inds, weights, list_matches, nord, npix):
        coeffs = coeffs.reshape((nord+1, npix+1))

        wavelengths = eval_wave_coeffs(coeffs, m0, norm_orders[inds], norm_peaks[inds], all_orders[inds])

        res =  np.abs(wavelengths - list_matches[inds])

        res[np.where(weights[inds] == 0)] = 0

        res =  np.sqrt(np.mean(np.square(res))) #/len(inds)
        #res = np.median(res[res > 0])

        return res


    # same as above, but do not scale and sum residuals
    def residuals_func(coeffs, inds, weights, list_matches, nord, npix):
        coeffs = coeffs.reshape((nord+1, npix+1))

        wavelengths = eval_wave_coeffs(coeffs, m0, norm_orders[inds], norm_peaks[inds], all_orders[inds])

        res =  wavelengths - list_matches[inds]

        #res[np.where(weights == 0)] = 0

        return res


    #initial coefficients of wavelength solution
    init_coeffs = WaveSolution.coeffs[:nord_init+1, :npix_init+1]

    # keep polynomial degree fixed and add orders to the fit
    continue_adding = True

    #indices which orders we added last
    current_low_order  = middle_order - datashare.instrument.final_wavelength_fit_number_orders_to_start//2
    current_high_order = middle_order + datashare.instrument.final_wavelength_fit_number_orders_to_start//2


    last_coeffs = init_coeffs
    while continue_adding:
        continue_fitting = True

        wavs = eval_wave_coeffs(last_coeffs, m0, norm_orders, norm_peaks, all_orders)

        list_matches = reference_interpolator(wavs)

        #indices which peaks have a close reference companion and which peaks don't
        is_close = np.where(np.abs(list_matches - wavs) < wav_tol)
        not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)


        #set weights of peaks, which are not in the current orders, to zero
        weights = np.ones_like(norm_peaks)
        weights[np.where(np.logical_or(all_orders < min_order, all_orders > max_order))] = 0
        weights[not_close] = 0

        nfit = 0

        # do sigma clipping, iterative fitting
        while continue_fitting and nfit < niter_max:
            nfit += 1

            #optimize coefficients
            result = optimize.minimize(fit_func, last_coeffs.ravel(), args=(inds, weights, list_matches,nord_init, npix_init))

            #if successful fit: update coeffs
            if result.success:
                last_coeffs = result.x.reshape(nord_init+1, npix_init+1)

            #do sigma clipping
            residuals = np.abs(residuals_func(result.x, np.arange(len(norm_peaks)), weights, list_matches, nord_init, npix_init))

            res_median = np.median(residuals[residuals > 0])

            nr_valid_peaks = len(np.where(weights > 0)[0])

            wavs = eval_wave_coeffs(last_coeffs, m0, norm_orders, norm_peaks, all_orders)

            list_matches = reference_interpolator(wavs)

            #indices which peaks have a close reference companion and which peaks don't
            is_close = np.where(np.abs(list_matches - wavs) < wav_tol)[0]
            not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)[0]

            #print(m0, nfit, len(np.where(weights > 0)[0]), len(inds))
            #print(m0, nfit, len(inds), inds)

            #sort out bad peaks
            new_weights = np.ones_like(norm_peaks)
            new_weights[residuals > 5 * res_median] = 0
            new_weights[not_close] = 0


            # break if not enough lines, no bad lines or maximum iteration reached
            if (nr_valid_peaks < 2 * (nord_init +1) * (npix_init +1)) or (not np.any(residuals > 5 * res_median)) or nfit > niter_max or np.array_equal(weights, new_weights):
                continue_fitting = False

            else:
                weights = new_weights



        #now add new orders
        current_low_order  -= 1
        current_high_order += 1

        #all orders added
        if current_low_order < 0 and current_high_order > np.max(all_orders):
            continue_adding = False
            break

        #add peaks of low_order
        if current_low_order >= 0:
            #inds = np.append(np.intersect1d(np.where(all_orders == current_low_order)[0], is_close), inds)
            inds = np.append(np.where(all_orders == current_low_order)[0], inds)

        #add peaks of high_order
        if current_high_order <= np.max(all_orders):
            #inds = np.append(inds, np.intersect1d(np.where(all_orders == current_high_order)[0], is_close))
            inds = np.append(inds, np.where(all_orders == current_high_order)[0])

        #sort indices
        inds = np.sort(np.unique(inds))

    #now start increasing the polynomial degree
    initial_guess = last_coeffs

    continue_adding = True

    nord, npix = nord_init, npix_init

    while continue_adding:
        # add one degree by one
        # increase order (cross-dispersion) degrees and pixel (dispersion) degrees alternately

        # order degree maxed out -> increase pixel degree
        if nord == nord_final:
            #pixel degree also maxed out: break
            if npix == npix_final:
                continue_adding = False
                break
            else:
                #increase pixel degree
                npix += 1
                new_pix_order = True
        # pixel degree maxed out, but order degree isn't
        elif npix == npix_final:
            nord += 1
            new_pix_order = False
        # npix more often increased than nord -> increase nord
        elif nord - nord_init < npix - npix_init:
            nord += 1
            new_pix_order = False
        # else increase npix
        else:
            npix += 1
            new_pix_order = True

        new_coeffs = np.zeros((nord+1, npix+1))

        #depending on whether we added pixel or order degree we need to transfer the old coefficients differently to the new coefficient matrix
        if new_pix_order:
            new_coeffs[:, :-1] = last_coeffs
        else:
            new_coeffs[:-1, :] = last_coeffs

        last_coeffs = new_coeffs

        # again set weights for all irrelevant peaks to zero
        weights = np.ones_like(norm_peaks)
        weights[np.where(np.logical_or(all_orders < min_order, all_orders > max_order))] = 0

        not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)[0]

        weights[not_close] = 0

        continue_fitting = True

        inds = np.arange(len(norm_peaks))

        nfit = 0
        while continue_fitting:
            nfit += 1

            # optimize coefficients
            result = optimize.minimize(fit_func, last_coeffs.ravel(), args=(inds, weights, list_matches, nord, npix))#, method='Nelder-Mead')


            # if fit successful update coeffs
            if result.success:
                last_coeffs = result.x.reshape((nord+1, npix+1))

            # do sigma clipping
            residuals = np.abs(residuals_func(last_coeffs.flatten(), np.arange(len(norm_peaks)), weights, list_matches, nord, npix))

            res_median = np.median(residuals[residuals > 0])

            weights_not_null = np.where(weights > 0)[0]
            nr_valid_peaks = len(weights_not_null)

            wavs = eval_wave_coeffs(last_coeffs, m0, norm_orders, norm_peaks, all_orders)

            list_matches = reference_interpolator(wavs)

            #indices which peaks have a close reference companion and which peaks don't
            is_close  = np.where(np.abs(list_matches - wavs) <  wav_tol)[0]
            not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)[0]

            #sort out bad peaks
            new_weights = np.ones_like(norm_peaks)
            new_weights[residuals > 5 * res_median] = 0
            new_weights[not_close] = 0


            # break if not enough lines, no bad lines or maximum iteration reached
            if (nr_valid_peaks < 2 * (nord_init +1) * (npix_init +1)) or (not np.any(residuals[weights_not_null] > 2 * res_median)) or nfit > niter_max or np.array_equal(weights, new_weights):
                continue_fitting = False

            else:
                weights = new_weights

    #find indices which were used for wavelength solution and which weren't
    final_inds = np.where(weights > 0)[0]
    bad_inds   = np.where(weights <= 0)[0]

    #create residuals for both
    residuals = residuals_func(last_coeffs, final_inds, weights, list_matches, nord_final, npix_final)
    bad_residuals = residuals_func(last_coeffs, bad_inds, weights, list_matches, nord_final, npix_final)

    #evaluate wavelengths for both
    wavelengths = eval_wave_coeffs(last_coeffs, m0, norm_orders[final_inds], norm_peaks[final_inds], all_orders[final_inds])
    bad_wavelengths = eval_wave_coeffs(last_coeffs, m0, norm_orders[bad_inds], norm_peaks[bad_inds], all_orders[bad_inds])

    # only keep peaks which are within the wavelength range of the reference
    keep_inds = np.where((bad_wavelengths >= np.min(reference)) & (bad_wavelengths <= np.max(reference)))[0]

    bad_inds        = bad_inds[keep_inds]
    bad_wavelengths = bad_wavelengths[keep_inds]
    bad_residuals   = bad_residuals[keep_inds]

    #minimal / maximal useful orders, as below / above these orders the wavelength solution isn't well constrained
    min_useful_order = np.min(all_orders[final_inds]).astype(int)
    max_useful_order = np.max(all_orders[final_inds]).astype(int)

    found_lines = []
    for i, ind in enumerate(final_inds):
        found_lines.append(Spectra.MatchedLine(all_orders[ind], all_peaks[ind], reference_interpolator(wavelengths[i])))

    #plot, if requested
    if len(final_inds) > 0 and len(bad_inds) > 0 and datashare.reduction_parameters.plot_WavelengthFit:
        #fig, axs = plt.subplots(1,2, figsize=(16,9))
        #fig, axs = plt.subplots(1, 2, figsize=(10,5), dpi=300, width_ratios=[5,1])

        colors = ['blue', 'red', 'green', 'black', 'yellow']

        fig = plt.Figure(figsize=(30,6), dpi=300)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5,1])

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1], sharey=plt.subplot(gs[0]))]

        for i, order in enumerate(np.unique(all_orders[final_inds])):
            order_inds = np.where(all_orders[final_inds] == order)[0]

            bad_order_inds = np.where(all_orders[bad_inds] == order)[0]

            axs[0].scatter(wavelengths[order_inds], residuals[order_inds], color=colors[i%len(colors)])
            #axs[1].scatter(bad_wavelengths[bad_order_inds], bad_residuals[bad_order_inds], color=colors[i%len(colors)])

        hist, bins, patches = axs[1].hist(residuals, bins=31, orientation='horizontal')

        bins = bins[:-1] + (bins[1] - bins[0])/2.

        #fit gaussian
        #guess parameters: A, x0, sigma of gaussian
        p0 = [np.max(hist), 0, (np.max(residuals) - np.min(residuals))/10.]

        try:
            popt, _ = optimize.curve_fit(FitFunctions.gaussian, bins, hist, p0=p0)

            x = np.linspace(0.8 * np.min(residuals), 0.8 * np.max(residuals), 200)
            hist_fit = FitFunctions.gaussian(x, *popt)

            axs[1].plot(hist_fit, x)

        except:
            pass


        axs[0].hlines(np.median(residuals), np.min(wavelengths), np.max(wavelengths), label='Median', linestyles='dashed')
        #axs[1].hlines(np.median(bad_residuals), np.min(bad_wavelengths), np.max(bad_wavelengths), label='Median', linestyles='dashed')

        axs[0].legend()
        #axs[1].legend()

        #axs[0].set_title('Accepted lines')
        #axs[1].set_title('Declined lines')
        axs[0].set_title('Calibration residuals')

        axs[0].set_xlabel(r'$\lambda$ in $\AA$')
        #axs[1].set_xlabel(r'$\lambda$ in $\AA$')
        axs[0].set_ylabel(r'residuals in $\AA$')

        axs[0].grid()
        axs[1].grid()

        fig.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_m0{}_WavelengthFit.png".format(WaveSolution.m0))
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')       #needed to show plot in Jupyter Notebook
            plt.show()

        plt.close()

    # calculate RMS in km/s (c is in km/s)
    rms = np.sqrt(np.sum(np.square(Spectra.Constants.c * residuals/wavelengths))) / len(final_inds)   #rms in km/s

    # initiate FinalWavelengthSolution object
    FinalWaveSolution = Spectra.FinalWavelengthSolution(last_coeffs, WaveSolution.m0, reference, found_lines, \
                                                        (min_order, max_order), (min_useful_order, max_useful_order),  \
                                                        (min_pix, max_pix), ThAr_Peaks, rms, mjd)


    return FinalWaveSolution

#TODO: Fertig umstellen mit beiden Refernzlisten
def wavelength_fit_referencelist(referencelist, complete_reference, thar_spectrum, mjd, niter_max=20, minlines= None):
    """
    # Create final wavelength solution from reference list with positions and wavelengths of reference peaks
    # Wavelengths are interpolated using Chebychev polynomials
    #
    # :param referencelist: list or str, instrument specific reference list with peak positions and corresponding wavelengths for each order or filename of that file
    # :param complete_reference: list or str, list of all reference wavelengths (without pixel or order information) or filename of reference file
    # :param thar_spectrum: RawSpectrum object, extracted ThAr spectrum
    # :param mjd: float, MJD of exposure (mid of exposure)
    # :param niter_max: int, maximal number of fitting (default 20)
    # :param minlines: minimal lines to fit. If None, will use 10 * (nord + 1) * (npix + 1) (Default None)
    #
    # :return FinalWaveSolution: FinalWavelengthSolution object, contains the final wavelength solution
    """
    # Final wavelength fit

    #load reference list if reference is filename
    #reference[:, 0] are the physical diffraction orders
    #reference[:, 1] are pixels
    #reference[:, 2] are wavelengths (in A)
    if type(referencelist) is str:
        referencelist = np.genfromtxt(referencelist, dtype=float, delimiter=';', comments='#')

    assert len(referencelist.shape) == 2
    assert referencelist.shape[1] == 3


    if type(complete_reference) is str:
        complete_reference = np.genfromtxt(complete_reference, dtype=float)

    assert len(complete_reference.shape) == 1



    reference_orders = referencelist[:, 0].astype(int)
    reference_pixels = referencelist[:, 1]
    reference_waves  = referencelist[:, 2]

    m0, median_shift = m0FromReference(thar_spectrum, referencelist)

    #median distance between two reference points
    median_reference_steps = np.median(np.abs(np.diff(reference_waves)))

    all_widths     = _getwidths(thar_spectrum, maxheight=0.9 * datashare.camera.get_maxcount(thar_spectrum.header))
    _,  ThAr_Peaks = _getPeaks(thar_spectrum, all_widths, npools = 1, all_threshold=2, maxheight=0.9 * datashare.camera.get_maxcount(thar_spectrum.header))

    #get peaks positions from overlap detection
    all_peaks, all_orders = ThAr_Peaks.allPeaks()

    #interpolator to find closest reference wavelength
    reference_interpolator = interpolate.interp1d(complete_reference, complete_reference, kind='nearest', bounds_error=False, \
                                                  fill_value=(np.min(reference_waves), np.max(reference_waves)))


    #min_order = WaveSolution.min_order
    #max_order = WaveSolution.max_order
    min_order = 0
    max_order = thar_spectrum.nr_of_orders() - 1
    min_pix   = 0
    max_pix   = np.max([np.max(thar_spectrum[ordnr].pixels) for ordnr in range(thar_spectrum.nr_of_orders())])

    wav_tol   = datashare.instrument.wav_tol    #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

    #polynomial degrees to start at and to end at in pixel (dispersion) and order (cross-dispersion) direction
    nord_init  = datashare.instrument.nord_init_final
    npix_init  = datashare.instrument.npix_init_final
    nord_final = datashare.instrument.nord_final
    npix_final = datashare.instrument.npix_final

    #minimal number of lines for fitting
    if minlines is None:
        minlines = 10 * (nord_final+1) * (npix_final+1)

    #normalize coordinates
    norm_peaks  = _normalize_coordinates(all_peaks, min_pix, max_pix)
    norm_orders = _normalize_coordinates(all_orders, min_order, max_order)

    #evaluate initial wavelengths to start with
    start_wavs = np.zeros_like(all_peaks)

    for ordnr in range(min_order , max_order +1):
        ref_inds  = np.where(reference_orders == ordnr + m0)[0]
        peak_inds = np.where(all_orders == ordnr)

        norm_ref_peaks  = _normalize_coordinates(reference_pixels[ref_inds] + median_shift, min_pix, max_pix)

        if len(norm_ref_peaks) > npix_init:
            temp_coeffs = np.polynomial.chebyshev.chebfit(norm_ref_peaks, reference_waves[ref_inds], deg=npix_init)

            start_wavs[peak_inds] = np.polynomial.chebyshev.chebval(norm_peaks[peak_inds], temp_coeffs)

    #find peaks that have a close representative in the reference
    list_matches = reference_interpolator(start_wavs)
    is_close  = np.asarray(np.abs(list_matches - start_wavs) < wav_tol)
    not_close = np.asarray(np.abs(list_matches - start_wavs) >= wav_tol)

    # start at middle order, then add orders from above and below
    middle_order = np.around(np.mean((max_order, min_order))).astype(int)


    #indices of peaks in middle order
    start_inds = np.array(np.logical_and(np.asarray(all_orders == middle_order), is_close).nonzero()).flatten()

    for i in range(1, datashare.instrument.final_wavelength_fit_number_orders_to_start//2 + 1):
        #add indicies of lower order
        new_order = middle_order - i
        new_inds = np.array(np.logical_and(np.asarray(all_orders == new_order), is_close).nonzero()).flatten()

        start_inds = np.concatenate((start_inds, new_inds))

        #add indicies of higher order
        new_order = middle_order + i
        new_inds = np.array(np.logical_and(np.asarray(all_orders == new_order), is_close).nonzero()).flatten()
        start_inds = np.concatenate((start_inds, new_inds))

    #if not enough peaks just use all peaks in that order, without the need of a close reference
    if len(start_inds) < (nord_init+1)*(npix_init+1):
        start_inds = np.where(all_orders == middle_order)[0]

        for i in range(1, datashare.instrument.final_wavelength_fit_number_orders_to_start//2 + 1):
            #add indicies of lower order
            new_order = middle_order - i
            new_inds = np.where(all_orders == new_order)[0]
            start_inds = np.concatenate((start_inds, new_inds))

            #add indicies of higher order
            new_order = middle_order + i
            new_inds = np.where(all_orders == new_order)[0]
            start_inds = np.concatenate((start_inds, new_inds))

    inds = np.sort(np.unique(start_inds))

    #fit functions
    #wavelength residuals, should be minimized. Nomrlalized by number of indices and summed
    def fit_func(coeffs, inds, weights, list_matches, nord, npix):
        coeffs = coeffs.reshape((nord+1, npix+1))

        wavelengths = eval_wave_coeffs(coeffs, m0, norm_orders[inds], norm_peaks[inds], all_orders[inds])

        res =  np.abs(wavelengths - list_matches[inds])

        res[np.where(weights[inds] == 0)] = 0

        res =  np.sqrt(np.sum(np.square(res))) /len(inds)
        #res = np.median(res[res > 0])

        return res


    # same as above, but do not scale and sum residuals
    def residuals_func(coeffs, inds, weights, list_matches, nord, npix):
        coeffs = coeffs.reshape((nord+1, npix+1))

        wavelengths = eval_wave_coeffs(coeffs, m0, norm_orders[inds], norm_peaks[inds], all_orders[inds])

        res =  wavelengths - list_matches[inds]

        #res[np.where(weights == 0)] = 0

        return res

    #VERY rough first estimate of coeffs
    min_wavs_per_order = []
    max_wavs_per_order = []

    for ordnr in range(min_order + m0, max_order + m0 +1):
        temp_inds = np.where(reference_orders == ordnr)[0]

        if len(inds) > 0:
            min_wavs_per_order.append(np.min(reference_waves[temp_inds]))
            max_wavs_per_order.append(np.max(reference_waves[temp_inds]))

    min_wavs_per_order = np.array(min_wavs_per_order)
    max_wavs_per_order = np.array(max_wavs_per_order)

    average_m = m0 + (max_order - min_order) // 2

    guess_coeffs      = np.zeros(shape=(nord_init+1, npix_init+1))
    guess_coeffs[0,0] = np.min(min_wavs_per_order) * average_m
    guess_coeffs[1,0] = np.median(np.diff(min_wavs_per_order)) * average_m
    guess_coeffs[0,1] = np.median(max_wavs_per_order - min_wavs_per_order) * average_m

    #print(m0, median_shift)
    #print(start_wavs, list_matches, is_close)

    #first rough fit
    weights = np.ones_like(norm_peaks)
    weights[np.where(np.logical_or(all_orders < min_order, all_orders > max_order))] = 0
    weights[not_close] = 0

    result = optimize.minimize(fit_func, guess_coeffs.ravel(), args=(np.arange(len(list_matches)), weights, list_matches, nord_init, npix_init))

    #if successful fit: update coeffs
    if result.success:
        init_coeffs = result.x.reshape(nord_init+1, npix_init+1)
    else:   #try to avoid this! Might break the routine
        init_coeffs = guess_coeffs

    #init_coeffs = guess_coeffs

    # keep polynomial degree fixed and add orders to the fit
    continue_adding = True

    #indices which orders we added last
    current_low_order  = middle_order - datashare.instrument.final_wavelength_fit_number_orders_to_start//2
    current_high_order = middle_order + datashare.instrument.final_wavelength_fit_number_orders_to_start//2


    last_coeffs = init_coeffs
    while continue_adding:
        continue_fitting = True

        wavs = eval_wave_coeffs(last_coeffs, m0, norm_orders, norm_peaks, all_orders)

        list_matches = reference_interpolator(wavs)

        #indices which peaks have a close reference companion and which peaks don't
        is_close = np.where(np.abs(list_matches - wavs) < wav_tol)
        not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)


        #set weights of peaks, which are not in the current orders, to zero
        weights = np.ones_like(norm_peaks)
        weights[np.where(np.logical_or(all_orders < min_order, all_orders > max_order))] = 0
        weights[not_close] = 0

        nfit = 0

        # do sigma clipping, iterative fitting
        while continue_fitting and nfit < niter_max:
            nfit += 1

            #optimize coefficients
            result = optimize.minimize(fit_func, last_coeffs.ravel(), args=(inds, weights, list_matches, nord_init, npix_init))

            #if successful fit: update coeffs
            if result.success:
                last_coeffs = result.x.reshape(nord_init+1, npix_init+1)


            #do sigma clipping
            residuals = np.abs(residuals_func(result.x, np.arange(len(norm_peaks)), weights, list_matches, nord_init, npix_init))

            res_median = np.median(residuals[residuals > 0])

            nr_valid_peaks = len(np.where(weights > 0))

            wavs = eval_wave_coeffs(last_coeffs, m0, norm_orders, norm_peaks, all_orders)

            list_matches = reference_interpolator(wavs)

            #indices which peaks have a close reference companion and which peaks don't
            is_close = np.where(np.abs(list_matches - wavs) < wav_tol)
            not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)

            # break if not enough lines, no bad lines or maximum iteration reached
            if (nr_valid_peaks < 2 * (nord_init +1) * (npix_init +1)) or (not np.any(residuals > 5 * res_median)) or nfit > niter_max or len(is_close) <= minlines:
                continue_fitting = False

            else:
                #sort out bad peaks
                weights = np.ones_like(norm_peaks)
                weights[residuals > 5 * res_median] = 0
                weights[not_close] = 0



        #now add new orders
        current_low_order  -= 1
        current_high_order += 1

        #all orders added
        if current_low_order < 0 and current_high_order > np.max(all_orders):
            continue_adding = False
            break

        #add peaks of low_order
        if current_low_order >= 0:
            inds = np.append(np.intersect1d(np.where(all_orders == current_low_order)[0], is_close), inds)

        #add peaks of high_order
        if current_high_order <= np.max(all_orders):
            inds = np.append(inds, np.intersect1d(np.where(all_orders == current_high_order)[0], is_close))

        #sort indices
        inds = np.sort(np.unique(inds))


    #now start increasing the polynomial degree
    initial_guess = last_coeffs

    continue_adding = True

    nord, npix = nord_init, npix_init

    while continue_adding:
        # add one degree by one
        # increase order (cross-dispersion) degrees and pixel (dispersion) degrees alternately

        # order degree maxed out -> increase pixel degree
        if nord == nord_final:
            #pixel degree also maxed out: break
            if npix == npix_final:
                continue_adding = False
                break
            else:
                #increase pixel degree
                npix += 1
                new_pix_order = True
        # pixel degree maxed out, but order degree isn't
        elif npix == npix_final:
            nord += 1
            new_pix_order = False
        # npix more often increased than nord -> increase nord
        elif nord - nord_init < npix - npix_init:
            nord += 1
            new_pix_order = False
        # else increase npix
        else:
            npix += 1
            new_pix_order = True

        new_coeffs = np.zeros((nord+1, npix+1))

        #depending on whether we added pixel or order degree we need to transfer the old coefficients differently to the new coefficient matrix
        if new_pix_order:
            new_coeffs[:, :-1] = last_coeffs
        else:
            new_coeffs[:-1, :] = last_coeffs

        last_coeffs = new_coeffs

        # again set weights for all irrelevant peaks to zero
        weights = np.ones_like(norm_peaks)
        weights[np.where(np.logical_or(all_orders < min_order, all_orders > max_order))] = 0

        not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)[0]

        weights[not_close] = 0

        continue_fitting = True

        inds = np.arange(len(norm_peaks))

        nfit = 0
        while continue_fitting:
            nfit += 1

            # optimize coefficients
            result = optimize.minimize(fit_func, last_coeffs.ravel(), args=(inds, weights, list_matches, nord, npix))#, method='Nelder-Mead')


            # if fit successful update coeffs
            if result.success:
                last_coeffs = result.x.reshape((nord+1, npix+1))

            # do sigma clipping
            residuals = np.abs(residuals_func(last_coeffs.flatten(), np.arange(len(norm_peaks)), weights, list_matches, nord, npix))

            res_median = np.median(residuals[residuals > 0])

            weights_not_null = np.where(weights > 0)[0]
            nr_valid_peaks = len(weights_not_null)
            #print(nr_valid_peaks)

            wavs = eval_wave_coeffs(last_coeffs, m0, norm_orders, norm_peaks, all_orders)

            list_matches = reference_interpolator(wavs)

            #indices which peaks have a close reference companion and which peaks don't
            is_close  = np.where(np.abs(list_matches - wavs) <  wav_tol)[0]
            not_close = np.where(np.abs(list_matches - wavs) >= wav_tol)[0]

            # break if not enough lines, no bad lines or maximum iteration reached
            if (nr_valid_peaks < 2 * (nord_init +1) * (npix_init +1)) or (not np.any(residuals[weights_not_null] > 2 * res_median)) or nfit > niter_max or len(is_close) <= minlines:
                continue_fitting = False

            else:
                #sort out bad peaks
                weights = np.ones_like(norm_peaks)
                weights[residuals > 5 * res_median] = 0
                weights[not_close] = 0

    #find indices which were used for wavelength solution and which weren't
    final_inds = np.where(weights > 0)[0]
    bad_inds   = np.where(weights <= 0)[0]


    #create residuals for both
    residuals = residuals_func(last_coeffs, final_inds, weights, list_matches, nord_final, npix_final)
    bad_residuals = residuals_func(last_coeffs, bad_inds, weights, list_matches, nord_final, npix_final)

    #evaluate wavelengths for both
    wavelengths = eval_wave_coeffs(last_coeffs, m0, norm_orders[final_inds], norm_peaks[final_inds], all_orders[final_inds])
    bad_wavelengths = eval_wave_coeffs(last_coeffs, m0, norm_orders[bad_inds], norm_peaks[bad_inds], all_orders[bad_inds])

    # only keep peaks which are within the wavelength range of the reference
    keep_inds = np.where((bad_wavelengths >= np.min(complete_reference)) & (bad_wavelengths <= np.max(complete_reference)))[0]

    bad_inds        = bad_inds[keep_inds]
    bad_wavelengths = bad_wavelengths[keep_inds]
    bad_residuals   = bad_residuals[keep_inds]

    #minimal / maximal useful orders, as below / above these orders the wavelength solution isn't well constrained
    min_useful_order = np.min(all_orders[final_inds]).astype(int)
    max_useful_order = np.max(all_orders[final_inds]).astype(int)

    found_lines = []
    for i, ind in enumerate(final_inds):
        found_lines.append(Spectra.MatchedLine(all_orders[ind], all_peaks[ind], reference_interpolator(wavelengths[i])))

    #plot, if requested
    if len(final_inds) > 0 and datashare.reduction_parameters.plot_WavelengthFit:
        #fig, axs = plt.subplots(1,2, figsize=(16,9))
        #fig, axs = plt.subplots(1, 2, figsize=(10,5), dpi=300, width_ratios=[5,1])

        colors = ['blue', 'red', 'green', 'black', 'yellow']

        fig = plt.Figure(figsize=(30,6), dpi=300)
        gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5,1])

        axs = [plt.subplot(gs[0]), plt.subplot(gs[1], sharey=plt.subplot(gs[0]))]

        for i, order in enumerate(np.unique(all_orders[final_inds])):
            order_inds = np.where(all_orders[final_inds] == order)[0]

            bad_order_inds = np.where(all_orders[bad_inds] == order)[0]

            axs[0].scatter(wavelengths[order_inds], residuals[order_inds], color=colors[i%len(colors)])

        hist, bins, patches = axs[1].hist(residuals, bins=31, orientation='horizontal')

        bins = bins[:-1] + (bins[1] - bins[0])/2.

        #fit gaussian
        #guess parameters: A, x0, sigma of gaussian
        p0 = [np.max(hist), 0, (np.max(residuals) - np.min(residuals))/10.]

        popt, _ = optimize.curve_fit(FitFunctions.gaussian, bins, hist, p0=p0)

        x = np.linspace(0.8 * np.min(residuals), 0.8 * np.max(residuals), 200)
        hist_fit = FitFunctions.gaussian(x, *popt)

        axs[1].plot(hist_fit, x)

        axs[0].hlines(np.median(residuals), np.min(wavelengths), np.max(wavelengths), label='Median', linestyles='dashed')
        #axs[1].hlines(np.median(bad_residuals), np.min(bad_wavelengths), np.max(bad_wavelengths), label='Median', linestyles='dashed')

        axs[0].legend()
        #axs[1].legend()

        #axs[0].set_title('Accepted lines')
        #axs[1].set_title('Declined lines')
        axs[0].set_title('Calibration residuals')

        axs[0].set_xlabel(r'$\lambda$ in $\AA$')
        #axs[1].set_xlabel(r'$\lambda$ in $\AA$')
        axs[0].set_ylabel(r'residuals in $\AA$')

        axs[0].grid()
        axs[1].grid()

        fig.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_WavelengthFit.png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')       #needed to show plot in Jupyter Notebook
            plt.show()

        plt.close()

    # calculate RMS in km/s (c is in km/s)
    rms = np.sqrt(np.sum(np.square(Spectra.Constants.c * residuals/wavelengths))) / len(final_inds)   #rms in km/s

    # initiate FinalWavelengthSolution object
    FinalWaveSolution = Spectra.FinalWavelengthSolution(last_coeffs, m0, referencelist, found_lines, \
                                                        (min_order, max_order), (min_useful_order, max_useful_order),  \
                                                        (min_pix, max_pix), ThAr_Peaks, rms, mjd)


    return FinalWaveSolution

def  _getPeaks(spectrum, all_widths, npools=None, overlap_threshold=5, all_threshold=2, maxheight=np.inf):
    """
    #   Get all ThAr peaks in a ThAr spectrum
    #
    #   :param spectrum: Spectra.RawSpectrum object, contains the raw (not calibrated) ThAr spectrum
    #   :param all_widths: numpy array or list of floats, contains the approximate width (FWHM) of the peaks for each order. Needs to be same length as number of orders
    #   :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    #   :param overlap_threshold: float, minimal SNR of peak which will be used to identify overlaps (default 5)
    #   :param all_threshold: float, minimal SNR of peak to be detected at all. Can use less SNR here (default 2)
    #   :param maxheight: float, maximal peak height to filter out saturated peaks (default np.inf, no filtering)
    #
    #   :return overlap_peaks: Spectra.ThArPeaks object, contains all peaks we can use to find overlaps
    #   :return all_peaks: Spectra.ThArPeaks object, contains all peaks
    """
    if spectrum.nr_of_orders() != len(all_widths):
        raise ValueError('Number of orders and number of widths don\'t match!')

    #go through orders
    #this might be called inside another subprocess. It is not allowed to create another subprocess inside a subprocess, so we need to do it in a loop. Make sure, that npools == 1 if calling this method in a subprocess

    if npools is None:
        npools = datashare.reduction_parameters.npools

    if npools <= 1:
        results = []

        args = [[ordernr, spectrum, all_widths, overlap_threshold, all_threshold, maxheight] for ordernr in range(spectrum.nr_of_orders())]

        for arg in args:
            results.append(_getPeaks_singleOrder(arg))

    else:
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            try:
                args = [[ordernr, spectrum, all_widths, overlap_threshold, all_threshold, maxheight] for ordernr in range(spectrum.nr_of_orders())]

                results = pool.map(_getPeaks_singleOrder, args)
            finally:
                pool.close()
                pool.join()


    #create ThArPeaks objects
    overlap_peaks = Spectra.ThArPeaks()
    all_peaks     = Spectra.ThArPeaks()

    for ordernr in range(spectrum.nr_of_orders()):
        ordernr, overlap_centers, overlap_errs, all_centers, all_errs = results[ordernr]

        if overlap_centers is not None:
            overlap_peaks.addPeaks(ordernr, overlap_centers, overlap_errs)
        if all_centers is not None:
            all_peaks.addPeaks(ordernr, all_centers, all_errs)


    #set widths
    overlap_peaks.setWidths(all_widths)
    all_peaks.setWidths(all_widths)

    return overlap_peaks, all_peaks


def _getPeaks_singleOrder(args):
    ordernr, spectrum, all_widths, overlap_threshold, all_threshold, maxheight = args


    order = spectrum[ordernr]
    width = all_widths[ordernr]

    #test different widths as peaks might not have the same width. find_ThAr_peaks will search for peaks with all widths
    widths = [0.5 * width, 0.66 * width, 0.75 * width, width, 1.25 * width, 1.33 * width, 1.5 * width]

    #find peaks used for overlaps
    if overlap_threshold > 0:
        overlap_centers, overlap_errs = ThAr_Peak_detection.find_ThAr_peaks(order, width=widths, maxheight=maxheight, threshold=overlap_threshold)
    else:
        overlap_centers, overlap_errs = None, None

    #find all peaks
    if all_threshold > 0:
        all_centers, all_errs  = ThAr_Peak_detection.find_ThAr_peaks(order, width=widths, maxheight=maxheight, threshold=all_threshold)
    else:
        all_centers, all_errs = None, None

    return [ordernr, overlap_centers, overlap_errs, all_centers, all_errs]


def _getwidths(spectrum, maxheight = np.inf):
    """
    # Get appriximate width (FWHM) of ThAr peaks in spectrum
    #
    # :param spectrum: Spectra.RawSpectrum object, contains the raw (not calibrated) ThAr spectrum
    # :param maxheight: float, maximal height of peaks (to avoid saturated peaks) (default np.inf)
    #
    # :return all_widths: numpy array of floats, width of ThAr peaks for each spectral order
    """
    all_widths = []
    #go through orders
    for order_nr in range(spectrum.nr_of_orders()):
        order = spectrum[order_nr]

        #get width
        width = ThAr_Peak_detection.get_ThAr_peak_width(order, threshold=25, maxheight=maxheight)

        if not (np.isnan(width) or width is None):
            all_widths.append(width)
        else:
            all_widths.append(np.nan)


    all_widths = np.array(all_widths)

    #set width to median of all widths if width of this order is nan
    all_widths[np.isnan(all_widths)] = np.nanmedian(all_widths)

    return all_widths

def _getOverlaps(spectrum, overlap_peaks, npools=None, nord=2, pixel_tol=1, rel_height_diff=0.2, maxinds_blue = 25, maxinds_red = 20, pixfraction_blue = 0.75, pixfraction_red = 0.6):
    """
    # Get all overlaps between orders in a raw ThAr spectrum
    # Use multithreading to increase speed
    #
    # :param spectrum: Spectra.RawSpectrum object, contains the raw (not calibrated) ThAr spectrum
    # :param overlap_peaks: Spectra.ThArPeaks object, contains all peaks we can use to find overlaps
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param nord: int, polynomial degree of interpolating between overlaps (default 2)
    # :param pixel_tol: int, maximal difference between original and interpolation to detect lines (default 1)
    # :param rel_height_diff: float, maximal accepted relative height difference to match peaks. Default is 0.2, but you might to increase this value if no flat normalization is applied
    # :param maxinds_blue: int, maximal number of peaks in the first (blue) order. Too high numbers lead to a very high computational cost (default 25)
    # :param maxinds_red: int, maximal number of peaks in the second (red) order. Too high numbers lead to a very high computational cost (default 20)
    # :param pixfraction_blue: float, between 0 and 1, fraction of all pixels used for search of overlaps on the blue order. E.g. 0.5 will use the most right half of the order (default 0.75)
    # :param pixfraction_red: float, between 0 and 1, fraction of all pixels used for search of overlaps on the red order. E.g. 0.5 will use the most left half of the order (default 0.6)
    #
    # :return Overlaps: Spectra.Overlaps object, contains all found overlaps
    """

    #initialize object
    Overlaps = Spectra.Overlaps()

    #invalid value, return to default
    if pixfraction_blue < 0 or pixfraction_blue > 1:
        pixfraction_blue = 0.75

    #invalid value, return to default
    if pixfraction_red < 0 or pixfraction_red > 1:
        pixfraction_red = 0.6

    nr_of_pixels = len(spectrum[0].pixels)

    maxpix_blue = (1. - pixfraction_blue) * nr_of_pixels
    maxpix_red  = pixfraction_red  * nr_of_pixels

    if npools is None:
        npools = datashare.reduction_parameters.npools

    #go through orders
    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
        try:
            args = [[ordnr, spectrum, overlap_peaks, nord, pixel_tol, rel_height_diff, maxinds_blue, maxinds_red, maxpix_blue, maxpix_red] for ordnr in range(spectrum.nr_of_orders() -1)]

            results = pool.map(_find_single_order_overlaps, args)
        finally:
            pool.close()
            pool.join()


    for ordnr in range(spectrum.nr_of_orders() -1):
        overlaps = results[ordnr]

        if overlaps is None:
            continue

        for ind in range(len(overlaps[0])):
            Overlaps.addOverlap(Spectra.Overlap(ordnr, ordnr +1, overlaps[0][ind], overlaps[1][ind]))

    return Overlaps


def _find_single_order_overlaps(args):
    ordnr, spectrum, overlap_peaks, nord, pixel_tol, rel_height_diff, maxinds_blue, maxinds_red, maxpix_blue, maxpix_red = args

    peaks1 = overlap_peaks[ordnr]   #peaks of left order
    peaks2 = overlap_peaks[ordnr+1] #peaks of right order

    peaks1 = np.round(peaks1).astype(int)
    peaks2 = np.round(peaks2).astype(int)

    # skip order if no peaks were found
    if len(peaks1) == 0 or len(peaks2) == 0:
        return None

    #limit left order to first maxpix_red pixels and right order to pixels above maxpix_blue (left and right edges)
    peaks1 = peaks1[peaks1 < maxpix_red]
    peaks2 = peaks2[peaks2 > maxpix_blue]

    #get orders
    order1 = spectrum[ordnr]
    order2 = spectrum[ordnr+1]

    #get only most left / right inds, else find_order_overlap is computationally very expensive
    maxinds1 = np.min((maxinds_red, len(peaks1))).astype(int)
    maxinds2 = np.min((maxinds_blue, len(peaks2))).astype(int)

    peak_height1 = order1.flux[peaks1] #/ order1.errors[peaks1]
    peak_height2 = order2.flux[peaks2] #/ order2.errors[peaks2]

    peaks_SNRsorted1 = peaks1[np.argsort(peak_height1)[::-1]]  #highest peak first
    peaks_SNRsorted2 = peaks2[np.argsort(peak_height2)[::-1]]

    ind1 = np.sort(peaks_SNRsorted1[:maxinds1])
    ind2 = np.sort(peaks_SNRsorted2[:maxinds2])

    #find overlaps
    res, overlaps = find_order_overlap(order1, order2, ind1, ind2, scale_range=(-1e5, 1e5), nord=nord, \
                                        pixel_tol=pixel_tol, rel_height_diff=rel_height_diff)

    # skip order if no good overlap was found
    if res is None or overlaps is None:
        return None


    #plot, if requested
    if datashare.reduction_parameters.plot_WaveOverlaps:
        red_peaks  = overlaps[0]
        blue_peaks = overlaps[1]


        #cheb = np.polynomial.Polynomial(res)
        cheb = np.polynomial.chebyshev.Chebyshev(res)


        inds1 = np.where(order1.pixels < 2000)[0]
        inds2 = np.where(order2.pixels > 1500)[0]

        plt.clf()
        fig, axs = plt.subplots(figsize=(10,6), dpi=300)
        axs.plot(cheb(order1.pixels[inds1]), order1.flux[inds1], linewidth=0.75 , label= 'Order {}'.format(ordnr))
        axs.plot(order2.pixels[inds2], order2.flux[inds2], linewidth=0.75 ,label= 'Order {}'.format(ordnr+1))
        axs.scatter(cheb(red_peaks), order1.flux[red_peaks.astype(int)])
        axs.scatter(blue_peaks, order2.flux[blue_peaks.astype(int)])

        ymax = np.max((np.max(order1.flux[order1.flux/order1.errors > 20]), \
                        np.max(order2.flux[order2.flux/order2.errors > 20])))

        axs.legend()
        axs.set_title('Overlaps of order {} and {}'.format(ordnr, ordnr+1))
        axs.set_xlabel('Pixels')
        axs.set_ylabel('Flux in a.u.')

        axs.set_ylim((-0.1 * ymax, 1.1 * ymax))

        if datashare.reduction_parameters.save_plots:
            filename = plot_utilities.getnextfilename(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_Overlaps_order{}".format(ordnr), ".png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')       #needed to show plot in Jupyter Notebook
            plt.show()

        plt.close()

    return overlaps




def WaveSolutionFromUnknownm0(spectrum, reference_filename, npools=None):
    """
    # Find a wavelength solution from a ThAr calibration spectrum without knowing m0 (the physical diffraction order number of the first software order)
    # This method can test multiple m0 candidates simultaneously to increase speed
    #
    # :param spectrum: Spectra.RawSpectrum object, contains the raw (not calibrated) ThAr spectrum
    # :param reference_filename: str, filename of reference wavelenght file. File needs to contain one reference wavelength in angstrom per line
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :return m0: int, the physical diffraction order number of the first software order
    # :return WaveSolution: Spectra.WavelengthSolution object, initial wavelength solution of the spectrum
    """
    #this is independent of m0

    if npools is None:
        npools = datashare.reduction_parameters.npools

    #get peak widths and peaks from spectum
    all_widths               = _getwidths(spectrum, maxheight=0.9 * datashare.camera.get_maxcount(spectrum.header))
    overlap_peaks, all_peaks = _getPeaks(spectrum, all_widths, npools = npools, overlap_threshold=5, all_threshold=2, maxheight=0.9 * datashare.camera.get_maxcount(spectrum.header))

    # default m0 as a starting point and range we search about that starting point
    m0_default     = datashare.instrument.m0_default
    m0_searchrange = datashare.instrument.m0_searchrange

    pixfraction_blue = datashare.instrument.pixfraction_blue
    pixfraction_red  = datashare.instrument.pixfraction_red

    # get overlap peaks between orders
    # TODO: this takes a lot of time, also use multiple processes?
    Overlaps = _getOverlaps(spectrum, overlap_peaks, npools=npools, nord=datashare.instrument.npix_init_overlaps, pixfraction_blue=pixfraction_blue, pixfraction_red=pixfraction_red, pixel_tol=1, rel_height_diff=0.2)


    #now start to search over different m0
    if type(m0_default) is list:
        m0_range = []

        for m0_d in m0_default:
            min_m0 = np.max((1, m0_d - m0_searchrange))
            max_m0 = m0_d + m0_searchrange

            for m in range(min_m0, max_m0 + 1):
                m0_range.append(m)

        m0_range = np.unique(m0_range)

    else:
        min_m0 = np.max((1, m0_default - m0_searchrange))
        max_m0 = m0_default + m0_searchrange

        m0_range = range(min_m0, max_m0 +1)


    results = []

    args = [[m0, Overlaps, spectrum, reference_filename, all_peaks, npools] for m0 in m0_range]

    for arg in args:
        results.append(_fitSolution(arg))

    #create lists of solutions, m0s and RMS of solutions
    SolutionList = [r[1] for r in results]
    m0List       = [r[0] for r in results]
    rms_list     = [Solution.rms for Solution in SolutionList]

    #best solution hast lowest RMS
    best_ind = np.argmin(rms_list)

    #return m0 and best solution
    return m0List[best_ind], SolutionList[best_ind]

# this method can be cumputed simultaneously in multiple threads
def _fitSolution(args):
    m0, Overlaps, spectrum, reference_filename, all_peaks, npools = args


    #create wavelength solution from overlaps
    logging.info('Start m0 {}'.format(m0))
    Overlap_WaveSolution = Spectra.OverlapWavelengthSolution(m0, Overlaps, spectrum.getPixBounds())
    logging.info('Finished Overlaps for m0 {}'.format(m0))

    """
    for o in range(spectrum.nr_of_orders()):
        plt.clf()
        x_range = np.arange(3300)
        y_range = Overlap_WaveSolution.eval_wavelengths(o, x_range)
        plt.plot(x_range, y_range)


        x_range = np.arange(3300)
        y_range = Overlap_WaveSolution.eval_wavelengths(o+1, x_range)
        plt.plot(x_range, y_range)

        peaks = Overlap_WaveSolution.used_Overlaps.fromSameOverlap(o, o+1)

        red_peaks  = np.array([ov.red_pixel  for ov in peaks])
        blue_peaks = np.array([ov.blue_pixel for ov in peaks])

        plt.scatter(red_peaks, Overlap_WaveSolution.eval_wavelengths(o, red_peaks))
        plt.scatter(blue_peaks, Overlap_WaveSolution.eval_wavelengths(o+1, blue_peaks))

        plt.xlabel('pix')
        plt.ylabel(r'$\lambda$')

        plt.show()
    """

    # get global scale for that m0
    GlobalScale = getGlobalScale(reference_filename, all_peaks, Overlap_WaveSolution, npools=npools)
    logging.info('Finished global scale for m0 {} with global scale {}'.format(m0, GlobalScale))

    #apply global scale
    Overlap_WaveSolution.apply_scale(GlobalScale)

    #create finale wavelength solution with this m0. The RMS of this wavelength solution will be minimal for the correct m0
    Final_WaveSolution = final_wavelength_fit(reference_filename, all_peaks, Overlap_WaveSolution, 0 , fromOverlaps=True)
    logging.info('Finished m0 {}'.format(m0))

    """
    for o in range(spectrum.nr_of_orders()):
        plt.clf()
        x_range = np.arange(3300)
        y_range = Final_WaveSolution.eval_wavelengths(o, x_range)
        plt.plot(x_range, y_range)


        x_range = np.arange(3300)
        y_range = Final_WaveSolution.eval_wavelengths(o+1, x_range)
        plt.plot(x_range, y_range)

        plt.xlabel('pix')
        plt.ylabel(r'$\lambda$')

        plt.show()
    """

    return m0, Final_WaveSolution



def CalibrateFromInitialSolution(spectra, Solution, reference_filename, npools=None, fixm0=True):
    """
    # Use a previously created wavelength solution of another file as a starting point to create a wavelength solution for new ThAr spectra, as the wavelength solutions should only differ slightly
    # Can use multiple threads to process spectra simultaneously to increase speed
    #
    # :param spectra: list of Spectrum.RawSpectrum objects, each RawSpectrum contains a raw ThAr spectrum to calculate a wavelength solution from
    # :param Solution: Spectra.FinalWavelengthSolution object, wavelength solution to start from
    # :param npools: int, number of parallel threads. If None, will use the one defined in config (default None)
    # :param fixm0: boolean.If true, do not calculate order shift, but assume same m0 (useful when comparing files from same night) (default True)
    #
    # :return Solutions: list of Spectra.FinalWavelengthSolution, contains the wavelength solutions for each input spectrum
    """

    if npools is None:
        npools = datashare.reduction_parameters.npools


    #calculate wavelength solutions in independent threads to increase speed
    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
        try:
            args = [[spectra[i], Solution, reference_filename, fixm0] for i in range(len(spectra))]
            Solutions = pool.map(_CalibrateFromInitial, args)
        finally:
            pool.close()
            pool.join()

    return Solutions


#this method can run simultaneously in multiple threads
def _CalibrateFromInitial(args):
    spectrum, Solution, reference_filename, fixm0 = args

    #get MJD
    mjd = datashare.instrument.getMJD(spectrum.header)

    if np.isnan(mjd):
        mjd = 0

    #get parameters from template
    m0                       = Solution.m0
    all_widths               = Solution.allPeaks.getWidths()

    #get overlaps
    overlap_peaks, all_peaks = _getPeaks(spectrum, all_widths, npools=1, overlap_threshold=-1, all_threshold=2, maxheight=0.9 * datashare.camera.get_maxcount(spectrum.header))  #no need for overlap peaks, just use one thread

    #get new m0. It might happen that there was one order mor or less detected, which changes m0
    if fixm0:
        max_order_shift = 0
    else:
        max_order_shift = 3

    m0_shifted, pixel_shift = getm0FromWaveSolution(spectrum, Solution, max_order_shift=max_order_shift)

    #create wavelength solution
    ThAr_Solution = final_wavelength_fit(reference_filename, \
                                    all_peaks, Solution, mjd,  \
                                    pixshift=pixel_shift, ordershift = m0_shifted - m0)

    #transfer header from template to new solution
    ThAr_Solution.ThAr_header = spectrum.header

    return ThAr_Solution



def CalibrateFromReferenceList(spectra, referencelist, complete_reference, npools=None):
    """
    # Use a previously created wavelength reference list to create a wavelength solution for hAr spectra
    # Can use multiple threads to process spectra simultaneously to increase speed
    #
    # :param spectra: list of Spectrum.RawSpectrum objects, each RawSpectrum contains a raw ThAr spectrum to calculate a wavelength solution from
    # :param referencelist: list or str, instrument specific reference list with peak positions and corresponding wavelengths for each order or filename of that file
    # :param complete_reference: list or str, list of all reference wavelengths (without pixel or order information) or filename of reference file
    # :param npools: int, number of parallel threads. If None, will use the one defined in config (default None)
    #
    # :return Solutions: list of Spectra.FinalWavelengthSolution, contains the wavelength solutions for each input spectrum
    """

    if npools is None:
        npools = datashare.reduction_parameters.npools

    #calculate wavelength solutions in independent threads to increase speed
    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
        try:
            args = [[spectra[i], referencelist, complete_reference] for i in range(len(spectra))]
            Solutions = pool.map(_CalibrateFromReferenceList, args)
        finally:
            pool.close()
            pool.join()

    return Solutions


#this method can run simultaneously in multiple threads
def _CalibrateFromReferenceList(args):
    spectrum, referencelist, complete_reference = args

    #get MJD
    mjd = datashare.instrument.getMJD(spectrum.header)

    if np.isnan(mjd):
        mjd = 0

    #create wavelength solution
    ThAr_Solution = wavelength_fit_referencelist(referencelist, complete_reference, spectrum, mjd)

    #transfer header from template to new solution
    ThAr_Solution.ThAr_header = spectrum.header

    return ThAr_Solution

def getm0FromWaveSolution(spectrum, Solution, max_order_shift=3, max_pix_shift=100):
    """
    # Get m0 (physical diffration order of the first software order) of a new spectrum in comparison to a existing wavelenght solution.
    # It might happen that there was one order mor or less detected, which changes m0.
    #
    # :param spectrum: Spectra.RawSpectrum object,  contains the raw (not calibrated) ThAr spectrum
    # :param Solution: Spectra.FinalWavelengthSolution object, existing wavelength solution
    # :param max_order_shift: int, maximal shift in orders (default 3)
    # :param max_pix_shift: int, maximal pixel shift (default 100)
    #
    # :return m0: int, m0 of the new ThAr spectrum
    # :return median_pixel_shift: float, median pixel shift between new spectrum and existing wavelength solution
    """

    assert isinstance(spectrum, Spectra.RawSpectrum)
    assert isinstance(Solution, Spectra.FinalWavelengthSolution)

    order_shifts = []
    pixel_shifts = []

    #go through orders
    for ordernr in range(spectrum.nr_of_orders()):
        order = spectrum[ordernr]

        #minimal and maximal orders to compare
        min_compare_order = np.max((ordernr - max_order_shift,0))
        max_compare_order = np.min((ordernr + max_order_shift, spectrum.nr_of_orders()-1))

        maximums, pix_shifts = [], []

        #go through comparison orders
        for compare_ordernr in range(min_compare_order, max_compare_order+1):
            #found lines of comparison order in existing solution
            found_lines = [line for line in Solution.lines_matched if line.order == compare_ordernr]

            # no lines known, skip this order
            if len(found_lines) < 1:
                maximums.append(-np.inf)
                pix_shifts.append(np.inf)
                continue
            else:
                #correlate order with known lines. Get CCF (maximum) and pixel shift
                maximum, pix_shift = OrderSolutionCCF(order, found_lines, max_pix_shift = max_pix_shift)

                maximums.append(maximum)
                pix_shifts.append(pix_shift)

        maximums   = np.array(maximums)
        pix_shifts = np.array(pix_shifts)

        #all CCFs are negative, this is a bad result, skip
        if not np.any(maximums > 0):
            continue
        else:
            #best fit is comparison order with highest CCF
            order_shifts.append(np.argmax(maximums) - (ordernr - min_compare_order))
            pixel_shifts.append(pix_shifts[np.argmax(maximums)])


    order_shifts = np.array(order_shifts)
    pixel_shifts = np.array(pixel_shifts)

    #order shift is the median one of all found order_shifts (order_shift should ideally be constant for all orders, practically there are some outliners)
    median_order_shift = np.median(order_shifts)
    median_pixel_shift = np.median(pixel_shifts[np.where(order_shifts == median_order_shift)])

    #new m0
    m0 = Solution.m0 + median_order_shift

    return m0, median_pixel_shift



def OrderSolutionCCF(order, solutionlines, max_pix_shift=100, neclect_ends=500):
    """
    # Correlate spectral ThAr order with known line positions from existing wavelength solution
    #
    # :param order: Spectra.RawSpectralOrder object, contains the ThAr order
    # :param solutionlines: list of MatchedLine objects, contains the previously known lines
    # :param max_pix_shift: int, maximal shift in pixels (default 100)
    # :param neglect_ends: int, number of pixels to neglect on both ends of the spectrum to avoid errors from noise near the edges (default 500)
    #
    # :return maximum: float, maximum of CCF (cross correlation function) between order and known lines
    # :return pix_shift: float, shift in pixel between order and known lines
    """

    # no lines, break
    if len(solutionlines) < 1:
        return -np.inf, 0


    # make copy to avoid changes at original
    order = order.copy()

    # set nan flux to zero
    order.flux[np.isnan(order.flux)] = 0

    #neglect edges
    order.flux[:neclect_ends]        = 0
    order.flux[-neclect_ends:]       = 0

    max_pix_shift = np.around(max_pix_shift)

    #this will store the known lines
    lines_order = np.zeros(len(order.flux) + 2 * max_pix_shift)

    #pixels where known lines are
    lines_pixels = np.array([np.around(line.pixel) + max_pix_shift for line in solutionlines]).astype(int)
    #set template to one at these positions
    lines_order[lines_pixels] = 1

    #correlate
    correlation = np.correlate(lines_order, order.flux)
    correlation[np.isnan(correlation)] = -np.inf

    """
    print('')

    fig, axs = plt.subplots(3)
    axs[0].plot(np.arange(len(lines_order)), lines_order)
    axs[1].plot(order.pixels, order.flux)
    axs[2].plot(np.arange(len(correlation)), correlation / len(solutionlines))
    plt.show()
    """

    #maximum of CCF
    maximum   = np.max(correlation) / len(solutionlines)   #take more/less lines into account

    #maximum position of CCF is pixel shift
    pix_shift = np.argmax(correlation) - (max_pix_shift + 0.5)

    """
    fig, axs = plt.subplots(2)

    max_flux = np.nanmax(order.flux)

    axs[0].plot(order.pixels, order.flux, linewidth=0.2)
    axs[0].vlines(lines_pixels, ymin=-0.1*max_flux, ymax=1.1*max_flux)

    axs[0].set_ylim((-0.1*max_flux, 1.1*max_flux))

    axs[1].plot(np.arange(len(correlation)) - (max_pix_shift + 0.5), correlation)

    plt.show()
    """

    return maximum, pix_shift


def PlotThArDrift(WaveSolutionList):
    """
    # Plot the shift of the wavelength solutions over the night. Compare shift in km/s and temperature drift. Return nothing, as this routine just creates a plot
    #
    # :param WaveSolutionList: list of Spectra.FinalWavelengthSolution object, all wavelength solutions of that night
    #
    # :return None
    """
    assert type(WaveSolutionList) is list

    for Solution in WaveSolutionList:
        assert isinstance(Solution, Spectra.FinalWavelengthSolution)

    #get master solution with lowest RMS
    MasterSolution = None
    BestRMS = np.inf

    #go though wavelength solutions
    for Solution in WaveSolutionList:
        #get the wavelength solution with the best (lowest) RMS
        if Solution.rms is not None and not np.isnan(Solution.rms) and Solution.rms < BestRMS:
            BestRMS        = Solution.rms
            MasterSolution = Solution

    #get minimal and maximal useful order by checking lowest and highest matched line
    master_minorder = np.min([Line.order for Line in MasterSolution.lines_matched])
    master_maxorder = np.max([Line.order for Line in MasterSolution.lines_matched])

    #get minimal and maximal pixels for mastersolution. This should be the same for all solutions, so we keep this fixed.
    minpix = np.min([Line.pixel for Line in MasterSolution.lines_matched])
    maxpix = np.max([Line.pixel for Line in MasterSolution.lines_matched])

    pixel_array = np.arange(minpix, maxpix+1)

    shifts_to_master = []
    rms_to_master    = []

    #go through solutions and calculate RMS and median shift (each in km/s) to master solution
    for Solution in WaveSolutionList:
        #again find minorder and maxorder
        minorder = np.min([Line.order for Line in Solution.lines_matched])
        maxorder = np.max([Line.order for Line in Solution.lines_matched])

        #only use orders where both solutions are well defined:
        minorder = np.max([minorder, master_minorder])
        maxorder = np.min([maxorder, master_maxorder])

        #compare calculated wavelengths of master solution and current solution. First index is orders, second index is pixels
        master_wavelengths = np.zeros(shape=(maxorder - minorder + 1, pixel_array.size))
        wavelengths        = np.zeros(shape=(maxorder - minorder + 1, pixel_array.size))

        #evaluate wavelengths for each order
        for i in range(wavelengths.shape[0]):
            master_wavelengths[i, :] = MasterSolution.eval_wavelengths(i, pixel_array)
            wavelengths[i, :]        = Solution.eval_wavelengths(i, pixel_array)

        #residuals
        residuals = wavelengths - master_wavelengths

        #calculate RMS and shift between master and current wavelength solution
        rms = np.sqrt(np.sum(np.square(residuals)) / residuals.size)
        shift = Spectra.Constants.c * np.median(residuals / master_wavelengths)

        shifts_to_master.append(shift)
        rms_to_master.append(rms)

    #get MJD list
    mjds = [Solution.mjd for Solution in WaveSolutionList]

    #get temps
    temps = [datashare.instrument.getTemperature(Solution.ThAr_header) for Solution in WaveSolutionList]

    #plot
    fix, [ax1, ax3] = plt.subplots(1,2,figsize=(20,5))
    ax1.set_xlabel('MJD')
    ax1.set_ylabel('Shift in km/s')
    ax1.scatter(mjds, shifts_to_master)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Temperature in °C')
    ax2.scatter(mjds, temps, color='red')

    ax3.plot(temps, shifts_to_master)
    ax3.set_xlabel('Temperature in °C')
    ax3.set_ylabel('Shift in km/s')

    plt.tight_layout()

    plt.show()

    plt.close()


def ThArGroupMedian(WaveSolutionList, deltaT=10, method='weighted', max_coverage=None):
    """
    # Group together wavelength solutions which underlying exposures were created directly after another. This is used to interpolate the wavelength shift over the night, as it is better to interpolate just the smoothed / grouped shift data points instead of each measurement.
    #
    # :param WaveSolutionList: list of Spectra.FinalWavelengthSolution objects, list with all wavelength solutions of that night
    # :param deltaT: float, time in minutes. If the next ThAr exposure was created sooner than deltaT, we group both exposures together. Note, that this is just the difference between two consecutive exposures, the total time for the ThAr exposures can be longer (e.g. 5 * 5min ThAr would be 25min in total, but they will still all be grouped together, as the time difference between two exposures is 5min < 10min) (default 10)
    # :param method: str, must be one of ['weighted', 'mean', 'median']. Defines the method how to combine the shift values (default 'weighted')
    # :param max_coverage: float or None, time in minutes. maximum time to group ThArs together. If None no limit is applied. AS an example: If max_coverage = 30, but we took 10 * 5 min ThArs, it will create a new group of ThArs after 30mins (so the first 6 exposures) (default None)
    #
    # :return WaveSolutionList: list of Spectra.FinalWavelengthSolution objects, WaveSolutionList as input, but sorted by MJD of exposure
    # :return master_ind: int, index of master wavelength solution in WaveSolutionList
    # :return combined_mjds: 1D numpy array of floats, list of all MJDs of the grouped wavelength solutions (the mid of the MJDs of the group)
    # :return combined_shifts: 1D numpy array of floats, list of all velocity shifts of the grouped wavelength solutions
    # :return shift_errors: 1D numpy array of floats, list of all errors of velocity shifts of the grouped wavelength solutions
    # :return combined_temps: 1D numpy array of floats, list of all instrument temperatures of the grouped wavelength solutions
    # :return temp_errors: 1D numpy array of floats, list of all instrument temperature errors of the grouped wavelength solutions
    """
    #deltaT in min
    #if max_coverage if given (in mins), will break the current bin in any case after this time

    assert type(WaveSolutionList) is list

    for Solution in WaveSolutionList:
        assert isinstance(Solution, Spectra.FinalWavelengthSolution)

    assert method in ['weighted', 'mean', 'median']

    #get lists
    mjds     = np.array([Solution.mjd for Solution in WaveSolutionList])
    rms_list = np.array([Solution.rms for Solution in WaveSolutionList])
    temps    = np.array([datashare.instrument.getTemperature(Solution.ThAr_header) for Solution in WaveSolutionList])

    #make sure that MJDs are sorted
    if not np.all(np.diff(mjds) >= 0):   #array is not sorted
        sort_inds        = np.argsort(mjds)

        mjds             = mds[sort_inds]
        rms_list         = rms_list[sort_inds]
        temps            = temps[sort_inds]
        WaveSolutionList = [WaveSolutionList[i] for i in sort_inds]

    #calclate shifts
    #get master solution with lowest RMS
    master_ind = np.argmin(rms_list)
    MasterSolution = WaveSolutionList[master_ind]

    #get minimal and maximal useful order by checking lowest and highest matched line
    master_minorder = np.min([Line.order for Line in MasterSolution.lines_matched])
    master_maxorder = np.max([Line.order for Line in MasterSolution.lines_matched])

    #get minimal and maximal pixels for mastersolution. This should be the same for all solutions, so we keep this fixed.
    minpix = np.min([Line.pixel for Line in MasterSolution.lines_matched])
    maxpix = np.max([Line.pixel for Line in MasterSolution.lines_matched])

    pixel_array = np.arange(minpix, maxpix+1)

    shifts_to_master = []
    rms_to_master    = []
    error_to_shift   = []

    #go through solutions and calculate RMS and median shift (each in km/s) to master solution
    for Solution in WaveSolutionList:
        #again find minorder and maxorder
        minorder = np.min([Line.order for Line in Solution.lines_matched])
        maxorder = np.max([Line.order for Line in Solution.lines_matched])

        #only use orders where both solutions are well defined:
        minorder = np.max([minorder, master_minorder]).astype(int)
        maxorder = np.min([maxorder, master_maxorder]).astype(int)

        #compare calculated wavelengths of master solution and current solution. First index is orders, second index is pixels
        master_wavelengths = np.zeros(shape=(maxorder - minorder + 1, int(pixel_array.size)))
        wavelengths        = np.zeros(shape=(maxorder - minorder + 1, int(pixel_array.size)))

        #evaluate wavelengths for each order
        for i in range(wavelengths.shape[0]):
            master_wavelengths[i, :] = MasterSolution.eval_wavelengths(i, pixel_array)
            wavelengths[i, :]        = Solution.eval_wavelengths(i, pixel_array)

        #residuals
        residuals = wavelengths - master_wavelengths

        #calculate RMS and shift between master and current wavelength solution
        rms = np.sqrt(np.sum(np.square(residuals)) / residuals.size)
        shift = Spectra.Constants.c * np.nanmedian(residuals / master_wavelengths)
        #error = np.sqrt(np.sum(np.square(Spectra.Constants.c *(residuals/master_wavelengths) - shift))/residuals.size + np.square(Solution.rms))
        error = np.sqrt((np.square(MasterSolution.rms) + np.square(Solution.rms)) /2)

        """
        for i in range(wavelengths.shape[0]):
            if i == wavelengths.shape[0] -1:
                plt.plot(wavelengths[i,:], Spectra.Constants.c * residuals[i,:]/master_wavelengths[i,:], color='blue', label='Measurements')
                plt.plot(wavelengths[i,:], shift * np.ones_like(wavelengths[i,:]), color='red', label='Median shift')
            else: #plot without label
                plt.plot(wavelengths[i,:], Spectra.Constants.c * residuals[i,:]/master_wavelengths[i,:], color='blue')
                plt.plot(wavelengths[i,:], shift * np.ones_like(wavelengths[i,:]), color='red')
        plt.xlabel('Wavelength')
        plt.ylabel('Rediduals to master in km/s')
        plt.title('Residuals in dependency of wavelength')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print(error)
        """

        shifts_to_master.append(shift)
        rms_to_master.append(rms)
        error_to_shift.append(error)

    shifts_to_master = np.array(shifts_to_master)
    rms_to_master    = np.array(rms_to_master)
    error_to_shift   = np.array(error_to_shift)

    #plot, if requested
    if datashare.reduction_parameters.plot_ThArGroupMedian:
        plot_mjds = mjds - np.min(mjds)

        fig, axs = plt.subplots(2, figsize=(16,9), sharex=True)
        axs[0].errorbar(plot_mjds, shifts_to_master, yerr=error_to_shift, fmt='o', color='red', label='RMS to master')
        axs[1].errorbar(plot_mjds, shifts_to_master, yerr=rms_list, fmt='o', color='blue', label='Solution internal RMS')
        axs[0].set_xlabel('MJD - {}'.format(np.round(np.min(mjds),2)))
        axs[0].set_ylabel('Shift to master in km/s')
        axs[0].set_title('Shift to master solution, Error = Shift to master')
        axs[1].set_xlabel('MJD - {}'.format(np.round(np.min(mjds),2)))
        axs[1].set_ylabel('Shift to master in km/s')
        axs[1].set_title('Shift to master solution, Error = internal solution RMS')
        #axs[0].legend()

        plt.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, "ThArShift.png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')       #needed to show plot in Jupyter Notebook
            plt.show()

        plt.close()

    #group by MJD
    #MJDs with same index will get grouped together
    indices = np.zeros(len(WaveSolutionList)).astype(int)

    current_index = 0

    diffs = np.diff(mjds) * (24 * 60)       #convert MJD (days) to mins

    #go through all wavelength solutions
    for i in range(len(diffs)):
        #next solution is not close: increase index (new group)
        if diffs[i] > deltaT:
            current_index += 1


        #break in any case if time coverage is larger than max_coverage
        elif max_coverage is not None and \
            (mjds[i+1] - np.min(mjds[np.where(indices == current_index)])) * (24 * 60) > max_coverage:
            current_index += 1

        #add index of current wavelength solution
        indices[i+1] = current_index

    #combine shifts
    combined_mjds   = []
    combined_shifts = []
    shift_errors    = []
    combined_temps  = []
    temp_errors     = []

    #go through all groups
    for i in range(indices[-1] + 1):        #indices are sorted
        #indices of current group
        inds = np.where(indices == i)[0]

        #just one wavelenght solution, nothing to combine
        if len(inds) == 1:
            combined_mjds.append(mjds[inds[0]])
            combined_shifts.append(shifts_to_master[inds[0]])
            shift_errors.append(error_to_shift[inds[0]])
            combined_temps.append(temps[inds[0]])
            temp_errors.append(0)                           #ToDo: Use different sensors to determine temp err

        #combine shifts with weights. Weights are higher for shifts with smaller errors
        elif method == 'weighted':
            #filter measurements with unrealistic high errors
            min_err = np.min(error_to_shift)

            inds = inds[np.where(error_to_shift[inds] < 3 * min_err)]

            if len(inds) == 0:
                continue

            #create weights
            weights = np.power(error_to_shift[inds], -2)     #weights = 1/error^2

            #create combined values
            combined_mjds.append(np.average(mjds[inds], weights=weights))
            mean_shift = np.average(shifts_to_master[inds], weights=weights)
            combined_shifts.append(mean_shift)
            shift_errors.append(np.sqrt(np.sum(weights * np.square(shifts_to_master[inds] - mean_shift)) / np.sum(weights)) + np.sum(np.square(error_to_shift[inds]))/len(inds))
            #shift_errors.append(np.sqrt(np.sum(weights * np.square(error_to_shift[inds])) / np.sum(weights)))

            mean_temp = np.average(temps[inds], weights=weights)
            combined_temps.append(mean_temp)
            temp_errors.append(np.sqrt(np.sum(weights * np.square(temps[inds] - mean_temp)) / np.sum(weights)))

        #combine shifts by just using the mean
        elif method == 'mean':
            combined_mjds.append(np.average(mjds[inds]))

            mean_shift = np.average(shifts_to_master[inds])
            combined_shifts.append(mean_shift)
            shift_errors.append(np.sqrt(np.sum(np.square(shifts_to_master[inds] - mean_shift)) / len(inds) + np.sum(np.square(error_to_shift[inds]))/len(inds)))
            #shift_errors.append(shift_errors.append(np.sqrt(np.sum(np.square(error_to_shift[inds])) / len(inds))))

            mean_temps = np.average(temps[inds])
            combined_temps.append(mean_temps)
            temp_errors.append(np.sqrt(np.sum(np.square(temps[inds] - mean_temp)) / len(inds)))

        #combine shifts by just using the median
        elif method == 'median':
            combined_mjds.append(np.median(mjds[inds]))

            median_shift = np.median(shifts_to_master[inds])
            combined_shifts.append(median_shift)
            shift_errors.append(np.sqrt(np.sum(np.square(shifts_to_master[inds] - median_shift)) / len(inds)) + np.sum(np.square(error_to_shift[inds]))/len(inds))
            #shift_errors.append(shift_errors.append(np.sqrt(np.sum(np.square(error_to_shift[inds])) / len(inds))))

            median_temp = np.median(temp[inds])
            combined_temps.append(median_temp)
            temp_errors.append(np.sqrt(np.sum(np.square(temps[inds] - median_temp)) / len(inds)))

    #comvert to numpy arrays
    combined_mjds   = np.array(combined_mjds)
    combined_shifts = np.array(combined_shifts)
    shifts_errors   = np.array(shift_errors)
    combined_temps  = np.array(combined_temps)
    temp_errors     = np.array(temp_errors)


    return  WaveSolutionList, master_ind, combined_mjds, combined_shifts, shift_errors, combined_temps, temp_errors


def _order_CCF(order_peaks, reference_peaks, npix, maxshift=100):
    if maxshift > npix //2:
        maxshift = npix //2 - 1

    #create order mask
    order_mask = np.zeros(npix)
    order_mask[np.round(order_peaks).astype(int)] = 1. #/ len(order_peaks)

    #smooth order mask
    order_mask = ndimage.gaussian_filter(order_mask, sigma=2)

    #create reference mask
    reference_mask = np.zeros(npix)
    reference_mask[np.round(reference_peaks).astype(int)] = 1. / len(reference_peaks)
    reference_mask = reference_mask[maxshift:-maxshift]

    CCF = np.correlate(order_mask, reference_mask, mode='valid')

    mid_pix = len(CCF) // 2.

    #plt.plot(order_mask / np.max(order_mask), linewidth=0.1)
    #plt.vlines(reference_peaks, ymin=0, ymax=np.max(order_mask), linestyle='dashed', color='red', linewidth=0.1)
    #plt.plot(np.arange(npix)[maxshift:-maxshift], reference_mask / np.max(reference_mask), linewidth=0.1, color='red')
    #print('')
    #plt.show()

    #plt.plot(np.arange(len(CCF)) - mid_pix, CCF)
    #print('')
    #plt.show()

    return np.max(CCF), np.argmax(CCF) - mid_pix


def m0FromReference(thar_spectrum, reference):
    """
    # Use an instrument specific reference list to calculate m0 and get a pixel shift between spectrum and reference
    #
    #
    # :param thar_spectrum: Spectra.Spectrum object, contains the extracted ThAr spectrum
    # :param reference: 2D numpy array or str, either a (X,3) numpy array with phys. ordernr, pixel and wavelength in one row or the filename of a csv with that data. Delimiter must be ';'
    #
    # :return m0: int, m0 of the spectrum
    # :return pix_shift: float, median pixel shift to reference
    """

    if isinstance(reference, str):
        reference = np.genfromtxt(reference, dtype=float, delimiter=';', comments='#')

    testorder = datashare.instrument.reference_list_testorder

    reference_inds  = np.where(np.abs(reference[:, 0] - testorder) < 0.1)     #== is not possible as reference is a float array
    reference_peaks = reference[reference_inds, 1]

    min_ordnr = np.max((0, testorder - datashare.instrument.m0_default - datashare.instrument.m0_searchrange))
    max_ordnr = np.min((thar_spectrum.nr_of_orders(), testorder - datashare.instrument.m0_default + datashare.instrument.m0_searchrange + 1))

    CCF_list   = []
    m_list    = []


    all_widths     = _getwidths(thar_spectrum, maxheight=0.9 * datashare.camera.get_maxcount(thar_spectrum.header))
    _,  ThAr_Peaks = _getPeaks(thar_spectrum, all_widths, npools = 1, all_threshold=2, maxheight=0.9 * datashare.camera.get_maxcount(thar_spectrum.header))

    all_peaks, all_orders = ThAr_Peaks.allPeaks()

    for ordnr in range(min_ordnr, max_ordnr):
        order = thar_spectrum[ordnr]

        npix = len(order.pixels)

        order_peaks = all_peaks[all_orders == ordnr]

        if len(order_peaks) < 1:
            CCF_list.append(-np.inf)
            continue

        CCF, _ = _order_CCF(order_peaks, reference_peaks, npix, maxshift=datashare.instrument.max_pixshift)

        CCF_list.append(CCF)

    m0 = int(testorder -  (min_ordnr + np.argmax(CCF_list)))

    #again go through all orders to get median shift
    shifts_list = []
    for ordernr in range(0, thar_spectrum.nr_of_orders()):
        reference_inds  = np.where(np.abs(reference[:, 0] - (ordernr + m0)) < 0.1)     #== is not possible as reference is a float array
        reference_peaks = reference[reference_inds, 1]

        order = thar_spectrum[ordernr]

        npix = len(order.pixels)

        order_peaks = all_peaks[all_orders == ordernr]

        if len(order_peaks) < 1 or len(reference_peaks) < 1:
            continue

        _, shift = _order_CCF(order_peaks, reference_peaks, npix, maxshift=datashare.instrument.max_pixshift)

        shifts_list.append(shift)

    return m0, np.median(shifts_list)
