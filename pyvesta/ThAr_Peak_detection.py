######################################
# Created  2026/01/05
#
# Author: Lukas Stock
#
#
#####################################

import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt
import numbers
from multiprocessing import Pool

from pyvesta import FitFunctions
from pyvesta import datashare
from pyvesta import plot_utilities

def init_pools(reduction_parameters, instrument, camera):
    datashare.reduction_parameters = reduction_parameters
    datashare.instrument           = instrument
    datashare.camera               = camera

def find_ThAr_peaks(order, width=5, threshold=20, maxheight=np.inf):
    """
    # Find ThAr peaks in a spectral order usind scipy.signal.find_peaks_cwt
    #
    # :param order: SpectralOrder object, contains the ThAr spectrum
    # :param width: float or list of floats, approximate width (FWHM) of ThAr peaks. In case of a list will find peaks for all widths and merhe them in one list (default 5)
    # :param threshold: float, minimal SNR of peaks (default 20)
    # :param maxheight: float, maximal peak height to filter out saturated peaks (default np.inf, no filtering)
    #
    # :return centers: numpy.ndarray, array with all found peak centers
    # :return errs:  numpy.ndarray, array with errors of found peak centers
    """

    # search peaks
    peaks = signal.find_peaks_cwt(order.flux, width)


    # apply minimal SNR
    good_peaks = peaks[np.where((order.flux/order.errors)[peaks] > threshold)]

    # apply maximal height
    max_peak_heights = []

    for peak in good_peaks:
        min_ind = np.round(np.max((0, peak - np.max(width)))).astype(int)
        max_ind = np.round(np.min((len(order.flux) - 1, peak + np.max(width) + 1))).astype(int)
        max_peak_heights.append(np.max(order.flux[min_ind:max_ind]))

    max_peak_heights = np.array(max_peak_heights)

    good_peaks = good_peaks[np.where(max_peak_heights < maxheight)]

    centers     = []
    sigma_list  = []
    errs        = []


    if type(width) is list or type(width) is np.ndarray:
        median_width = np.median(width)
    else:
        median_width = width

    #fit center of peak
    for peak in good_peaks:
        min_ind = np.around(np.max((0, peak-2*median_width))).astype(int)
        max_ind = np.around(np.min((len(order.flux)-1, peak+2*median_width+1))).astype(int)
        fit_spec = order.flux[min_ind:max_ind]
        fit_errs = order.errors[min_ind:max_ind]

        x = np.arange(len(fit_spec)) + min_ind

        #fit gaussian plus constant background
        #make guesses. A, x0, sigma, b

        #standard deviation is correlated to width / FWHM
        sigma = np.abs(median_width) / 2.355
        p0 = [order.flux[peak], peak, sigma, np.mean((fit_spec[0], fit_spec[-1]))]

        #make bounds. (lower_bounds), (upper_bounds).
        bounds = ((0, peak-median_width, 0.1*sigma, -np.inf), (np.inf, peak + median_width, 10*sigma, np.inf))

        #fit
        try:
            popt, pconv = optimize.curve_fit(FitFunctions.gaussianAndConstant, x, fit_spec, p0=p0, bounds=bounds)
        except:
            centers.append(np.nan)
            sigma_list.append(np.nan)
            errs.append(np.nan)
            continue

        perr = np.sqrt(np.diag(pconv))

        residuals = fit_spec - FitFunctions.gaussianAndConstant(x, *popt)

        #plot, if requested
        if datashare.reduction_parameters.plot_ThArPeaks:
            fig, axs = plt.subplots(1,2,figsize=(16,9))
            axs[0].plot(x, fit_spec, color='black', label='peak')
            axs[0].plot(x, FitFunctions.gaussianAndConstant(x, *popt), color='red', label='fit')
            axs[1].plot(x, (fit_spec - FitFunctions.gaussianAndConstant(x, *popt))/np.sqrt(np.sum(np.square(fit_errs))))
            #axs[1].plot(x, residuals/np.sqrt(np.sum(np.power(residuals, 2.))))
            axs[0].legend()

            if datashare.reduction_parameters.save_plots:
                filename = plotutilities.getnextfilename(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'')+ "_ThArPeaks", ".png")
                plt.savefig(filename, dpi=300)

            if datashare.reduction_parameters.show_plots:
                plt.show()

            plt.close()

        #check if fit went well. If yes append results to result lists
        amplitude = popt[0]
        center    = popt[1]
        sigma     = popt[2]
        c         = popt[3]

        center_err = perr[1]

        if np.abs(center - peak) < 3 and c < amplitude:
            centers.append(center)
            sigma_list.append(sigma)
            errs.append(center_err)

        # bad fit, add nans to result list
        else:
            centers.append(np.nan)
            sigma_list.append(np.nan)
            errs.append(np.nan)


    centers    = np.array(centers)
    sigma_list = np.array(sigma_list)
    errs       = np.array(errs)

    #filter out all nans
    good_inds  = np.where([not np.isnan(value) for value in centers])[0]
    centers    = centers[good_inds]
    sigma_list = sigma_list[good_inds]
    errs       = errs[good_inds]

    #filter detected lines by sigma clipping
    median_sigma = np.nanmedian(sigma_list)
    rms = np.sqrt(np.sum(np.square(sigma_list - median_sigma))/len(sigma_list))

    good_inds = np.where(np.abs(sigma_list - median_sigma) < 3 * rms)

    return centers[good_inds], errs[good_inds]


def get_equipotential_line(peak_array, rel_height=0.25, method='plt'):
    """
    # calculate indicies of pixels corresponding to an equipotential line of the peak
    # This function will not be used, but is still included for completion reasons
    #
    # :param peak_array: 2D numpy array, a peak with at least 3x3 entries (boundaries are ignored)
    # :param rel_height: float, height of the equipotential line relative to the maximum of the peak, must be between 0 and 1
    #                    if rel_height is a list, return the pixels for each entry
    # :param method: str, can be 'custom' or 'plt'. If 'custom' a self written routine will be used, 'plt' will use a routine by matplotlib
    #
    # :return
    """
    if np.any(np.array(peak_array.shape) < 3):
        raise ValueError('peak_array must be at least an 3x3 array!')

    def _get_pixels(peak_array, rel_height):
        if rel_height < 0 or rel_height > 1:
            rel_height = 0.5

        #threshold
        threshold = rel_height * np.max(peak_array)

        #use custom method
        if method == 'custom':
            line_indicies = ([], [])

            #remove boundaries from indexing
            index_shape = tuple(np.array(peak_array.shape) - 2)


            for row, col in np.ndindex(index_shape):
                #think about adding 1 to row and col to account for removed boundaries
                neighbors = peak_array[row:row+2,col:col+2]


                if np.any(neighbors > threshold) and np.any(neighbors < threshold):
                    # add only 0.5 to set contour to middle of pixels
                    line_indicies[0].append(row+0.5)
                    line_indicies[1].append(col+0.5)


            # convert to numpy array
            line_indicies = (np.array(line_indicies[0]), np.array(line_indicies[1]))

        #use matplotlib
        elif method == 'plt':
            fig, axs = plt.subplots(figsize=(1,1))

            x,y = np.arange(peak_array.shape[1]), np.arange(peak_array.shape[0])

            X,Y = np.meshgrid(x,y)

            cs = axs.contour(X,Y,peak_array, [threshold])

            path = cs.collections[0].get_paths()[0] #use only first path
            line_indicies = path.vertices.T
            line_indicies = line_indicies[::-1, ::-1]

            plt.close(fig)

        else:
            raise ValueError('{} is no valid method'.format(method))

        return line_indicies

    if type(rel_height) is list:
        line_indicies = [_get_pixels(peak_array, height) for height in rel_height]
    else:
        line_indicies = _get_pixels(peak_array, rel_height)

    if False:
        # plot
        fig, axs = plt.subplots(figsize=(16,9))

        x,y = np.arange(peak_array.shape[1]), np.arange(peak_array.shape[0])

        X,Y = np.meshgrid(x,y)

        axs.imshow(peak_array)


        params = np.stack([fit_ellipses(x,y) for (x,y) in line_indicies])

        y = lambda x, centerx, centery, angle: centery + (x - centerx) * np.sin(angle)
        x = np.linspace(0,peak_array.shape[1], 100)

        for i in range(params.shape[0]):
            axs.plot(x, y(x, params[i][0], params[i][1], -params[i][-1]))

        if type(line_indicies) is list:
            colors = ['red', 'yellow', 'white', 'blue']

            axs.contour(X,Y,peak_array, np.array(rel_height) * np.max(peak_array))
            for j in range(len(line_indicies)):
                for i in range(len(line_indicies[j][0])):
                    axs.scatter(line_indicies[j][1][i], line_indicies[j][0][i], c=colors[j%len(colors)])
        else:
            axs.contour(X,Y,peak_array, [rel_height * np.max(peak_array)])
            for i in range(len(line_indicies[0])):
                axs.scatter(line_indicies[1][i], line_indicies[0][i], c='red')

        plt.show()

        plt.close()

    return line_indicies

def fit_ellipses(x,y):
    """
    #return the parameters of the ellipse defined by the x and y coordinates
    #algorithm from https://stackoverflow.com/questions/52818206/fitting-an-ellipse-to-a-set-of-2-d-points
    #ellipse equation: 1 = Ax^2 + Bx*y + Cy^2 + Dx + Fy + G
    #
    # This function will not be used, but is still included for completion reasons
    #
    # :param x: numpy.ndarray, onedimensional array with all x coordinates of the ellipse
    # :param y: numpy.ndarray, onedimensional array with all y coordinates of the ellipse
    #
    # :return np.array([x_center, y_center, ax1, ax2, eccentricity, angle, rms]): np.ndarray of floats. x_center is the x-position of the center, y_center the y-position of the center, ax1 and ax2 the semimajor axes, eccentricity the eccentricity, angle the angle (in rad) of the ellipse against the x-axis and rms the RMS of the fit
    """

    #convert to arrays
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    try:
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])               #6 different parameters of the ellipse
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))    #solve by getting eigenvalues
        n = np.argmax(np.abs(E))
        a = V[:,n]
        #a[0] = A, a[1] = 2B, a[2] = C, a[3] = 2D, a[4] = 2F, a[5] = G
    except:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

    #convert parameters to better understandable parameters
    #get equation parameters
    a, b,c,d,f,g = a[0], a[1]/2, a[2], a[3]/2, a[4]/2, a[5]

    residuals = a * np.power(x, 2.) + b * x * y + c * np.power(y, 2.) + d * x + f * y + g -1
    rms       = np.sqrt(np.sum(np.square(residuals))) / len(residuals)

    #center
    y_center = (c*d-b*f)/(b*b-a*c)
    x_center = (a*f-b*d)/(b*b-a*c)

    #major axes
    top     = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    bottom1 =(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    bottom2 =(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))

    ax1 = np.sqrt(top/bottom1)
    ax2 = np.sqrt(top/bottom2)

    if ax2 > ax1:
        ax1, ax2 = ax2, ax1

    #eccentricity
    eccentricity = np.sqrt(1 - ax2/ax1)

    #angle of rotation
    if b == 0:
        if a > c:
            angle = 0
        else:
            angle = np.pi/2.
    else:
        b = float(b)
        a = float(a)
        c = float(c)

        #angle = np.pi/2. - np.arctan2(2. * b,a-c)/2.
        angle = np.pi/2. - np.arctan2(b,a-c)

        if angle > np.pi/2.:
            angle -= np.pi

        #if a > c:
        #    angle = np.pi/2. + np.arctan(2*b/(a-c))/2.
        #else:
        #    angle = np.arctan(2*b/(a-c))/2.


    if ax1 < ax2:
        ax1, ax2 = ax2, ax1

        angle += np.pi/2.

        if angle > np.pi/2.:
            angle -= np.pi

    if angle < 0:
        angle += np.pi / 2.

    return np.array([x_center, y_center, ax1, ax2, eccentricity, angle, rms])


def get_median_peak_sigma(order, approx_width = 20, maxheight=np.inf):
    """
    #    Calculate median peak sigma of all peaks in this ThAr order
    #    sigma is correlated to FWHM of the peaks, assuming they have a gaussian shape
    #
    #    :param order: SpectralOrder object, which contains the spectral order
    #    :param approx_width: float, approximate width of ThAr peaks in pixels (default 20)
    #    :param maxheight: float, maximal peak height to filter out saturated peaks (default np.inf, no filtering)
    #
    #    :return median_sigma: float, median sigma of all ThAr peaks in this order
    #    :return good_peaks: numpy.ndarray, list of all indices (pixels) with good ThAr peaks
    """


    #min_height = 10*np.median(np.sort(order.errors))

    min_number_of_peaks = 10            #minimal number of peaks
    SNR_steps = [25, 20, 15, 10, 5]     #SNR steps, will decrease if not enough peaks were found
    peaks = []

    j = 0

    #test with multiple iterations with different starting points, wasn't successfull
    #widths = [0.75, 1., 1.33] * approx_width
    widths = approx_width

    #search for peaks in spectrum. Decrease minimal SNR till enough peaks were found
    while len(peaks) < min_number_of_peaks and j < len(SNR_steps):
        #minimal peak height, correlated to errors
        min_height = SNR_steps[j] * order.errors
        #peaks, _ = signal.find_peaks(order.flux, height=min_height, prominence=approx_width//2)
        #peaks = signal.find_peaks_cwt(order.flux, widths=np.ones(order.flux.shape)*approx_width//2, min_snr=SNR_steps[j])

        #find peaks
        peaks = signal.find_peaks_cwt(order.flux, widths)

        #save only peaks with enough SNR
        peaks = peaks[order.flux[peaks] > min_height[peaks]]

        j += 1

    # apply maximal height
    max_peak_heights = []

    for peak in peaks:
        min_ind = np.round(np.max((0, peak - np.max(widths)))).astype(int)
        max_ind = np.round(np.min((len(order.flux) - 1, peak + np.max(widths) + 1))).astype(int)
        max_peak_heights.append(np.max(order.flux[min_ind:max_ind]))

    max_peak_heights = np.array(max_peak_heights)

    peaks = peaks[np.where(max_peak_heights < maxheight)]

    #print('Number of peaks: {}'.format(len(peaks)))

    #filter good peaks

    good_peaks  = []
    good_sigmas = []
    sigmas      = []

    #this will store all fits of the peaks for later plotting
    fits = np.zeros_like(order.flux)


    #go through all peaks
    i = 0
    while i < len(peaks):
        peak = peaks[i]

        #how many gaussians do we fit? Start with one
        npeaks = 1

        #check if next peak belongs to the current one -> double peak
        if i < len(peaks) - 1:
            next_peak = peaks[i+1]

            #double peak
            if next_peak - peak < 2 * approx_width and order.flux[next_peak] > 0.1 * order.flux[peak]:
                #print('Reject peak {} as double peak'.format(peak))
                npeaks = 2

                i += npeaks
                continue


        #fit current peak with gaussian
        if npeaks == 1:
            min_ind = np.max((0, peak-approx_width))
            max_ind = np.min((len(order.flux)-1, peak+approx_width+1))
            fit_spec = order.flux[min_ind:max_ind]
            fit_errs = order.errors[min_ind:max_ind]

            fitfunc = FitFunctions.gaussianAndConstant

            # A, x0, sigma, b
            p0 = [order.flux[peak], peak, approx_width, np.mean((fit_spec[0], fit_spec[-1]))]

            #bounds, (lower_bounds), (upper_bounds)
            bounds = ((0, peak-approx_width, 0, -np.inf), (np.inf, peak+approx_width, 5*approx_width, np.inf))




        #attempt to fit double gaussian for double peaks, did not work well, will not be used
        if False:
            #peak < next_peak
            min_ind = np.max((0, peak-approx_width))
            max_ind = np.min((len(order.flux)-1, next_peak+approx_width+1))
            fit_spec = order.flux[min_ind:max_ind]
            fit_errs = order.errors[min_ind:max_ind]

            fitfunc = FitFunctions.doublegaussianAndConstant

            # A1, x1, sigma1, A2, x2, sigma2, b
            p0 = [order.flux[peak], peak, approx_width, order.flux[next_peak], next_peak, approx_width, np.mean((fit_spec[0], fit_spec[-1]))]

            #bounds, (lower_bounds), (upper_bounds)
            bounds = ((0, peak-approx_width, 0, 0, next_peak-approx_width, 0, -np.inf), (np.inf, peak+approx_width, 5*approx_width, np.inf, next_peak+approx_width, 5*approx_width, np.inf))

        x = np.arange(len(fit_spec)) + min_ind

        #fit peak
        try:
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_spec, p0=p0, bounds=bounds)
        except:
            #bad fit, continue
            i += 1
            continue

        #get sigma. In case of double gaussian use mean
        if True:
            sigma = popt[2]
        elif False:
            sigma = 0.5 * (popt[2] + popt[5])

        #check if fit went well
        fit_rms = np.sqrt(np.mean(np.square(fitfunc(x, *popt) - fit_spec)))
        spec_rms = np.sqrt(np.mean(np.square(fit_spec)))

        #TODO
        #Criteria not very useful, might need an update in the future
        #fit_success = fit_rms <  0.5 * spec_rms and (not np.any(np.abs(fitfunc(x, *popt) - fit_spec) > 3 * fit_errs))
        fit_success = np.all(np.abs(fitfunc(x, *popt) - fit_spec) < 0.5 * np.max(fit_spec))

        #do not use bad fits for sigma median
        if not fit_success:
            #print('Reject peak {} for bad fitting'.format(peak))
            i += 1
            continue

        good_peaks.append(peak)
        good_sigmas.append(sigma)

        #add fit to plot
        if datashare.reduction_parameters.plot_ThArSigma:
            fits[x[0]:x[-1] +1] += fitfunc(x, *popt)

        i += 1

    good_peaks  = np.array(good_peaks)
    good_sigmas = np.array(good_sigmas)

    median_sigma = np.median(good_sigmas)

    #no peaks found
    if good_peaks.size < 1:
        return -1, np.array([])

    #TODO: check sigmas with median sigma for good_peaks
    #maybe better

    #check that difference from sigmas to median_sigma is less than 50%
    good_peaks = good_peaks[(np.abs(good_sigmas - median_sigma)/median_sigma < 0.5)]

    #plot, if requested
    if datashare.reduction_parameters.plot_ThArSigma:
        fig, axs = plt.subplots(figsize=(16,9))

        axs.plot(order.pixels, order.flux, color='red', label='spectrum')
        axs.plot(order.pixels, fits, linestyle='dashed', color='black', label='fit')

        axs.legend()

        axs.set_xlabel('Pixels')
        axs.set_ylabel('Spectrum in a.u.')

        axs.set_title('Peak Fit')

        plt.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = plotutilities.getnextfilename(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_ThArSigmafit", ".png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            plt.show()

        plt.close()


    return median_sigma, good_peaks

def get_ThAr_peak_width(order, approx_width=5, threshold=50, maxheight=np.inf):
    """
    # Get median FWHMs of ThAr peaks in order by fitting gaussians at each peak
    # If more than 10 peaks are found just use the 10 heighest peaks
    #
    # :param order: SpectralOrder object, contains the flux
    # :param approx_width: float, approximate width of peaks iin pixels as a starting point of fit (default 5)
    # :param threshold: float, minimal SNR ratio of peak to be used (default 50)
    #
    # :return median_width: float, median FWHM of ThAr peaks in this order
    """
    # create minimal heights
    min_height = threshold * order.errors

    # find peaks
    peaks, _ = signal.find_peaks(order.flux, height=min_height, distance=approx_width//2)

    #filter out saturated peaks
    peaks = peaks[np.where(order.flux[peaks] < maxheight)]

    FWHMs = []

    # if more than 10 peaks just use the 10 heighest ones
    if len(peaks) > 10:
        sort_inds = np.argsort(order.flux[peaks])[::-1]

        peaks = peaks[sort_inds[:10]]

    FWHMs = []

    # fit each peak with a gaussian
    for peak in peaks:
        # cut out peaks to avoid interference with other peaks
        min_ind = np.max((0, peak-2*approx_width))
        max_ind = np.min((len(order.flux)-1, peak+2*approx_width+1))
        fit_spec = order.flux[min_ind:max_ind]
        fit_errs = order.errors[min_ind:max_ind]

        x = np.arange(len(fit_spec)) + min_ind

        #standard deviation is correlated to width / FWHM
        sigma = np.abs(approx_width) / 2.355
        p0 = [order.flux[peak], peak, sigma, np.mean((fit_spec[0], fit_spec[-1]))]

        #make bounds. (lower_bounds), (upper_bounds).
        bounds = ((0, peak-approx_width, 0.1*sigma, -np.inf), (np.inf, peak + approx_width, 5*sigma, np.inf))

        #fit
        try:
            popt, pconv = optimize.curve_fit(FitFunctions.gaussianAndConstant, x, fit_spec, p0=p0, bounds=bounds)
        except:
            continue

        perr = np.sqrt(np.diag(pconv))

        residuals = fit_spec - FitFunctions.gaussianAndConstant(x, *popt)

        sigma = popt[2]

        # convert gaussian sigma to FWHM and add to list
        FWHMs.append(sigma * 2.355)

    # build median. If not possible return nan
    if len(FWHMs) > 0:
        return np.nanmedian(FWHMs)
    else:
        return np.nan


def get_good_peaks(Spec, npools = None, approx_width = 10):
    """
    # find good emission lines in ThAr calibration frames
    # for calculating the line tilt and the PSF
    # Use single peaks and no double peaks
    #
    # :param Spec: RawSpectrum object, extracted (without any calibrations) ThAr spectrum to find peaks in
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param approx_width: float, approximate width of ThAr peaks (default 10)
    #
    # :return peak_list: list of lists of ints, this list contains a list per order with the indicies (pixels) of the peaks
    """


    peak_list = []

    if datashare.camera is None:
        maxheight = np.inf
    else:
        maxheight = 0.9 * datashare.camera.get_maxcount(Spec.header)

    if npools is None:
        npools = datashare.reduction_parameters.npools

    #go through orders
    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera)) as pool:
        try:
            args = [[order, approx_width, maxheight] for order in Spec.allOrders()]

            all_peaks = pool.map(_get_good_peaks_order, args)
        finally:
            pool.close()
            pool.join()


    for good_peaks in all_peaks:
        #any good peak found?
        if len(good_peaks) < 1:
            peak_list.append([])
            continue

        #append peaks to list
        peak_list.append(good_peaks)

    #return peaks
    return peak_list


def _get_good_peaks_order(args):
    order, approx_width, maxheight = args

    median_sigma, good_peaks = get_median_peak_sigma(order, approx_width=approx_width, maxheight=maxheight)

    if len(good_peaks) < 1:
        return []

    #ensure peaks are sorted
    good_peaks = np.sort(good_peaks)

    #peak heights
    peak_heights = order.flux[good_peaks]


    #throw away all double peaks

    #minimum distance between peaks
    min_dist = 10 * median_sigma

    bad_inds = []

    #go through all peaks and look whether there are too close neighbors
    #if there are neighbors higher than 20% of peak remove both
    for i in range(good_peaks.size):
        #check neighbors
        if i < good_peaks.size - 1:
            #we do not need to check prevoius peak as this distance is checked by the peak before

            #next peak
            if good_peaks[i+1] - good_peaks[i] < min_dist:
                if np.abs(peak_heights[i+1] - peak_heights[i])/np.nanmax((peak_heights[i],peak_heights[i+1])) > 0.2:
                    bad_inds.append(i)
                    bad_inds.append(i+1)



            #check if environment around line is low enough
            max_environment_height = 0.2 * peak_heights[i]

            #min_dist//2 > 3 * median_sigma

            min_idx1 = np.max((0, good_peaks[i] - int(min_dist//2)))
            min_idx2 = np.max((0, good_peaks[i] - int(3 * median_sigma)))
            max_idx1 = np.min((len(order.flux) -1, good_peaks[i] + int(3 * median_sigma)))
            max_idx2 = np.min((len(order.flux) -1, good_peaks[i] + int(min_dist//2)))

            #environment should be way smaller than peak!
            env = np.append(order.flux[min_idx1:min_idx2], order.flux[max_idx1:max_idx2])

            median_env = np.nanmedian(env)

            if np.any(env > max_environment_height + median_env):
                bad_inds.append(i)

    #peaks might be added twice, remove duplicates
    bad_inds = np.unique(bad_inds)

    #delete bad peaks
    if len(bad_inds) > 0:
        good_peaks = np.delete(good_peaks, bad_inds)

    #plot, if requested
    if datashare.reduction_parameters.plot_ThArGoodPeaks:
        fig, axs = plt.subplots(figsize=(16,9))

        axs.plot(order.pixels, order.flux)
        flux_max = np.nanmax(order.flux)
        for peak in good_peaks:
            axs.vlines(peak, ymin=1.05 * flux_max, ymax = 1.15 * flux_max)

        axs.set_xlabel('Pixels')
        axs.set_ylabel('Spectrum in a.u.')

        axs.set_title('Found peaks')

        plt.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = plotutilities.getnextfilename(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_ThArGoodPeaks", ".png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')
            plt.show()

        plt.close()


    return good_peaks


def PeakRowFit(line, saturation_limit=60000):
    """
    # Fit peak row by row and determine center of each row
    # Get tilt of peak by fitting centers with straight line
    #
    # :param line: 2D numpy ndarray, image of ThAr peak
    # :param saturation_limit: float, values above this value are saturated and will be ignored
    #
    # :return tilt: Tilt of line against x-axis in rad
    # :return tilt_err: Error of tilt
    """
    nrows, ncols = line.shape


    mids    = np.zeros(nrows)
    weights = np.zeros(nrows)

    x = np.arange(ncols)

    #fit row by row with a gaussian
    for row in range(nrows):

        data = line[row, :].astype(float)

        #do not use oversaturated pixels
        fit_sigmas = np.ones_like(data)
        fit_sigmas[data > saturation_limit] = np.inf      #big value


        #fit with gaussian

        #guess parameters, A, x0, sigma
        p0 = [np.max(data), ncols/2., ncols/4.]

        #bounds, (lower_bounds), (upper_bounds)
        bounds = ((0, 0, 0), (np.inf, ncols, ncols/2.))

        #print(x, data, p0, bounds, fit_sigmas)
        try:
            #popt, pconv = optimize.curve_fit(FitFunctions.gaussian, x, data, p0=p0, bounds=bounds, sigma=fit_sigmas)
            popt, pconv = optimize.curve_fit(FitFunctions.gaussian, x, data)#, p0=p0, sigma=fit_sigmas)
        except:
            return 0, 2 * np.pi

        #get center and weight of row. weights are correlated to signal
        mids[row]    = popt[1]
        weights[row] = np.square(np.sum(data))

    weights = np.maximum(weights, 1e-10)

    #fit centers with a straight line. Here y-axis is used as "the x-axis"
    y = np.arange(nrows)

    #guess parameters. a, b
    p0 = (0, np.median(mids))

    #bounds (lower_bounds), (upper_bounds)
    bounds = ((-np.inf, 0), (np.inf, ncols))

    try:
        popt, pconv = optimize.curve_fit(FitFunctions.linear, y, mids, p0=p0, bounds=bounds, sigma=1./weights)
        perr = np.sqrt(np.diag(pconv))
    except:
        return 0, 2*np.pi

    #get tilt
    a     = popt[0]
    a_err = perr[0]

    tilt  = np.arctan(a)

    #arbitrary limit, but usually a is between 0 and 2
    if np.isnan(a_err) or a_err is None or a_err > 10:
        tilt_err = 2 * np.pi
    else:
        tilt_err = a_err / (a**2 +1)

    tilt_err = np.max((tilt_err, 1e-10))

    #print('')
    #plt.imshow(line)
    #plt.plot(FitFunctions.linear(y, *popt), y)
    #plt.show()

    return tilt, tilt_err


# some useful methods
# Modified gaussian
def ModGauss(x, sigma, p, alpha):
    return np.exp( -0.5 * (np.power(np.abs(x), p)/np.power(np.abs(sigma), p))) * (1 + erf(alpha * np.abs(x)/ np.sqrt(2)))

# 2D gaussian
def gauss(x,y, amplitude=1., centerx=0., centery=0., sigmax=1., sigmay=1., px=2., py=2., alphax=0., alphay=0., rotation=0., q=1.):
    xp = (x - centerx)*np.cos(rotation)*q + (y - centery)*np.sin(rotation)/q
    yp = (x - centerx)*np.sin(rotation)*q + (y - centery)*np.cos(rotation)/q

    ret = amplitude * ModGauss(xp, sigmax, px, alphax) * ModGauss(yp, sigmay, py, alphay)

    if np.isnan(ret).any():
        return np.zeros_like(X)
    else:
        return ret


def fit_ThAr_lines(image, Fiber_traces, Peaks_list, npools=None, degx=2, degy=2, use_ellipses=False, rel_heights=[0.25, 0.33,0.5, 0.66, 0.75], contour_method='plt'):
    """
    # Get shape of the peaks of a ThAr calibration image to better extract the science spectra.
    # At the moment we can only correct tilted lines, but full PSF modelling will be added in the future.
    # The results will be stored in Fiber_traces, which will be returned
    #
    # :param image: Image object, ThAr image to extract peaks from
    # :param Fiber_traces: Fiber_traces object, contains the location of single orders
    # :param Peaks_list: list of list, contains indices of peaks for each order
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param degx: int, polynomial degree of tilt interpolation over the order (default 2)
    # :param degy: int, polynomial degree of tilt interpolation between orders (default 2)
    # :param use_ellipses: boolean, whether to use ellipses or row fitting to estimate the tilt of the peaks (default false)
    # :param rel_heights: list of floats, passed to get_equipotential_line, take a look there, only used if use_ellipses is True (default [0.25, 0.33,0.5, 0.66, 0.75])
    # :param contour_method: str, passed to get_equipotential_line, take a look there, only used if use_ellipses is True (default 'plt')
    #
    # :return Fiber_traces: Fiber_traces object, same as input but updated with the informations about the peak shape. This object will then be used at the extraction
    """

    # ThAr image data
    data   = image.data
    errors = image.errors

    if errors is None:
        gain = image.gain
        RON  = image.RON

        errors = np.sqrt(np.abs(data* gain) + np.power(RON, 2.)) / gain

    # make copy just to be sure
    Fiber_traces = Fiber_traces.copy()
    Peaks_list   = Peaks_list.copy()

    if len(Peaks_list) != len(Fiber_traces):
        raise ValueError('Peaks_list must have length {}!'.format(len(Fiber_traces)))

    if npools is None:
        npools = datashare.reduction_parameters.npools


    #go through orders/traces
    with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera)) as pool:
        try:
            args = [[trace_nr, data, errors, Fiber_traces, Peaks_list, use_ellipses, rel_heights, contour_method] for trace_nr in range(len(Fiber_traces))]

            results = pool.map(_fit_tilt_singleorder, args)
        finally:
            pool.close()
            pool.join()


    #fit tilt
    all_orders    = np.array([])
    all_peaks     = np.array([])
    all_tilts     = np.array([])
    all_tilt_errs = np.array([])
    all_sums      = np.array([])

    for ind in range(len(results)):
        trace_nr, good_peaks, window_sums, order_rotations, order_rotations_errs = results[ind]

        all_orders    = np.concatenate((all_orders   , np.ones_like(good_peaks) * trace_nr))
        all_peaks     = np.concatenate((all_peaks    , good_peaks))
        all_tilts     = np.concatenate((all_tilts    , order_rotations))
        all_tilt_errs = np.concatenate((all_tilt_errs, order_rotations_errs))
        all_sums      = np.concatenate((all_sums, window_sums))


    tilt_eval, tilt_errs = _fittilt(all_orders, all_peaks, all_tilts, all_tilt_errs, all_sums, len(Fiber_traces), data.shape[1], degx, degy)

    for ind in range(tilt_eval.shape[0]):
        trace = Fiber_traces.traces[ind]

        #update trace
        trace.tilt     = tilt_eval[ind,:]
        trace.tilt_err = tilt_errs[ind]

        Fiber_traces.traces[ind] = trace


    return Fiber_traces



def _fit_tilt_singleorder(args):
    trace_nr, data, errors, Fiber_traces, Peaks_list, use_ellipses, rel_heights, contour_method = args


    x_range = np.arange(data.shape[1])

    trace = Fiber_traces.traces[trace_nr]

    order_peaks = Peaks_list[trace_nr]
    trace_sigma = trace.sigma

    #default value of ThAr peak sigma is 2
    if trace_sigma < 0:
        trace_sigma = 2


    order_rotations      = []
    order_rotations_errs = []
    order_sigmas         = []
    order_sigmas_errs    = []
    window_sums          = []

    good_peaks           = []

    fit_results          = []

    #go through peaks
    for peak in order_peaks:
        #cut out window of peak
        window_width = 3 * trace_sigma
        peak_center  = trace.center_at_pixel(peak)

        min_idx_x = int(np.max((0, peak - window_width)))
        min_idx_y = int(np.max((0, peak_center - window_width)))
        max_idx_x = int(np.min((data.shape[1] -1, peak + window_width +1)))
        max_idx_y = int(np.min((data.shape[0] -1, peak_center + window_width +1)))

        window        = data[min_idx_y:max_idx_y, min_idx_x:max_idx_x]
        window_errors = errors[min_idx_y:max_idx_y, min_idx_x:max_idx_x]

        #weights. Equal weights for now. Weights propto SNR did not work, as Peaks are not really gaussian and top will disturb the shape. Maybe TODO
        #weights = np.square(window/window_errors)
        #weights[window <= 0]        = 0
        #weights[window_errors <= 0] = 0

        weights = np.ones_like(window)

        x = np.arange(max_idx_x - min_idx_x) + (peak - (max_idx_x - min_idx_x)//2)
        y = np.arange(max_idx_y - min_idx_y) + (peak_center - (max_idx_y - min_idx_y)//2)

        X,Y = np.meshgrid(x,y)

        try:
            #use ellipses to get tilt of line. Not recommended anymore
            if use_ellipses:
                #get initial values from contours
                lines = get_equipotential_line(window, rel_height=rel_heights, method=contour_method)

                #multiple relative heights, use all of them as indepentend measurements
                if type(rel_heights) is list or isinstance(rel_heights, np.ndarray):
                    #fit ellipses
                    params = np.stack([fit_ellipses(x,y) for (x,y) in lines])

                    #heighest rel. height first, invert order
                    rms = params[:, -1]
                    params = params[:, :-1]

                    #calculate mean parameters
                    #different weights, rms seams to be best way
                    #ellipse_weights = np.array([len(x) for (x,y) in lines]) * np.power(np.array(rel_heights), 2.)[::-1]
                    #ellipse_weights = np.ones(params.shape[0])
                    ellipse_weights = 1./np.power(rms, 2.)

                    mean_params = []
                    mean_errs   = []

                    #calculate mean of all parameters
                    for idx in range(len(params[0])):
                        values = np.array([p[idx] for p in params])

                        #mean_param = np.average(values, weights=ellipse_weights)
                        #param_err  = np.sqrt(np.average(np.power(values - mean_param, 2.), \
                        #                                weights=ellipse_weights))

                        #sort out strong missfits
                        good_inds = []
                        for i in range(len(values) -1):
                            #sort out value if error is much better without this value
                            new_idx = np.array([j for j in np.arange(len(values)) if j != i])
                            new_mean_param = np.average(values[new_idx], weights=ellipse_weights[new_idx])
                            #new_mean_param = np.median(values[new_idx])
                            new_err = np.sqrt(np.average(np.power(values[new_idx] - new_mean_param, 2.), \
                                                        weights=ellipse_weights[new_idx]))

                            if np.abs(values[i] - new_mean_param) < 5 * new_err:
                                good_inds.append(i)

                        good_inds = np.array(good_inds)

                        if len(good_inds) > 0:
                            mean_param = np.average(values[good_inds], weights=ellipse_weights[good_inds])
                            #mean_param = np.median(values[good_inds])
                            param_err  = np.sqrt(np.average(np.power(values[good_inds] - mean_param, 2.), \
                                                            weights=ellipse_weights[good_inds]))

                            mean_params.append(mean_param)
                            mean_errs.append(param_err)
                        else:
                            mean_params.append(0)
                            mean_errs.append(-9999)



                    mean_params = np.array(mean_params)
                    mean_errs   = np.array(mean_errs)

                #just one relative height, much easier
                else:
                    params = fit_ellipses(lines[0],lines[1])
                    mean_params = params[:-1].copy()

                    mean_errs = np.zeros_like(mean_params)


                centerx = mean_params[0] + min_idx_x
                centery = mean_params[1] + min_idx_y

                eccentricity = mean_params[4]
                tilt         = - mean_params[5]
                tilt_err     = np.abs(mean_errs[5])

                if tilt_err == 0:
                    tilt_err = 2 * np.pi

            #use fit along rows, recommended
            else:
                try:
                    tilt, tilt_err = PeakRowFit(window)
                except:
                    tilt     = 0
                    tilt_err = 2 * np.pi

        except np.linalg.LinAlgError as e:
            #print(e)
            continue
        #max error of 15 degrees
        if tilt_err < 15 * np.pi / 180:
            order_rotations.append(tilt)
            order_rotations_errs.append(tilt_err)
            good_peaks.append(peak)
            window_sums.append(np.sum(window))


    order_rotations      = np.array(order_rotations)
    order_rotations_errs = np.array(order_rotations_errs)
    good_peaks           = np.array(good_peaks)
    window_sums          = np.array(window_sums)




    return (trace_nr, good_peaks, window_sums, order_rotations, order_rotations_errs)


def _fittilt(orders, pixels, tilts, tilt_errs, window_sums, maxordnr, maxpixnr, degx, degy, niter_max = 50):
    #normalize orders and pixels
    orders_norm = (2 * orders/maxordnr) - 1
    pixels_norm = (2 * pixels/maxpixnr) - 1

    tilt_errs   = np.clip(tilt_errs  , a_min=1e-10, a_max=2*np.pi)
    window_sums = np.clip(window_sums, a_min=1e-10, a_max=np.inf)

    #weights = 1./np.square(tilt_errs)
    weights = window_sums / tilt_errs

    W = np.sqrt(weights)[:, np.newaxis]

    last_inds = np.arange(len(orders))


    tilt_eval      = np.zeros(shape=(maxordnr, maxpixnr))
    tilt_errs_eval = np.ones(shape=maxordnr) * 2 * np.pi

    #2D fit
    if degy >= 0:
        niter = 0
        #continiously fit and remove outliners till no more outliners are detected
        while True:
            niter += 1

            #fit chebyshev polynomials
            vander = np.polynomial.chebyshev.chebvander2d(pixels_norm[last_inds], orders_norm[last_inds], [degx, degy])

            W_vander = W[last_inds, :] * vander
            W_tilts  = np.sqrt(weights[last_inds]) * tilts[last_inds]

            coeffs, residuals, rank, sv = np.linalg.lstsq(W_vander, W_tilts, rcond=None)
            coeffs = coeffs.reshape(degx+1, degy+1)

            #determine outliners
            residuals = np.polynomial.chebyshev.chebval2d(pixels_norm, orders_norm, coeffs) - tilts
            good_inds = np.asarray(np.abs(residuals) < 5 * np.median(np.abs(residuals))).nonzero()[0]

            #update used indicies or break, if no new outliners are found
            if len(good_inds) < 2 * degx * degy or np.array_equal(last_inds, good_inds) or niter >= niter_max:
                break
            else:
                last_inds  = good_inds



        orders_norm_eval = (2 * np.arange(maxordnr)/maxordnr) - 1
        pixels_norm_eval = (2 * np.arange(maxpixnr)/maxpixnr) - 1

        for i, ordnr_norm in enumerate(orders_norm_eval):
            tilt_eval[i, :] = np.polynomial.chebyshev.chebval2d(pixels_norm_eval, np.ones_like(pixels_norm_eval) * ordnr_norm, coeffs)

            inds = np.asarray(orders == i).nonzero()
            if len(inds) > 0:
                ord_tilts      = tilts[inds]
                ord_tilts_eval = np.polynomial.chebyshev.chebval2d(pixels_norm[inds], orders_norm[inds], coeffs)
                tilt_errs_eval[i]   = np.min((2 * np.pi, np.sqrt(np.mean(np.square(ord_tilts - ord_tilts_eval)))))

    #just do a 1D fit
    else:
        pixels_norm_eval = (2 * np.arange(maxpixnr)/maxpixnr) - 1

        for ordnr in range(maxordnr):
            inds = np.asarray(orders == ordnr).nonzero()[0]

            if len(inds) < 2 * degx:
                tilt_eval[ordnr, :]   = 0
                tilt_errs_eval[ordnr] = 2 * np.pi
                continue

            ord_pixels  = pixels_norm[inds]
            ord_tilts   = tilts[inds]
            ord_weights = weights[inds]

            last_inds = np.arange(len(inds))

            niter = 0
            #continiously fit and remove outliners till no more outliners are detected
            while True:
                niter += 1

                #fit chebyshev polynomials
                coeffs = np.polynomial.chebyshev.chebfit(ord_pixels[last_inds], ord_tilts[last_inds], deg=degx, w=ord_weights[last_inds])

                #determine outliners
                residuals = np.polynomial.chebyshev.chebval(ord_pixels, coeffs) - ord_tilts
                good_inds = np.asarray(np.abs(residuals) < 5 * np.median(np.abs(residuals))).nonzero()[0]

                #update used indicies or break, if no new outliners are found
                if len(good_inds) < 2 * degx or np.array_equal(last_inds, good_inds) or niter >= niter_max:
                    break
                else:
                    last_inds  = good_inds



            tilt_eval[ordnr, :] = np.polynomial.chebyshev.chebval(pixels_norm_eval, coeffs)

            ord_tilts_eval        = np.polynomial.chebyshev.chebval(ord_pixels, coeffs)
            tilt_errs_eval[ordnr] = np.min((2 * np.pi, np.sqrt(np.mean(np.square(ord_tilts - ord_tilts_eval)))))


    if datashare.reduction_parameters.plot_ThArTiltFit:
        for i in np.arange(maxordnr):
            plt.plot(np.arange(maxpixnr), tilt_eval[i,:] + i, color='black')


            inds = np.asarray(orders == i).nonzero()
            plt.errorbar(pixels[inds], tilts[inds] + i, yerr=tilt_errs[inds], color='red', fmt='o', ms=2)

        if datashare.reduction_parameters.save_plots:
            filename = plotutilities.getnextfilename(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_ThArtiltfit", ".png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')
            plt.show()

        plt.close()

    return tilt_eval, tilt_errs_eval


def interpolate_tilts(Fiber_traces, nord=2, nsigmas=5):
    """
    # Interpolate mean tilts from measured ThAr line tilts over all orders (cross-dispersion interpolation) using chebyshev polynimials
    # We assume a constant tilt in one order
    #
    # :param Fiber_traces: Fiber_traces object, contains the information about the tilts
    # :param nord: int, polynomial degree of interpolation (default 2)
    # :param nsigmas: float, multiplyier on median of residuals used for sigma clipping (default 5)
    #
    # :return Fiber_traces: Fiber_traces object with interpolated tilts
    """

    #copy traces just to be sure
    Fiber_traces = Fiber_traces.copy()

    # get all tilts and tilt errors
    tilts = np.array([trace.tilt for trace in Fiber_traces.traces])
    tilt_errs = np.array([trace.tilt_err for trace in Fiber_traces.traces])

    # fit parameters
    x = np.arange(len(tilts))
    use_inds = np.arange(len(tilts))

    # delete indices where no tilt is provided
    use_inds = np.delete(use_inds, np.where(np.logical_or((tilts == 0.0), np.isnan(tilts)))[0])

    # continue fitting as long as there are outliners
    while True:
        # no weighting were
        weights = np.ones_like(use_inds)

        # fit tilts
        fit = np.polynomial.chebyshev.chebfit(x[use_inds], tilts[use_inds], deg=nord, w= weights)

        #sigma clipping
        residuals = tilts[use_inds] - np.polynomial.chebyshev.chebval(use_inds, fit)

        bad_inds = np.where(np.abs(residuals) > nsigmas * np.median(np.abs(residuals)))[0]

        # if no outliners found or too less remaining measurements: Break
        if len(bad_inds) < 1 or len(use_inds) < nord + 3:
            break
        # else remove outliners
        else:
            use_inds = np.delete(use_inds, bad_inds)


    # Plot, is requested
    if datashare.reduction_parameters.plot_ThArTiltInterpolation:
        x = np.arange(len(Fiber_traces.traces))
        plt.plot(x, tilts)
        plt.plot(x, np.polynomial.chebyshev.chebval(x, fit))

        if datashare.reduction_parameters.save_plots:
            filename = plotutilities.getnextfilename(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_ThArTiltInterpolation", ".png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('')
            plt.show()

        plt.close()

    # replace mean tilts by interpolated values
    for ind in range(len(Fiber_traces.traces)):
        Fiber_traces.traces[ind].tilt     = np.polynomial.chebyshev.chebval(ind, fit)
        Fiber_traces.traces[ind].tilt_err = 2 * np.pi

    # return updated Fiber_traces
    return Fiber_traces



