###############################
# Created 2025/10/16
#
# Author: Lukas Stock
#
##############################

import numpy as np
import numbers
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import signal, ndimage, interpolate, optimize


from pyvesta import Spectra
from pyvesta import datashare
from pyvesta import plot_utilities
from pyvesta import CCD_corrections


def init_pools(reduction_parameters, instrument, camera, current_filename):
    datashare.reduction_parameters = reduction_parameters
    datashare.instrument           = instrument
    datashare.camera               = camera
    datashare.current_filename     = current_filename


#return whether an index pair (x,y) is within range
def _valid_index(x,y, max_x, max_y):
    return ((((x > -0.5)        & \
            (y > -0.5))         & \
            (x < max_x + 0.5))  & \
            (y < max_y + 0.5))


def _filterordershapematrix(ordershape_matrix, rel_threshold = 0.05, fill_value=1e-10):
    """
    # Filter ordershape matrix (ordershape for each pixel) such that small peaks at the side of the maximum peaks are ignored and all peak values
    #
    # :param ordershape_matrix: 2D numpy array, shape (ylen, npix): ordershape matrix
    # :param rel_threshold: float, relative threshold (to maximum). If any value drops below this value, all outer values will be set to fill_value (default 0.05)
    # :param fill_value: float, value to fill masked elements with (default 1e-10)
    #
    # :returns result: 2D numpy array, shape (ylen, npix): Filtered ordershape matrix
    """

    center = (ordershape_matrix.shape[0] // 2) + 1
    center = np.broadcast_to(center, ordershape_matrix.shape[1])


    rows = np.arange(ordershape_matrix.shape[0])[:, None]
    center_rows = center[None, :]

    col_max = np.max(ordershape_matrix, axis=0, keepdims=True)
    threshold = rel_threshold * col_max

    is_below = (ordershape_matrix < threshold)
    is_right = (rows >= center_rows)
    is_left  = (rows <= center_rows)

    below_right = np.where(is_right, is_below, False)
    accum_right = np.logical_or.accumulate(below_right, axis=0)

    below_left = np.where(is_left, is_below, False)
    accum_left = np.logical_or.accumulate(below_left[::-1, :], axis=0)[::-1, :]     # from right to left -> reverse order

    mask   = np.where(is_right, accum_right, accum_left)
    result = np.where(mask, fill_value, ordershape_matrix)

    row_sums = np.sum(result, axis=0, keepdims=True)
    row_sums[row_sums == 0] = 1

    result = result / row_sums

    return result


def _filtersingleordershape(ordershape, rel_threshold = 0.05, fill_value=1e-10):
    """
    # Filter individual ordershape such that small peaks at the side of the maximum peaks are ignored and all peak values
    #
    # :param ordershape: numpy array, shape (ylen,): ordershape
    # :param rel_threshold: float, relative threshold (to maximum). If any value drops below this value, all outer values will be set to fill_value (default 0.05)
    # :param fill_value: float, value to fill masked elements with (default 1e-10)
    #
    # :returns result: numpy array, shape (ylen,): Filtered ordershape
    """

    center = (len(ordershape)// 2) + 1

    y_range = np.arange(len(ordershape))

    ord_max = np.max(ordershape)
    threshold = rel_threshold * ord_max

    is_below = (ordershape < threshold)
    is_right = (y_range >= center)
    is_left  = (y_range <= center)

    below_right = np.where(is_right, is_below, False)
    accum_right = np.logical_or.accumulate(below_right)

    below_left = np.where(is_left, is_below, False)
    accum_left = np.logical_or.accumulate(below_left[::-1])[::-1]     # from right to left -> reverse order

    mask   = np.where(is_right, accum_right, accum_left)
    result = np.where(mask, fill_value, ordershape)

    ord_sum = np.sum(result)

    if ord_sum == 0:
        ord_sum = 1

    result = result / ord_sum

    return result

def create_weights(Image, Trace_obj, default_tilt=None, npools=None):
    """
    # Create the extraction mask from a flat image, taking tilted lines into account. If case of no tilt this is just a 1D array of ones.
    # Only tilt mask, no cosmic filtering etc here, as they can differ from image to image
    # This mask will be used for all images from that night
    # Used multithreading to work on multiple orders simultaneously
    #
    # Weights are defined via a 2D array of weights for each pixel as well as a set of pixel coordinates to which pixels in the image this window belongs
    #
    # :param Image: Image object, flat image to extract weights from
    # :param Trace_obj: Fiber_traces object, contains the information about the position of the spectral orders
    # :param default_tilt: float or numpy.ndarray of floats or None. if None, will use the tilt contained in Trace_obj. If float, will assume tilt is constant over the order. When ndarray, length has to be the same as pixel length, will use default_tilt[pixel] as tilt, which can change over the order
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    #
    # :returns Weights: ExtractionWeights object, contains the weights for weighted extracion
    """

    #get data from image
    image  = Image.data.copy()
    errors = Image.errors.copy()
    gain   = Image.gain
    RON    = Image.RON

    # number of threads from datashare
    if npools is None:
        npools = datashare.reduction_parameters.npools

    #get number of fibers, orders and pixels
    nr_of_fibers = Trace_obj.nr_of_fibers()
    nr_of_orders = np.max([len(trace) for trace in Trace_obj.traces])
    nr_of_pixels = image.shape[1]

    #initialize weights
    Weights = Spectra.ExtractionWeights(nr_of_fibers, nr_of_orders, nr_of_pixels)

    #if default_tilt is a list (for each fiber), but dimensions do not match, just ignore it /set it to None
    if type(default_tilt) is list or isinstance(default_tilt, np.ndarray):
        if len(default_tilt) != nr_of_fibers:
            default_tilt = [None] * nr_of_fibers

    #each fiber will have the same tilt
    else:
        default_tilt = [default_tilt] * nr_of_fibers

    #go through all fibers
    for fiber_nr in range(nr_of_fibers):
        Fiber_trace = Trace_obj.traces[fiber_nr]

        #use multiprocessing to speed things up, go through orders
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            args = [(i, image.copy(), errors, Fiber_trace, default_tilt[i%nr_of_fibers]) for i in range(len(Fiber_trace))]

            result = pool.map(_trace_weights, args)

            #set weights for the single orders
            for order_nr in range(Fiber_trace.nr_of_orders):
                for x in range(nr_of_pixels):
                    Weights.set_weights(fiber_nr, order_nr, x, result[order_nr][0][x])
                    Weights.set_boundaries(fiber_nr, order_nr, x, result[order_nr][1][x])

            #plot weights for one specific pixel (roughly middle of spectrum) to briefly check weights calculation
            if datashare.reduction_parameters.plot_Weights:
                mid_order = len(result) // 2
                mid_pix   = len(result[mid_order][0]) // 2

                plt.imshow(result[mid_order][0][mid_pix])

                if datashare.reduction_parameters.save_plots:
                    filename = os.path.join(datashare.reduction_parameters.plot_dir, "Weights.png")
                    plt.savefig(filename)

                if datashare.reduction_parameters.show_plots:
                    print('')
                    plt.show()

                plt.close()

    Weights = CreateOrderShapes(Image, Trace_obj, Weights, npools=npools)

    #return final weights object
    return Weights


def veryfast_extract(image, Trace_obj):
    """
    # Very fast extraction of echelle spectra, not for scientific use
    # Used for first rough ThAr line extraction to find lines for PSF modelling / tilt calculation
    # Just use center pixels (and maybe neighbor pixels, if trace not perfectly centered)
    #
    # :param image: Image object, image of spectrum to extract
    # :param Trace_obj: Fiber_traces object, contains the information about the order traces
    #
    # :return spectralist: SpectraList object, contains a RawSpectrum object for each fiber
    """

    #read image
    data   = image.data
    errors = image.errors

    #calculate errors if not yet done
    if errors is None:
        gain = image.gain
        RON  = image.RON

        errors = np.sqrt(np.abs(data * gain) + np.power(RON, 2.)) / gain

    all_spectra = []

    #go through all fibers
    for fiber in range(Trace_obj.nr_of_fibers()):
        Fiber_trace = Trace_obj.traces[fiber]

        x_range = np.arange(data.shape[1])

        spectrum = Spectra.RawSpectrum()

        #go through orders
        for order, trace in enumerate(Fiber_trace.all_traces()):
            #compute centers of this trace. Centers are the positions of the trace in y-direction / cross-dispersion direction
            trace.compute_centers(x_range)
            Centers   = trace.Centers

            # a center of x + 0.5 corresponds fully to pixel x
            # a center of x + 0.2 corresponds to (1- (0.5-0.2)) = 0.7 to pixel x and to (0.5 - 0.2) = 0.3 to pixel x-1  (0.2 < 0.5)
            # a center of x + 0.9 corresponds to (1- (0.9-0.5)) = 0.6 to pixel x and to (0.9 - 0.5) = 0.4 to pixel x+1  (0.9 > 0.5)

            #difference between center and pixels
            Center_offsets = Centers - (Centers.astype(int) + 0.5)
            negative_inds  = np.where(Center_offsets < 0)[0]
            positive_inds  = np.where(Center_offsets >= 0)[0]

            #coefficients for center pixel
            Center_coeffs = np.zeros_like(Centers)
            Center_coeffs[negative_inds] = 1 + Center_offsets[negative_inds] #Center_offsets < 0
            Center_coeffs[positive_inds] = 1 - Center_offsets[positive_inds] #Center_offsets > 0

            #coefficients for neighbor pixels
            Neighbor_coeffs = np.zeros_like(Centers)
            Neighbor_coeffs[negative_inds] = - Center_offsets[negative_inds] #Center_offsets < 0
            Neighbor_coeffs[positive_inds] = Center_offsets[positive_inds]   #Center_offsets > 0

            #convert to integers to use centers as indices
            Center_pixels   = Centers.astype(int)
            Neighbor_pixels = Centers.astype(int)

            #neighbors are pixels above / below the center
            Neighbor_pixels[negative_inds] -= 1
            Neighbor_pixels[positive_inds] += 1

            #now extract, just sum up center and neighbor with corresponding coefficients
            extracted_spectrum = Center_coeffs * data[Center_pixels, x_range] + Neighbor_coeffs * data[Neighbor_pixels, x_range]
            extracted_errors   = np.sqrt(np.power(Center_coeffs * errors[Center_pixels, x_range], 2.) + np.power(Neighbor_coeffs * errors[Neighbor_pixels, x_range], 2.))

            #create order from extracted spectrum
            order_obj = Spectra.RawSpectralOrder(x_range, extracted_spectrum, errors=extracted_errors, ordernr=order)

            #add order to spectrum
            spectrum.addOrder(order_obj)

            #plot if requested
            if datashare.reduction_parameters.plot_Veryfastextraction:
                filter_width = 5

                #moving average, reduce noise
                mean_spec = np.convolve(extracted_spectrum , np.ones(filter_width)/filter_width, mode='valid')
                mean_errs = np.convolve(extracted_errors, np.ones(filter_width)/filter_width, mode='valid')

                fig, axs = plt.subplots(2,2, figsize=(16,8))
                fig.suptitle('Order {}, Mid Center {}'.format(order, int(np.nanmax(Centers))))

                axs[0,0].plot(x_range, extracted_spectrum , color='black', label='Extracted spectrum')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec + 2 * mean_errs, color='red')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec - 2 * mean_errs, color='red', label=r'2 $\sigma$ error range')
                axs[0,1].plot(x_range,extracted_errors)
                #axs[0,1].plot(x_range[:-2], np.diff(extracted_spectrum, n=2), color='black', label='Extracted spectrum')

                axs[1,0].plot(x_range[filter_width//2:-filter_width//2 + 1], np.abs(extracted_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs)
                axs[1,0].hlines([1,2,3], filter_width//2, len(extracted_spectrum ) - filter_width//2, color='red', linestyle='dashed')
                axs[1,1].hist(np.abs(extracted_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs, bins=np.linspace(0,5,10), cumulative=True, density=True)
                axs[1,1].hlines([0.6827, 0.9545, 0.9972], 0, 5, color='red', linestyle='dashed')

                axs[0,0].set_title('Extracted spectrum')
                axs[0,1].set_title('Errors')
                #axs[0,1].set_title('Diff')
                axs[1,0].set_title('Variances from mean devided by sigma')
                axs[1,1].set_title('Error histogramm')
                axs[0,0].legend()

                axs[0,0].set_xlabel('Pixel')
                axs[0,0].set_ylabel('spectrum in a.u.')
                axs[0,1].set_xlabel('Pixel')
                axs[0,1].set_ylabel('errors in a.u.')
                axs[1,0].set_xlabel('Pixel')
                axs[1,0].set_ylabel('normalized variance')
                axs[1,1].set_xlabel(r'$\sigma$')
                axs[1,1].set_ylabel('density')

                plt.tight_layout()

                if datashare.reduction_parameters.save_plots:
                    filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_verfastextracted_order{}.png".format(order))
                    plt.savefig(filename, dpi=300)

                if datashare.reduction_parameters.show_plots:
                    plt.show()

                plt.close(fig)

        #add spectrum to list
        all_spectra.append(spectrum)

    # create SpectraList object from list of spectra
    spectralist = Spectra.SpectraList(spectra=all_spectra)

    return spectralist


def mask_cosmics(image, nsigmas=20, dx=1):
    #TODO: NOT WORKING YET FOR THARs!!!

    """
    # Mask cosmics or hotpixels found in image
    # Use sigma clipping to detect cosmics and hotpixels
    #
    # :param image: Image object, image to find cosmics in
    # :param nsigmas: float, multiples of standard deviation. Pixels with a deviation from the mean larger than this value will be marked as cosmics (default 20)
    # :param dx: int, quadratic windows of size 2*dx +1 will be used to median convolve the image (default 1)
    #
    # :return mask: numpy ndarray, binary mask with same size as image, cosmics have weight 0
    """
    mask = np.ones_like(image)

    dx = int(dx)
    if dx < 1:
        dx = 1

    median_image = ndimage.median_filter(image, size=2*dx+1)

    filt = np.ones(shape=(2*dx+1, 2*dx+1))/(np.power(2*dx+1., 2.) -1)
    filt[dx, dx] = 0    #not use center pixel

    mean_image  = signal.convolve2d(image, filt, mode='same')

    mask[np.where(np.abs(median_image - image) > nsigmas* mean_image)] = 0

    return mask

# TODO: Maybe new with deviation friom local median? Would not work with ThArs
def filter_cosmics_spec(flux, threshold = None, iterations=5, median_fac=5, groundlevel=None):
    """
    # Filter cosmics in flux
    # Replace cosmics by mean of neighbor pixels
    # A cosmic is detected, when he deviation of the flux is higher than threshold and the next deviation is smaller than the negative threshold (one pixel wide cosmic)
    # Can be repeated to filter wider cosmics
    #
    # :param flux: numpy.ndarray, contains the spectral flux
    # :param iterations: int, number of iterations
    # :param threshold: float, minimal threshold of deviation to detect cosmics. Will be set to median_fac * median deviation if not set (default None)
    # :param median_fac: float, used to calculate threshold if not specified.  Threshold will be set to median_fac * median deviation if not specified.
    # :param groundlevel: float, flux below this level will not be filtered (so no shallow absorption lines are affacted). Will not be used if None (default None)
    #
    # :return new_flux: numpy.ndarray, flux with filtered out cosmics
    """

    # copy flux
    orig_flux = flux.copy()

    # plot original flux, if requested
    if datashare.reduction_parameters.plot_speccosmicfilter:
        plt.plot(np.arange(len(flux)), orig_flux, label='original')

    # specify threshold
    init_threshold = threshold


    new_flux = flux.copy()
    # go through iterations
    for j in range(iterations):
        # set result from previous iteration as starting point and calculate deviation
        flux = new_flux.copy()
        diff = np.diff(flux)

        # this will be the filtered flux
        new_flux = flux.copy()

        #calculate new threshold, if not specified
        if init_threshold is None:
            # get median noise
            median_noise = ndimage.median_filter(np.abs(diff), size=10)
            threshold = median_fac * median_noise
        else:
            threshold = np.ones_like(flux) * init_threshold

        # go through pixels. Skip first and last pixel, as we cannot get deviations there
        for i in range(1, len(diff)-1):

            if (diff[i] > threshold[i] and -diff[i-1] < threshold[i-1]) or \
                (- diff[i] > threshold[i] and diff[i-1] < threshold[i-1]):
                #this is a cosmic, interpolate

                #if goundlevel is given, only delete values above this level
                if groundlevel is not None and flux[i+1] < groundlevel:
                    continue

                # first pixel, just use next pixel
                if i == 0:
                    new_flux[1] = flux[2]
                # replace by mean of neighbors
                elif i < len(flux) - 2:
                    new_flux[i+1] = 0.5 * (flux[i] + flux[i + 2])
                #last pixel, just use previous pixel
                else:
                    new_flux[i+1] = flux[i]

        # plot, if requested
        if datashare.reduction_parameters.plot_speccosmicfilter:
            plt.plot(np.arange(len(flux)), new_flux, label='iteration {}'.format(j))

    # plot, if requested
    if datashare.reduction_parameters.plot_speccosmicfilter:
        plt.legend()

        if datashare.reduction_parameters.save_plots:
            filename = plot_utilities.getnextfilename(reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_speccosmicfilter", '.png')
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            plt.show()

        plt.close()
    # return filtered flux
    return new_flux


#TODO, FIXME: Immernoch Oszillationen in den Fehlern, WARUM? Ich sehe keine Osillationen mehr im Flux
def weighted_extract(Image, Trace_obj, Weights, npools=None, filter_cosmics=False, nsigmas=3, plot=False):
    """
    # Extract the image along the given traces using weights.
    # Use ExtractionWeights for this to weight pixels relative to SNR and to account for tilted lines
    # Use multithreading to work on multiple orders simultaneously
    #
    # :param Image: Image object, contains the image to ectraxt
    # :param Trace_obj: Fiber_traces object, contains the informations about the order positions and PSF/ line tilt
    # :param Weights: ExtractionWeights object, contains the extraction weights for each pixel in each order
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param filter_cosmics: boolean, whether to filter cosmics, still experimental (default False)
    # :param nsigmas: float, multiple of errors. Deviations larger than nsigmas * errors from the expected ordershape will be filtered
    # :param plot: boolean, whether to plot the results (default False)
    #
    # :return spectralist: SpectraList object, contains a RawSpectrum object for each fiber
    """

    # get informations from image
    image  = Image.data
    errors = Image.errors
    gain   = Image.gain
    RON    = Image.RON

    # get number of threads from config
    if npools is None:
        npools = datashare.reduction_parameters.npools_extract

    # calculate errors, if those were not calculated before
    if errors is None:
        errors = np.sqrt(np.abs(data * gain) + np.power(RON, 2.)) / gain

    all_spectra = []

    # iterate over all traces
    for fiber_nr in range(Trace_obj.nr_of_fibers()):
        Fiber_trace = Trace_obj.traces[fiber_nr]

        # here the extraction happens, use multiple threads
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            try:
                args = [(image, errors, Weights, trace, fiber_nr, trace_nr, filter_cosmics, nsigmas) for trace_nr, trace in enumerate(Fiber_trace.all_traces())]

                orders = pool.map(_extract_trace, args)
            finally:
                pool.close()
                pool.join()

        # plot, if requested
        if datashare.reduction_parameters.plot_Extraction:
            filter_width = 5

            for i in range(len(orders)):
                order_spectrum = orders[i][0]
                order_errors   = orders[i][1]
                x_range = np.arange(len(order_spectrum))

                trace = Fiber_trace.get_trace(i)
                trace.compute_centers(x_range)
                Mid_center = np.max(trace.Centers)

                #moving average
                mean_spec = np.convolve(order_spectrum, np.ones(filter_width)/filter_width, mode='valid')
                mean_errs = np.convolve(order_errors, np.ones(filter_width)/filter_width, mode='valid')

                fig, axs = plt.subplots(2,2, figsize=(16,8))
                fig.suptitle('Order {}, Mid Center {}'.format(i+1, int(Mid_center)))

                axs[0,0].plot(x_range, order_spectrum, color='black', label='Extracted spectrum')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec + 2 * mean_errs, color='red')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec - 2 * mean_errs, color='red', label=r'2 $\sigma$ error range')
                axs[0,1].plot(x_range, order_errors)
                #axs[0,1].plot(x_range[:-2], np.diff(order_spectrum, n=2), color='black', label='Extracted spectrum')

                axs[1,0].plot(x_range[filter_width//2:-filter_width//2 + 1], np.abs(order_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs)
                axs[1,0].hlines([1,2,3], filter_width//2, len(order_spectrum) - filter_width//2, color='red', linestyle='dashed')
                axs[1,1].hist(np.abs(order_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs, bins=np.linspace(0,5,10), cumulative=True, density=True)
                axs[1,1].hlines([0.6827, 0.9545, 0.9972], 0, 5, color='red', linestyle='dashed')

                axs[0,0].set_title('Extracted spectrum')
                axs[0,1].set_title('Errors')
                #axs[0,1].set_title('Diff')
                axs[1,0].set_title('Variances from mean devided by sigma')
                axs[1,1].set_title('Error histogramm')
                axs[0,0].legend()

                axs[0,0].set_xlabel('Pixel')
                axs[0,0].set_ylabel('spectrum in a.u.')
                axs[0,1].set_xlabel('Pixel')
                axs[0,1].set_ylabel('errors in a.u.')
                axs[1,0].set_xlabel('Pixel')
                axs[1,0].set_ylabel('normalized variance')
                axs[1,1].set_xlabel(r'$\sigma$')
                axs[1,1].set_ylabel('density')

                plt.tight_layout()


                if datashare.reduction_parameters.save_plots:
                    filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_weightedextracted_order{}.png".format(i))
                    plt.savefig(filename, dpi=300)

                if datashare.reduction_parameters.show_plots:
                    plt.show()

                plt.close(fig)

        # create Spectrum object, copy image header to spectrum
        spectrum = Spectra.RawSpectrum()
        spectrum.header = Image.header

        # append orders to spectrum
        for i in range(len(orders)):
            # flux is first element, errors is second element
            order_spectrum = orders[i][0]
            order_errors   = orders[i][1]

            #clip negative values in spectrum, assure errors are positive
            order_spectrum = np.clip(order_spectrum, a_min=1e-10, a_max=np.inf)
            order_errors   = np.abs(order_errors)

            x_range = np.arange(len(order_spectrum))

            order_obj = Spectra.RawSpectralOrder(x_range, order_spectrum, errors=order_errors, ordernr=i)

            spectrum.addOrder(order_obj)

        # add spectrum to list
        all_spectra.append(spectrum)

    # create SpectraList object from list of spectra
    spectralist = Spectra.SpectraList(spectra=all_spectra)

    return spectralist

#extract one trace / order
#this method can be used in multiple threads simultaneously
def _extract_trace(args):
    image, errors, Weights, trace, fiber_nr, order_nr, filter_cosmics, nsigmas = args

    #pixel range
    x_range = np.arange(image.shape[1]).astype(int)

    order_spectrum = np.zeros(image.shape[1])
    order_errors   = np.zeros(image.shape[1])

    trace.compute_centers(x_range)
    Centers   = trace.Centers


    ordershape        = Weights.get_ordershape(fiber_nr, order_nr)

    #iteration over all pixels
    for x in x_range:
        center = Centers[x]

        #get weights
        tilt_weights                               = Weights.get_weights(fiber_nr, order_nr, x)
        min_idx_x, min_idx_y, max_idx_x, max_idx_y = Weights.get_boundaries(fiber_nr, order_nr, x)

        tilt_weights /= np.sum(tilt_weights)

        #we want to shift individual pixels so, that the center always is exactly the center of y_range
        mid_y       = center - min_idx_y
        order_shift = mid_y - len(ordershape)/2.


        #get extraction window
        window        = image[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]
        window_errors = errors[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]

        window_product  = tilt_weights * window

        window_rows = np.clip(np.nansum(window_product, axis=1), a_min=1e-10, a_max=np.inf)
        error_rows  = np.sqrt(np.nansum(np.square(tilt_weights * window_errors), axis=1))

        ordershape_spline = interpolate.CubicSpline(np.arange(ordershape.shape[0]), ordershape[:, x], extrapolate=False)

        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift

        ordershape_eval = ordershape_spline(ext_y_range)
        #ordershape_eval =_ordershape_int(ext_y_range)

        not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()

        ordershape_eval[not_nan_inds] = np.clip(ordershape_eval[not_nan_inds], a_min=1e-10, a_max=np.inf)

        ordershape_eval /= np.nansum(ordershape_eval)

        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift

        ordershape_eval = ordershape_spline(ext_y_range)

        #use median of upper half of values as scaling parameter
        window_median = np.median(window_rows)
        ordershape_median = np.nanmedian(ordershape_eval)

        window_median = np.median(window_rows[window_rows > window_median])
        ordershape_median = np.nanmedian(ordershape_eval[ordershape_eval> ordershape_median])

        #ordershape_eval *= np.percentile(window_rows, 90) / np.nanpercentile(ordershape_eval, 90)
        ordershape_eval *= window_median / ordershape_median

        #replace bad indicies, e.g. cosmics or hot pixels, with expected value
        nan_inds     = np.asarray(np.isnan(ordershape_eval)).nonzero()[0]
        not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()[0]

        rms = np.sqrt(np.mean(np.square(window_rows[not_nan_inds]/ordershape_eval[not_nan_inds] -1)))

        bad_inds = np.asarray(np.abs(window_rows[not_nan_inds]/ordershape_eval[not_nan_inds] - 1) > nsigmas * rms).nonzero()[0]
        bad_inds = np.unique(np.concatenate((not_nan_inds[bad_inds], nan_inds)))

        #ensure not all indices are masked
        if len(bad_inds) < len(window_rows):
            #mask bad indices
            tilt_weights_copy = tilt_weights.copy()
            tilt_weights[bad_inds, :] = 0
            tilt_weights *= np.nansum((np.sum(tilt_weights_copy, axis=1) * ordershape_eval)) / np.nansum((np.sum(tilt_weights, axis=1) * ordershape_eval))

        #if x == len(x_range) //2:
        #    plt.plot(window_rows[not_nan_inds])
        #    plt.plot(ordershape_eval[not_nan_inds], color='red')
        #    print('')
        #    plt.show()


        #while np.count_nonzero(tilt_weights > 0) > 5:
        #    tilt_weights_temp = tilt_weights.copy()
        #    tilt_weights_temp[tilt_weights_temp == 0] = np.inf

        #    tilt_weights[np.unravel_index(np.argmin(tilt_weights_temp), tilt_weights.shape)] = 0


        #tilt_weights /= np.sum(tilt_weights)

        #window_rows[bad_inds] = ordershape_eval[bad_inds]

        #print(order_nr, x,bad_inds, len(window_rows))

        #if x == len(x_range) // 2:
        #    print(bad_inds, rms)

        #TODO:How to deal with errors!?

        #sum up values
        try:
            order_spectrum[x] = np.clip(np.nansum(tilt_weights * window), a_min=1e-10, a_max=np.inf)
            order_errors[x]   = np.sqrt(np.nansum(np.square(tilt_weights * window_errors)))

            #f = 1. / ordershape_eval[not_nan_inds]
            #w = np.square(f / error_rows[not_nan_inds])
            #a = (w / f) / np.sum(w)

            #window_rows = np.clip(np.nansum(tilt_weights * window, axis=1), a_min=1e-10, a_max=np.inf)
            #error_rows  = np.clip(np.nansum(tilt_weights * window_errors, axis=1), a_min=1e-10, a_max=np.inf)

            #order_spectrum[x] = np.sum(a * f * window_rows[not_nan_inds])

            #order_errors[x] = np.sqrt(np.sum(np.sqrt(a * f * error_rows[not_nan_inds])))


        except Exception as e:
            print(e)
            order_spectrum[x] = np.nan
            order_errors[x]   = np.inf

        """
        if x >= 1500 and x <= 1600 and x%5 == 0:
            print(x)
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(window)
            axs[1].imshow(final_weights)
            plt.show()
        """

    # filter cosmics in spectrum, additionally to filter in raw image
    if filter_cosmics:
        try:
            order_spectrum = filter_cosmics_spec(order_spectrum)
        except:
            pass

    # return flux and errors
    return order_spectrum, order_errors

#TODO, FIXME: Immernoch Oszillationen in den Fehlern, WARUM? Ich sehe keine Osillationen mehr im Flux
def weighted_extract_optimal(Image, Trace_obj, Weights, npools=None, filter_cosmics=False, nsigmas=3):
    """
    # Extract the image along the given traces using weights.
    # Use ExtractionWeights for this to weight pixels relative to SNR and to account for tilted lines
    # Use multithreading to work on multiple orders simultaneously
    #
    # :param Image: Image object, contains the image to ectraxt
    # :param Trace_obj: Fiber_traces object, contains the informations about the order positions and PSF/ line tilt
    # :param Weights: ExtractionWeights object, contains the extraction weights for each pixel in each order
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param filter_cosmics: boolean, whether to filter cosmics, still experimental (default False)
    # :param nsigmas: float, multiple of errors. Deviations larger than nsigmas * errors from the expected ordershape will be filtered
    #
    # :return spectralist: SpectraList object, contains a RawSpectrum object for each fiber
    """

    # get informations from image
    image  = Image.data
    errors = Image.errors
    gain   = Image.gain
    RON    = Image.RON

    # get number of threads from config
    if npools is None:
        npools = datashare.reduction_parameters.npools_extract


    # calculate errors, if those were not calculated before
    if errors is None:
        errors = np.sqrt(np.abs(image * gain) + np.power(RON, 2.)) / gain

    all_spectra = []

    # iterate over all traces
    for fiber_nr in range(Trace_obj.nr_of_fibers()):
        Fiber_trace = Trace_obj.traces[fiber_nr]

        # here the extraction happens, use multiple threads
        with  Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            try:
                args = [(image, errors, Weights, trace, fiber_nr, trace_nr, filter_cosmics, nsigmas) for trace_nr, trace in enumerate(Fiber_trace.all_traces())]

                orders = pool.map(_extract_trace_optimal, args)
            finally:
                pool.close()
                pool.join()

        # plot, if requested
        if datashare.reduction_parameters.plot_Extraction:
            filter_width = 5

            for i in range(len(orders)):
                order_spectrum = orders[i][0]
                order_errors   = orders[i][1]
                x_range = np.arange(len(order_spectrum))

                trace = Fiber_trace.get_trace(i)
                trace.compute_centers(x_range)
                Mid_center = np.max(trace.Centers)

                #moving average
                mean_spec = np.convolve(order_spectrum, np.ones(filter_width)/filter_width, mode='valid')
                mean_errs = np.convolve(order_errors, np.ones(filter_width)/filter_width, mode='valid')

                fig, axs = plt.subplots(2,2, figsize=(16,8))
                fig.suptitle('Order {}, Mid Center {}'.format(i+1, int(Mid_center)))

                axs[0,0].plot(x_range, order_spectrum, color='black', label='Extracted spectrum')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec + 2 * mean_errs, color='red')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec - 2 * mean_errs, color='red', label=r'2 $\sigma$ error range')
                axs[0,1].plot(x_range, order_errors)
                #axs[0,1].plot(x_range[:-2], np.diff(order_spectrum, n=2), color='black', label='Extracted spectrum')

                axs[1,0].plot(x_range[filter_width//2:-filter_width//2 + 1], np.abs(order_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs)
                axs[1,0].hlines([1,2,3], filter_width//2, len(order_spectrum) - filter_width//2, color='red', linestyle='dashed')
                axs[1,1].hist(np.abs(order_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs, bins=np.linspace(0,5,10), cumulative=True, density=True)
                axs[1,1].hlines([0.6827, 0.9545, 0.9972], 0, 5, color='red', linestyle='dashed')

                axs[0,0].set_title('Extracted spectrum')
                axs[0,1].set_title('Errors')
                #axs[0,1].set_title('Diff')
                axs[1,0].set_title('Variances from mean devided by sigma')
                axs[1,1].set_title('Error histogramm')
                axs[0,0].legend()

                axs[0,0].set_xlabel('Pixel')
                axs[0,0].set_ylabel('spectrum in a.u.')
                axs[0,1].set_xlabel('Pixel')
                axs[0,1].set_ylabel('errors in a.u.')
                axs[1,0].set_xlabel('Pixel')
                axs[1,0].set_ylabel('normalized variance')
                axs[1,1].set_xlabel(r'$\sigma$')
                axs[1,1].set_ylabel('density')

                plt.tight_layout()


                if datashare.reduction_parameters.save_plots:
                    filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_weightedextracted_order{}.png".format(i))
                    plt.savefig(filename, dpi=300)

                if datashare.reduction_parameters.show_plots:
                    plt.show()


                plt.close(fig)

        # create Spectrum object, copy image header to spectrum
        spectrum = Spectra.RawSpectrum()
        spectrum.header = Image.header

        # append orders to spectrum
        for i in range(len(orders)):
            # flux is first element, errors is second element
            order_spectrum = orders[i][0]
            order_errors   = orders[i][1]

            #clip negative values in spectrum, assure errors are positive
            #order_spectrum = np.clip(order_spectrum, a_min=1e-10, a_max=np.inf)
            order_errors   = np.abs(order_errors)

            x_range = np.arange(len(order_spectrum))

            order_obj = Spectra.RawSpectralOrder(x_range, order_spectrum, errors=order_errors, ordernr=i)

            spectrum.addOrder(order_obj)

        # add spectrum to list
        all_spectra.append(spectrum)

    # create SpectraList object from list of spectra
    spectralist = Spectra.SpectraList(spectra=all_spectra)

    return spectralist

#extract one trace / order
#this method can be used in multiple threads simultaneously
def _extract_trace_optimal(args):
    global datashare

    image, errors, Weights, trace, fiber_nr, order_nr, filter_cosmics, nsigmas = args

    #pixel range
    x_range = np.arange(image.shape[1]).astype(int)

    order_spectrum = np.zeros(image.shape[1])
    order_errors   = np.zeros(image.shape[1])

    trace.compute_centers(x_range)
    Centers   = trace.Centers


    ordershape = Weights.get_ordershape(fiber_nr, order_nr)

    #iteration over all pixels
    for x in x_range:
        center = Centers[x]

        #get weights
        tilt_weights                               = Weights.get_weights(fiber_nr, order_nr, x)
        min_idx_x, min_idx_y, max_idx_x, max_idx_y = Weights.get_boundaries(fiber_nr, order_nr, x)

        #tilt_weights /= np.clip(np.sum(tilt_weights, axis=1).reshape(-1,1), a_min=1e-10, a_max=np.inf)

        #we want to shift individual pixels so, that the center always is exactly the center of y_range
        mid_y       = center - min_idx_y
        order_shift = mid_y  - ordershape.shape[0]/2.


        #get extraction window
        window        = image[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]
        window_errors = errors[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]

        window_product  = tilt_weights * window

        window_rows = np.nansum(window_product, axis=1)
        #error_rows  = np.sqrt(np.nansum(np.square(tilt_weights * window_errors), axis=1))
        error_rows = np.nansum(tilt_weights * window_errors, axis=1)

        #set errors which are too small to maximal value
        error_rows[error_rows < 0.001 * np.max(error_rows)] = np.max(error_rows)

        ordershape_spline = interpolate.CubicSpline(np.arange(ordershape.shape[0]), ordershape[:, x], extrapolate=False)

        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift

        ordershape_eval = ordershape_spline(ext_y_range)

        if np.all(np.isnan(ordershape_eval)) or np.all(np.logical_or(ordershape_eval < 1e-10, np.isnan(ordershape_eval))):
            print('Bad ordershape at pixel {}, order {}'.format(x, order_nr))

        not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()

        ordershape_eval[not_nan_inds] = np.clip(ordershape_eval[not_nan_inds], a_min=1e-10, a_max=np.inf)


        ordershape_eval /= np.nansum(ordershape_eval)

        #if order_nr == 7 and x > 2000 and x % 100 == 0:
        if False:
            plt.imshow(tilt_weights)
            plt.title('pix {}, order {}'.format(x, order_nr))
            print('')
            plt.show()

            plt.close()

        #get weights as calculated by Horne (1986) / Marsh (1989)
        ordershape_notnan = ordershape_eval.copy()
        ordershape_notnan[np.isnan(ordershape_notnan)] = 1e-10
        ordershape_notnan = np.clip(ordershape_notnan, a_min=1e-10, a_max=np.inf)
        ordershape_notnan /= np.sum(ordershape_notnan)

        ordershape_notnan = _filtersingleordershape(ordershape_notnan)

        V = np.clip(np.square(error_rows), a_min=1e-10, a_max=np.inf)
        V /= np.nansum(V)
        P = ordershape_notnan
        #P = np.ones_like(ordershape_notnan) / len(ordershape_notnan)
        PV = np.sum(P * P / V)

        W = (P / V) / PV

        tilt_weights_copy = tilt_weights.copy()

        #FIXME: hier irgendwo ist das Problem!
        #TODO: Ordershape mit 2D polynom fitten, um es zu glätten!
        tilt_weights = tilt_weights * W[:, np.newaxis]
        tilt_weights *= np.nansum((np.sum(tilt_weights_copy, axis=1) * ordershape_notnan)) / np.nansum((np.sum(tilt_weights, axis=1) * ordershape_notnan))

        #if x == x_range[-1]//2 or x == x_range[-1] * 3 // 4:
        if False:
            #plt.title('Extraction weights')
            #plt.imshow(tilt_weights)
            print('')
            #plt.show()

            plt.plot(P, label='P')
            #plt.show()

            #plt.plot(V)
            #plt.show()

            #plt.title('W')
            plt.plot(W / np.sum(W), label='W')
            #plt.show()

            #plt.title('Flux')
            plt.plot(window_rows / np.sum(window_rows), label='Flux' )

            plt.legend()
            plt.savefig(os.path.join(datashare.reduction_parameters.plot_dir, 'Test_order{}_pix{}.png'.format(order_nr, x)))
            #plt.show()

            #plt.title('Errors')
            #plt.plot(error_rows)
            #plt.show()

            #plt.title('flux / error^2')
            #plt.plot(window_rows / error_rows**2)
            #plt.show()

            plt.close('all')

            #print(np.sum(tilt_weights))

            if np.any((W / np.sum(W)) > 0.5):
                print(W / np.sum(W))
                print(P)
                print(window_rows / np.sum(window_rows))
                print(V)
                print(window_errors)
                print(error_rows)


        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift

        ordershape_eval = ordershape_spline(ext_y_range)

        #use median of upper half of values as scaling parameter
        window_median = np.median(window_rows)
        ordershape_median = np.nanmedian(ordershape_eval)

        window_median = np.median(window_rows[window_rows > window_median])
        ordershape_median = np.nanmedian(ordershape_eval[ordershape_eval> ordershape_median])

        #ordershape_eval *= np.percentile(window_rows, 90) / np.nanpercentile(ordershape_eval, 90)
        ordershape_eval *= window_median / ordershape_median

        #replace bad indicies, e.g. cosmics or hot pixels, with expected value
        nan_inds     = np.asarray(np.isnan(ordershape_eval)).nonzero()[0]
        not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()[0]

        ordershape_eval[not_nan_inds] = np.clip(ordershape_eval[not_nan_inds], a_min=1e-10, a_max=np.inf)

        rms = np.sqrt(np.mean(np.square(window_rows[not_nan_inds]/ordershape_eval[not_nan_inds] -1)))

        bad_inds = np.asarray(np.abs(window_rows[not_nan_inds]/ordershape_eval[not_nan_inds] - 1) > nsigmas * rms).nonzero()[0]
        bad_inds = np.unique(np.concatenate((not_nan_inds[bad_inds], nan_inds)))

        #ensure less than half of all indices are masked
        if len(bad_inds) < len(window_rows) // 2:
            #mask bad indices
            tilt_weights_copy = tilt_weights.copy()
            tilt_weights[bad_inds, :] = 0
            tilt_weights *= np.nansum((np.sum(tilt_weights_copy, axis=1) * ordershape_eval)) / np.nansum((np.sum(tilt_weights, axis=1) * ordershape_eval))


        #if x == len(x_range) //2:
        #    plt.plot(window_rows[not_nan_inds])
        #    plt.plot(ordershape_eval[not_nan_inds], color='red')
        #    print('')
        #    plt.show()


        #while np.count_nonzero(tilt_weights > 0) > 5:
        #    tilt_weights_temp = tilt_weights.copy()
        #    tilt_weights_temp[tilt_weights_temp == 0] = np.inf

        #    tilt_weights[np.unravel_index(np.argmin(tilt_weights_temp), tilt_weights.shape)] = 0


        #tilt_weights /= np.sum(tilt_weights)

        #window_rows[bad_inds] = ordershape_eval[bad_inds]

        #print(order_nr, x,bad_inds, len(window_rows))

        #if x == len(x_range) // 2:
        #    print(bad_inds, rms)

        #TODO:How to deal with errors!?

        #sum up values
        try:
            #order_spectrum[x] = np.clip(np.nansum(tilt_weights * window), a_min=1e-10, a_max=np.inf)
            order_spectrum[x] = np.nansum(tilt_weights * window)
            order_errors_rows = np.nansum(tilt_weights * window_errors, axis=1)
            order_errors[x]   = np.sqrt(np.nansum(np.square(order_errors_rows)))


            #order_errors[x]    = np.sqrt(np.nansum(tilt_weights * np.square(window_errors)))
            #order_errors[x]   = np.sqrt(np.nansum(np.square(tilt_weights * window_errors)))

            #if np.sum(tilt_weights) < 1:
            #    print(tilt_weights)

            #f = 1. / ordershape_eval[not_nan_inds]
            #w = np.square(f / error_rows[not_nan_inds])
            #a = (w / f) / np.sum(w)

            #window_rows = np.clip(np.nansum(tilt_weights * window, axis=1), a_min=1e-10, a_max=np.inf)
            #error_rows  = np.clip(np.nansum(tilt_weights * window_errors, axis=1), a_min=1e-10, a_max=np.inf)

            #order_spectrum[x] = np.sum(a * f * window_rows[not_nan_inds])

            #order_errors[x] = np.sqrt(np.sum(np.sqrt(a * f * error_rows[not_nan_inds])))


        except Exception as e:
            print(e)
            order_spectrum[x] = np.nan
            order_errors[x]   = np.inf

        """
        if x >= 1500 and x <= 1600 and x%5 == 0:
            print(x)
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(window)
            axs[1].imshow(final_weights)
            plt.show()
        """

    # filter cosmics in spectrum, additionally to filter in raw image
    if filter_cosmics:
        try:
            order_spectrum = filter_cosmics_spec(order_spectrum)
        except:
            pass


    #filter too low values
    max_err = np.max(order_errors)
    order_errors[order_errors < 0.001 * max_err] = max_err

    # return flux and errors
    return order_spectrum, order_errors


#method to create the weights. This method is called once per order and can be used in multiple threads simultaneously
def _trace_weights(args):
    i, image, errors, Fiber_traces, default_tilt = args

    # get trace
    trace = Fiber_traces.get_trace(i)

    # get previous and next trace to estimate cross dispersion distance between orders
    if i > 0:
        prev_trace = Fiber_traces.get_trace(i -1)
    else:
        prev_trace = None

    if i < len(Fiber_traces) -1:
        next_trace = Fiber_traces.get_trace(i+1)
    else:
        next_trace = None

    order_sigma = trace.sigma

    #2 pixels is standard with of orders
    if order_sigma <= 0:
        order_sigma = 2

    all_weights = []
    all_boundaries = []

    #number of dots for line tracing
    ndots = 500

    #range of pixels
    x_range = np.arange(image.shape[1]).astype(int)

    #calculate centers (y-coordinates) of trace. These can be floats (between two pixels)
    trace.compute_centers(x_range)
    Centers   = trace.Centers

    #compute centers for previous and next trace
    if prev_trace is not None:
        prev_trace.compute_centers(x_range)
        prev_centers = prev_trace.Centers
    else:
        prev_centers = None

    if next_trace is not None:
        next_trace.compute_centers(x_range)
        next_centers = next_trace.Centers
    else:
        next_centers = None

    #get tilt. Prefer default_tilt, else use tilt in Trace_obj
    if default_tilt is None:
        order_tilt = trace.tilt
    else:
        order_tilt = default_tilt

    if (not isinstance(order_tilt, numbers.Number)):
        if isinstance(order_tilt, np.ndarray) and len(order_tilt) == 1:
            # tilt is constant, convert array to scalar
            order_tilt = order_tilt[0]
        elif isinstance(order_tilt, np.ndarray) and len(order_tilt) == image.shape[1]:
            pass
        elif isinstance(order_tilt, np.ndarray) and len(order_tilt) != image.shape[1]:
            raise ValueError('length of order_tilt ({}) does not match length of pixels ({})'.format(len(order_tilt), image.shape[1]))
        else:
            raise ValueError('order_tilt must be a scalar or a array of tilts for each pixel!')

    #create weights for all pixels
    for x, Center in zip(x_range, Centers):
        #check if order_tilt is constant
        if isinstance(order_tilt, numbers.Number):
            tilt = order_tilt
        else:
            tilt = order_tilt[int(x)]

        #cut out image to relevant window
        if tilt == 0:   #this corresponds to a window with width of 1 -> weights will be within one column
            window_width_x = 0
            window_width_y = np.ceil(3* order_sigma).astype(int)
        else: #line is tilted, we need a larger window so that the line fits in it
            window_width_x = np.ceil(3* order_sigma * np.abs(np.sin(tilt))).astype(int)
            window_width_y = np.ceil(3* order_sigma * np.abs(np.cos(tilt))).astype(int)

        #get distance to previous / next order. Window should reach maximal to middle between orders
        if prev_centers is not None:
            prev_dist = np.ceil(np.abs(Center - prev_centers[x])) // 2
        else:
            prev_dist = np.inf

        if next_centers is not None:
            next_dist = np.ceil(np.abs(Center - next_centers[x])) // 2
        else:
            next_dist = np.inf

        window_width_y = np.min((window_width_y, prev_dist, next_dist)).astype(int)

        min_idx_x = int(np.max((0, x - window_width_x)))
        min_idx_y = int(np.max((0, Center - window_width_y)))
        max_idx_x = int(np.min((image.shape[1] - 1, x + window_width_x)))
        max_idx_y = int(np.min((image.shape[0] - 1, Center + window_width_y)))

        window        = image[min_idx_y:max_idx_y +1, min_idx_x:max_idx_x +1]
        window_errors = errors[min_idx_y:max_idx_y +1, min_idx_x:max_idx_x +1]


        # left offsets, in case we are too close to the boundaries and the full window would be located outside of the image. Will be 0 in most cases
        x_offset = np.min((x - window_width_x, 0))
        y_offset = np.min((Center - window_width_y, 0))


        #if tilt is zero, we can make things easier:
        if tilt == 0:
            #equal weights
            line_weights_pix = np.ones_like(window)

            #apply SNR weights
            SNR_window = window / window_errors

            line_weights_pix = line_weights_pix * np.square(SNR_window)
            line_weights_pix /= np.sum(line_weights_pix)

        else:
            #calculate line weights, which accounts for the tilted lines
            #we assume, that the spectral PSF is a tilted straight line. We trace this line over the order. The weights are set corresponding to the amount the line lies within this pixel

            line_weights_pix = np.zeros_like(window)

            #x and y coordinates of the line
            mid_x = window_width_x + x_offset
            mid_y = (Center - min_idx_y + y_offset) #set middle y to center of trace (Center % 1 returnes decimal places of Center)
            #fac_x = np.sin(tilt) * window_width_x/ndots    #slope in x-direction
            fac_x = 2 * np.sin(tilt) * window_width_x/ndots    #slope in x-direction. Times 2 because we have dots = [-dots/2, dots/2] instead of [0, dots]
            fac_y = np.cos(tilt) * window_width_y/ndots    #slope in y-direction

            #parametrize x- and y-coordinates with a straight line
            line_x = lambda t: mid_x + t * fac_x
            line_y = lambda t: mid_y + t * fac_y

            #line parameters
            dots = (3 * mid_y / (np.cos(tilt) * window_width_y)) * (np.arange(ndots) - ndots/2. + 0.5)      #plus 0.5 so that we have +/- 0.5 for even ndots. Factor 2 in front would be enough, we use 3 just to be sure that boundaries are not undersampled

            #x- and y-coordinates of tilted line. No rounding, this is compensated by the +/- 0.5 above
            xs, ys = np.round(line_x(dots)).astype(int), np.round(line_y(dots)).astype(int)

            #cut to valid indices
            valid_inds = np.where(_valid_index(xs, ys, line_weights_pix.shape[1] - 1, line_weights_pix.shape[0] - 1))[0]

            xs = xs[valid_inds]
            ys = ys[valid_inds]

            #set weights to one for each pixel the line crosses
            for (x_l,y_l) in zip(xs,ys):
                    line_weights_pix[y_l,x_l] += 1


            #norm each row individually
            for y in range(line_weights_pix.shape[0]):
                nansum = np.nansum(line_weights_pix[y,:])

                if nansum > 0:
                    line_weights_pix[y,:] /= nansum    #devide by number of non-zero entries, normalization
                else:
                    line_weights_pix[y,:] = 0

            #if all entries are nan set all weights of middle line to one. If single entries are nan set them to zero
            if np.isnan(line_weights_pix).all():
                line_weights_pix = np.zeros_like(window)
                line_weights_pix[:, window_width_x + x_offset] = 1

            elif np.isnan(line_weights_pix).any():
                line_weights_pix[np.isnan(line_weights_pix)] = 0

                #norm each row individually
                for y in range(line_weights_pix.shape[0]):
                    nansum = np.nansum(line_weights_pix[y,:])

                    if nansum > 0:
                        line_weights_pix[y,:] /= nansum    #devide by number of non-zero entries, normalization
                    else:
                        line_weights_pix[y,:] = 0




            #apply SNR weights
            SNR_window = window / window_errors

            line_weights_pix = line_weights_pix * np.square(SNR_window)
            line_weights_pix /= np.sum(line_weights_pix)

        line_weights_pix = np.clip(line_weights_pix, a_min=0, a_max=1)

        #norm weights again
        line_weights_pix /= np.sum(line_weights_pix)

        #save boundaries of weights window
        boundaries = (min_idx_x, min_idx_y, max_idx_x, max_idx_y)

        all_weights.append(line_weights_pix)
        all_boundaries.append(boundaries)

    #print('finished weights!')

    return (all_weights, all_boundaries)


def create_weights_noSNR(Image, Trace_obj, npools=None, default_tilt=None):
    """
    # Create the extraction mask from a flat image, taking tilted lines into account. If case of no tilt this is just a 1D array of ones.
    # Only tilt mask, no cosmic filtering etc here, as they can differ from image to image
    # This mask will be used for all images from that night
    # Used multithreading to work on multiple orders simultaneously
    #
    # Weights are defined via a 2D array of weights for each pixel as well as a set of pixel coordinates to which pixels in the image this window belongs
    #
    # :param Image: Image object, flat image to extract weights from
    # :param Trace_obj: Fiber_traces object, contains the information about the position of the spectral orders
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param default_tilt: float or numpy.ndarray of floats or None. if None, will use the tilt contained in Trace_obj. If float, will assume tilt is constant over the order. When ndarray, length has to be the same as pixel length, will use default_tilt[pixel] as tilt, which can change over the order
    #
    # :returns Weights: ExtractionWeights object, contains the weights for weighted extracion
    """

    #get data from image
    image  = Image.data.copy()
    errors = Image.errors.copy()
    gain   = Image.gain
    RON    = Image.RON

    # get number of threads from config
    if npools is None:
        npools = datashare.reduction_parameters.npools

    #get number of fibers, orders and pixels
    nr_of_fibers = Trace_obj.nr_of_fibers()
    nr_of_orders = np.max([len(trace) for trace in Trace_obj.traces])
    nr_of_pixels = image.shape[1]

    #initialize weights
    Weights = Spectra.ExtractionWeights(nr_of_fibers, nr_of_orders, nr_of_pixels)

    #if default_tilt is a list (for each fiber), but dimensions do not match, just ignore it /set it to None
    if type(default_tilt) is list or isinstance(default_tilt, np.ndarray):
        if len(default_tilt) != nr_of_fibers:
            default_tilt = [None] * nr_of_fibers

    #each fiber will have the same tilt
    else:
        default_tilt = [default_tilt] * nr_of_fibers

    #go through all fibers
    for fiber_nr in range(nr_of_fibers):
        Fiber_trace = Trace_obj.traces[fiber_nr]

        #use multiprocessing to speed things up, go through orders
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            order_dists = [np.floor(np.min(Trace_obj.getMinDistances(fiber_nr, trace_nr, np.arange(image.shape[1]), image.shape[0]))/2.).astype(int) for trace_nr in range(len(Fiber_trace.traces))]

            args = [(i, image.copy(), errors, Fiber_trace, default_tilt[i%nr_of_fibers], order_dists[i]) for i in range(len(Fiber_trace))]

            result = pool.map(_trace_weights_noSNR, args)

            #set weights for the single orders
            for order_nr in range(Fiber_trace.nr_of_orders):
                for x in range(nr_of_pixels):
                    Weights.set_weights(fiber_nr, order_nr, x, result[order_nr][0][x])
                    Weights.set_boundaries(fiber_nr, order_nr, x, result[order_nr][1][x])

            #plot weights for one specific pixel (roughly middle of spectrum) to briefly check weights calculation
            if datashare.reduction_parameters.plot_Weights:
                mid_order = len(result) // 2
                mid_pix   = len(result[mid_order][0]) // 2

                plt.imshow(result[mid_order][0][mid_pix])

                if datashare.reduction_parameters.save_plots:
                    filename = os.path.join(datashare.reduction_parameters.plot_dir, "Weights.png")
                    plt.savefig(filename)

                if datashare.reduction_parameters.show_plots:
                    print('')
                    plt.show()

                plt.close()

    Weights = CreateOrderShapes(Image, Trace_obj, Weights, npools=npools)

    #return final weights object
    return Weights


#method to create the weights. This method is called once per order and can be used in multiple threads simultaneously
def _trace_weights_noSNR(args):
    i, image, errors, Fiber_traces, default_tilt, orderdist = args

    # get trace
    trace = Fiber_traces.get_trace(i)

    # get previous and next trace to estimate cross dispersion distance between orders
    if i > 0:
        prev_trace = Fiber_traces.get_trace(i -1)
    else:
        prev_trace = None

    if i < len(Fiber_traces) -1:
        next_trace = Fiber_traces.get_trace(i+1)
    else:
        next_trace = None

    order_sigma = trace.sigma

    #2 pixels is standard with of orders
    if order_sigma <= 0:
        order_sigma = 2

    all_weights = []
    all_boundaries = []

    #number of dots for line tracing
    ndots = 500


    #range of pixels
    x_range = np.arange(image.shape[1]).astype(int)

    #compute centers
    trace.compute_centers(x_range)
    Centers = trace.Centers


    #get tilt. Prefer default_tilt, else use tilt in Trace_obj
    if default_tilt is None:
        order_tilt = trace.tilt
    else:
        order_tilt = default_tilt

    if (not isinstance(order_tilt, numbers.Number)):
        if isinstance(order_tilt, np.ndarray) and len(order_tilt) == 1:
            # tilt is constant, convert array to scalar
            order_tilt = order_tilt[0]
        elif isinstance(order_tilt, np.ndarray) and len(order_tilt) == image.shape[1]:
            pass
        elif isinstance(order_tilt, np.ndarray) and len(order_tilt) != image.shape[1]:
            raise ValueError('length of order_tilt ({}) does not match length of pixels ({})'.format(len(order_tilt), image.shape[1]))
        else:
            raise ValueError('order_tilt must be a scalar or a array of tilts for each pixel!')

    #create weights for all pixels
    for x, Center in zip(x_range, Centers):
        #check if order_tilt is constant
        if isinstance(order_tilt, numbers.Number):
            tilt = order_tilt
        else:
            tilt = order_tilt[int(x)]

        #cut out image to relevant window
        if tilt == 0:   #this corresponds to a window with width of 1 -> weights will be within one column
            window_width_x = 0
            window_width_y = np.ceil(3* order_sigma).astype(int)
        else: #line is tilted, we need a larger window so that the line fits in it
            window_width_x = np.ceil(3* order_sigma * np.abs(np.sin(tilt))).astype(int)
            window_width_y = np.ceil(3* order_sigma * np.abs(np.cos(tilt))).astype(int)

        #compare window_width_y to distance to previous / next order. Window should reach maximal to middle between orders
        window_width_y = np.min((window_width_y, orderdist)).astype(int)

        min_idx_x = int(np.max((0, x - window_width_x)))
        min_idx_y = int(np.max((0, Center - window_width_y)))
        max_idx_x = int(np.min((image.shape[1] - 1, x + window_width_x)))
        max_idx_y = int(np.min((image.shape[0] - 1, Center + window_width_y)))

        window        = image[min_idx_y:max_idx_y +1, min_idx_x:max_idx_x +1]
        window_errors = errors[min_idx_y:max_idx_y +1, min_idx_x:max_idx_x +1]


        # left offsets, in case we are too close to the boundaries and the full window would be located outside of the image. Will be 0 in most cases
        x_offset = np.min((x - window_width_x, 0))
        y_offset = np.min((int(Center - window_width_y), 0))


        #if tilt is zero, we can make things easier:
        if tilt == 0:
            #equal weights
            line_weights_pix = np.ones_like(window)

            #apply SNR weights
            #SNR_window = window / window_errors

            #line_weights_pix = line_weights_pix * np.square(SNR_window)

        else:
            #calculate line weights, which accounts for the tilted lines
            #we assume, that the spectral PSF is a tilted straight line. We trace this line over the order. The weights are set corresponding to the amount the line lies within this pixel

            line_weights_pix = np.zeros_like(window)

            #x and y coordinates of the line
            mid_x = window_width_x + x_offset
            mid_y = (Center % 1 + window_width_y + y_offset) #set middle y to center of trace (Center % 1 returnes decimal places of Center)
            #fac_x = np.sin(tilt) * window_width_x/ndots    #slope in x-direction
            fac_x = 2 * np.sin(tilt) * window_width_x/ndots    #slope in x-direction. Times 2 because we have dots = [-dots/2, dots/2] instead of [0, dots]
            fac_y = np.cos(tilt) * window_width_y/ndots    #slope in y-direction

            #parametrize x- and y-coordinates with a straight line
            line_x = lambda t: mid_x + t * fac_x
            line_y = lambda t: mid_y + t * fac_y

            #line parameters
            dots = (3 * mid_y / (np.cos(tilt) * window_width_y)) * (np.arange(ndots) - ndots/2. + 0.5)      #plus 0.5 so that we have +/- 0.5 for even ndots. Factor 2 in front would be enough, we use 3 just to be sure that boundaries are not undersampled

            #x- and y-coordinates of tilted line. No rounding, this is compensated by the +/- 0.5 above
            xs, ys = np.round(line_x(dots)).astype(int), np.round(line_y(dots)).astype(int)

            #cut to valid indices
            valid_inds = np.where(_valid_index(xs, ys, line_weights_pix.shape[1] - 1, line_weights_pix.shape[0] - 1))[0]

            xs = xs[valid_inds]
            ys = ys[valid_inds]

            #set weights to one for each pixel the line crosses
            for (x_l,y_l) in zip(xs,ys):
                    line_weights_pix[y_l,x_l] += 1


            #norm each row individually
            for y in range(line_weights_pix.shape[0]):
                nansum = np.nansum(line_weights_pix[y,:])

                if nansum > 0:
                    line_weights_pix[y,:] /= nansum    #devide by number of non-zero entries, normalization
                else:
                    line_weights_pix[y,:] = 0

            #if all entries are nan set all weights of middle line to one. If single entries are nan set them to zero
            if np.isnan(line_weights_pix).all():
                line_weights_pix = np.zeros_like(window)
                line_weights_pix[:, window_width_x + x_offset] = 1

            elif np.isnan(line_weights_pix).any():
                line_weights_pix[np.isnan(line_weights_pix)] = 0

                #norm each row individually
                for y in range(line_weights_pix.shape[0]):
                    nansum = np.nansum(line_weights_pix[y,:])

                    if nansum > 0:
                        line_weights_pix[y,:] /= nansum    #devide by number of non-zero entries, normalization
                    else:
                        line_weights_pix[y,:] = 0


            #apply SNR weights
            #SNR_window = window / window_errors

            #line_weights_pix = line_weights_pix * np.square(SNR_window)

        line_weights_pix = np.clip(line_weights_pix, a_min=0, a_max=1)

        #norm weights again
        #line_weights_pix /= np.sum(line_weights_pix)

        #save boundaries of weights window
        boundaries = (min_idx_x, min_idx_y, max_idx_x, max_idx_y)

        all_weights.append(line_weights_pix)
        all_boundaries.append(boundaries)

    #print('finished weights!')

    return (all_weights, all_boundaries)


#TODO, FIXME: Immernoch Oszillationen in den Fehlern, WARUM? Ich sehe keine Osillationen mehr im Flux

#IMPORTANT: This devides the current ordershape by the master ordershape and uses the weighted average. Still unclear whether this is the best extraction method.
#Weights should not weighted by SNR in this case!
def weighted_extract_average(Image, Trace_obj, Weights, npools=None, filter_cosmics=False, nsigmas=3):
    """
    # Extract the image along the given traces using weights.
    # Use ExtractionWeights for this to weight pixels relative to SNR and to account for tilted lines
    # Use multithreading to work on multiple orders simultaneously
    #
    # :param Image: Image object, contains the image to ectraxt
    # :param Trace_obj: Fiber_traces object, contains the informations about the order positions and PSF/ line tilt
    # :param Weights: ExtractionWeights object, contains the extraction weights for each pixel in each order
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param filter_cosmics: boolean, whether to filter cosmics, still experimental (default False)
    # :param nsigmas: float, multiple of errors. Deviations larger than nsigmas * errors from the expected ordershape will be filtered
    #
    # :return spectralist: SpectraList object, contains a RawSpectrum object for each fiber
    """

    # get informations from image
    image  = Image.data
    errors = Image.errors
    gain   = Image.gain
    RON    = Image.RON

    # get number of threads from config
    if npools is None:
        npools = datashare.reduction_parameters.npools_extract


    # calculate errors, if those were not calculated before
    if errors is None:
        errors = np.sqrt(np.abs(data * gain) + np.power(RON, 2.)) / gain

    all_spectra = []

    # iterate over all traces
    for fiber_nr in range(Trace_obj.nr_of_fibers()):
        Fiber_trace = Trace_obj.traces[fiber_nr]

        # here the extraction happens, use multiple threads
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            try:
                args = [(image, errors, Weights, trace, fiber_nr, trace_nr, filter_cosmics, nsigmas) for trace_nr, trace in enumerate(Fiber_trace.all_traces())]

                orders = pool.map(_extract_trace_average, args)
            finally:
                pool.close()
                pool.join()

        # plot, if requested
        if datashare.reduction_parameters.plot_Extraction:
            filter_width = 5

            for i in range(len(orders)):
                order_spectrum = orders[i][0]
                order_errors   = orders[i][1]
                x_range = np.arange(len(order_spectrum))

                trace = Fiber_trace.get_trace(i)
                trace.compute_centers(x_range)
                Mid_center = np.max(trace.Centers)

                #moving average
                mean_spec = np.convolve(order_spectrum, np.ones(filter_width)/filter_width, mode='valid')
                mean_errs = np.convolve(order_errors, np.ones(filter_width)/filter_width, mode='valid')

                fig, axs = plt.subplots(2,2, figsize=(16,8))
                fig.suptitle('Order {}, Mid Center {}'.format(i+1, int(Mid_center)))

                axs[0,0].plot(x_range, order_spectrum, color='black', label='Extracted spectrum')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec + 2 * mean_errs, color='red')
                axs[0,0].plot(x_range[filter_width//2:-filter_width//2 + 1], mean_spec - 2 * mean_errs, color='red', label=r'2 $\sigma$ error range')
                axs[0,1].plot(x_range, order_errors)
                #axs[0,1].plot(x_range[:-2], np.diff(order_spectrum, n=2), color='black', label='Extracted spectrum')

                axs[1,0].plot(x_range[filter_width//2:-filter_width//2 + 1], np.abs(order_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs)
                axs[1,0].hlines([1,2,3], filter_width//2, len(order_spectrum) - filter_width//2, color='red', linestyle='dashed')
                axs[1,1].hist(np.abs(order_spectrum[filter_width//2:-filter_width//2 + 1] - mean_spec)/mean_errs, bins=np.linspace(0,5,10), cumulative=True, density=True)
                axs[1,1].hlines([0.6827, 0.9545, 0.9972], 0, 5, color='red', linestyle='dashed')

                axs[0,0].set_title('Extracted spectrum')
                axs[0,1].set_title('Errors')
                #axs[0,1].set_title('Diff')
                axs[1,0].set_title('Variances from mean devided by sigma')
                axs[1,1].set_title('Error histogramm')
                axs[0,0].legend()

                axs[0,0].set_xlabel('Pixel')
                axs[0,0].set_ylabel('spectrum in a.u.')
                axs[0,1].set_xlabel('Pixel')
                axs[0,1].set_ylabel('errors in a.u.')
                axs[1,0].set_xlabel('Pixel')
                axs[1,0].set_ylabel('normalized variance')
                axs[1,1].set_xlabel(r'$\sigma$')
                axs[1,1].set_ylabel('density')

                plt.tight_layout()

                if datashare.reduction_parameters.save_plots:
                    filename = os.path.join(datashare.reduction_parameters.plot_dir, datashare.current_filename.replace('.fits' ,'') + "_weightedextracted_order{}.png".format(order))
                    plt.savefig(filename, dpi=300)

                if datashare.reduction_parameters.show_plots:
                    plt.show()


                plt.close(fig)

        # create Spectrum object, copy image header to spectrum
        spectrum = Spectra.RawSpectrum()
        spectrum.header = Image.header

        # append orders to spectrum
        for i in range(len(orders)):
            # flux is first element, errors is second element
            order_spectrum = orders[i][0]
            order_errors   = orders[i][1]

            #clip negative values in spectrum, assure errors are positive
            order_spectrum = np.clip(order_spectrum, a_min=1e-10, a_max=np.inf)
            order_errors   = np.abs(order_errors)

            x_range = np.arange(len(order_spectrum))

            order_obj = Spectra.RawSpectralOrder(x_range, order_spectrum, errors=order_errors, ordernr=i)

            spectrum.addOrder(order_obj)

        # add spectrum to list
        all_spectra.append(spectrum)

    # create SpectraList object from list of spectra
    spectralist = Spectra.SpectraList(spectra=all_spectra)

    return spectralist

#extract one trace / order
#this method can be used in multiple threads simultaneously
def _extract_trace_average(args):
    image, errors, Weights, trace, fiber_nr, order_nr, filter_cosmics, nsigmas = args

    #pixel range
    x_range = np.arange(image.shape[1]).astype(int)

    order_spectrum = np.zeros(image.shape[1])
    order_errors   = np.zeros(image.shape[1])

    trace.compute_centers(x_range)
    Centers   = trace.Centers


    ordershape = Weights.get_ordershape(fiber_nr, order_nr)


    def _fitamplshift(x, window_rows, ordershape_spline, ext_y_range):
        ext_y_range = ext_y_range.copy() - x[1]

        ordershape_eval = ordershape_spline(ext_y_range) * x[0]

        ordershape_eval[np.isnan(ordershape_eval)] = 0

        return np.sum(np.square(window_rows - ordershape_eval))

    #iteration over all pixels
    for x in x_range:
        center = Centers[x]

        #get weights
        tilt_weights                               = Weights.get_weights(fiber_nr, order_nr, x)
        min_idx_x, min_idx_y, max_idx_x, max_idx_y = Weights.get_boundaries(fiber_nr, order_nr, x)

        #we want to shift individual pixels so, that the center always is exactly the center of y_range
        mid_y       = center - min_idx_y
        order_shift = mid_y - ordershape.shape[0]/2.+ 0.1

        ordershape_spline = interpolate.CubicSpline(np.arange(ordershape.shape[0]), ordershape[:, x], extrapolate=False)

        #get extraction window
        window        = image[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]
        window_errors = errors[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]

        window_product  = tilt_weights * window

        window_rows = np.clip(np.nansum(window_product, axis=1), a_min=1e-10, a_max=np.inf)
        error_rows  = np.sqrt(np.nansum(np.square(tilt_weights * window_errors), axis=1))

        #set unrealistic low errors to maximal value
        error_rows[error_rows < 0.1] = np.max(error_rows)

        #get estimates for shift and amplitude
        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift
        ordershape_eval = ordershape_spline(ext_y_range)
        #ordershape_eval =_ordershape_int(ext_y_range)

        not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()
        nan_inds     = np.asarray(np.isnan(ordershape_eval)).nonzero()

        ordershape_eval[nan_inds] = 1e-10
        ordershape_eval= np.clip(ordershape_eval, a_min=1e-10, a_max=np.inf)

        amp_guess   = np.median(window_rows) / np.median(ordershape_eval)
        shift_guess = order_shift

        #get shift from minimizing
        try:
            args   = (window_rows, ordershape_spline, np.arange(0, max_idx_y - min_idx_y +1))
            x0     = (amp_guess, shift_guess)
            bounds = ((0, np.inf), (order_shift - 2, order_shift + 2))

            res = optimize.minimize(_fitamplshift, x0, args=args)

            if res.success:
                amp   = res.x[0]
                shift = res.x[1]

            else:
                amp   = amp_guess
                shift = order_shift
        except Exception as e:
            amp   = amp_guess
            shift = order_shift


        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - shift
        ordershape_eval = ordershape_spline(ext_y_range)

        not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()

        #use "normal" order shift, if too many values of ordershape_eval are nan
        if len(not_nan_inds) < 3:
            ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift
            ordershape_eval = ordershape_spline(ext_y_range)
            not_nan_inds = np.asarray(~np.isnan(ordershape_eval)).nonzero()


        #TODO, FIXME: How to normalize ordershape_eval, so that is is independend of shape (no median), shift (no max) and number of pixels (no sum)

        ordershape_eval[not_nan_inds] = np.clip(ordershape_eval[not_nan_inds], a_min=1e-10, a_max=np.inf)
        #ordershape_eval /= np.nanmedian(ordershape_eval)

        #bring all values to approximately the same value
        #eliminate the effect of darkening of single pixels, because of the PSF
        #errors will also get larger. Now errors in the mid are smallest
        corr_rows = window_rows[not_nan_inds] / ordershape_eval[not_nan_inds]
        corr_errs = error_rows[not_nan_inds]  / ordershape_eval[not_nan_inds]

        corr_errs = np.clip(np.abs(corr_errs), a_min=1e-10, a_max=np.inf)

        try:
            median_row = np.average(corr_rows, weights=1./np.square(corr_errs))
        except:
            median_row = np.median(corr_rows)

        #only use good indicies, not e.g. cosmics or hot pixels
        #rms       = np.sqrt(np.average(np.square(corr_rows - median_row)))
        good_inds = np.asarray(np.abs(corr_rows - median_row) < nsigmas * corr_rows).nonzero()

        if len(good_inds) < len(corr_errs) // 2:
            good_inds = np.arange(len(corr_errs))

        if x == len(x_range)//2:
        #if False:
            plt.errorbar(np.arange(len(corr_rows)), corr_rows, yerr=corr_errs, fmt='o', label='corrected measurements')
            plt.hlines(median_row, 0, len(corr_rows), linestyle='dashed', color='red', label='weighted average')
            print('')
            plt.legend()
            plt.show()

            plt.plot(window_rows, label='Ordershape')
            plt.plot(ordershape_eval * np.max(window_rows) / np.nanmax(ordershape_eval), color='red', label='Median ordershape')
            plt.legend()
            plt.show()

            plt.close('all')

            #print(good_inds)


        #build weighted average values
        try:
            corr_errs_square  = np.clip(np.square(corr_errs[good_inds]), a_min=1e-10, a_max=np.inf)
            weights           = 1./corr_errs_square
            weights           /= np.sum(weights)

            order_spectrum[x] = np.average(corr_rows[good_inds], weights=weights)
            order_errors[x]   = 1./np.sqrt(np.nansum(1./corr_errs_square))

            #if order_errors[x] < 0.2:
            #    print(error_rows)
            #    print(corr_errs)
            #    print(corr_errs_square)
            #    print('a')
        except:
            order_spectrum[x] = np.nan
            order_errors[x]   = np.inf


        """
        if x >= 1500 and x <= 1600 and x%5 == 0:
            print(x)
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(window)
            axs[1].imshow(final_weights)
            plt.show()
        """

    # filter cosmics in spectrum, additionally to filter in raw image
    if filter_cosmics:
        try:
            order_spectrum = filter_cosmics_spec(order_spectrum)
        except:
            pass

    # return flux and errors
    return order_spectrum, order_errors


def CreateOrderShapes(Masterflat, Trace_data, Weights, npools=None, median_width=250, minSNR=100):
    """
    # Create template order shapes from masterflat (or masterorderdef, if available) frames. These templates will be used to compare other spectral order shapes pixelwise to this template.
    # Any large deviation to the template is most likely a defect (bad pixel, cosmic, etc.) and will be masked during extraction
    #
    # :param Masterflat: Image object, contains the masterflat or masterorderdef image
    # :param Trace_data: Trace_data object, contains the informations about the order positions and PSF/ line tilt
    # :param Weights: ExtractionWeights object, will be updated with ordershapes
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param median_width; int, width of median filter (default 100).
    # :param Weights: ExtractionWeights object, contains the extraction weights for each pixel in each order
    # :param minSNR: float, minimum SNR, which one pixel needs to have, so that the spectrum is used for ordershape calculation (default 100)
    #
    # :return Weights: ExtractionWeights object, updated weights with ordershapes
    """

    # get informations from image
    image  = Masterflat.data
    errors = Masterflat.errors

    # get number of threads from config
    if npools is None:
        npools = datashare.reduction_parameters.npools

    all_spectra = []

    # iterate over all traces
    for fiber_nr in range(Trace_data.nr_of_fibers()):
        Fiber_trace = Trace_data.traces[fiber_nr]

        # here the extraction happens, use multiple threads
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            try:
                order_dists = [np.floor(np.min(Trace_data.getMinDistances(fiber_nr, trace_nr, np.arange(image.shape[1]), image.shape[0]))/2.).astype(int) for trace_nr in range(len(Fiber_trace.traces))]

                args = [(image, errors, Weights, trace, fiber_nr, trace_nr, median_width, minSNR, order_dists[trace_nr]) for trace_nr, trace in enumerate(Fiber_trace.all_traces())]

                orders = pool.map(_getordershape, args)
            finally:
                pool.close()
                pool.join()


        for ordernr in range(len(orders)):
            Weights.set_ordershape(fiber_nr, ordernr, orders[ordernr])


    return Weights


def _getordershape(args):
    image, errors, Weights, trace, fiber_nr, order_nr, median_width, minSNR, order_dist = args

    #pixel range
    x_range = np.arange(image.shape[1]).astype(int)

    y_len = np.minimum(2 * order_dist + 1, np.round(6 * trace.sigma).astype(int))       #3 sigma in both directions

    #assure y_len is odd
    if y_len % 2 == 0:
        y_len += 1

    y_range = np.arange(y_len)

    trace.compute_centers(x_range)
    Centers   = trace.Centers

    ordershape_matrix      = np.zeros(shape=(y_len, len(x_range)))
    ordershape_matrix_errs = np.zeros(shape=(y_len, len(x_range)))

    enough_SNR = np.zeros(len(x_range), bool)
    weights    = np.ones(len(x_range))

    #iteration over all pixels
    for x in x_range:
        #get weights
        pix_weights = Weights.get_weights(fiber_nr, order_nr, x)
        min_idx_x, min_idx_y, max_idx_x, max_idx_y = Weights.get_boundaries(fiber_nr, order_nr, x)

        center = Centers[x]

        #get image window
        window        = image[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]
        window_errors = errors[min_idx_y:max_idx_y+1, min_idx_x:max_idx_x+1]

        weights[x] = np.abs(np.sum(window) / np.maximum(np.sum(window_errors), 1e-10))

        #we want to shift individual pixels sp, that the center always is exactly the center of y_range
        mid_y = center - min_idx_y

        order_shift = mid_y - y_len/2.

        ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift

        ext_ordershape = np.sum(pix_weights * window, axis=1)  #extract masterflat, but only sum along rows to get cross dispersion shape. np.sum(pix_weights, axis=1) should be roughly constant (despite minor computational errors)
        ext_ordershape_errs = np.sum(pix_weights * window_errors, axis=1)


        #subtract background
        ext_ordershape -= (np.min(ext_ordershape) - 1e-10)

        ext_ordershape[np.isnan(ext_ordershape)] = 0

        #interpolate ordershape. Do not extrapolate, but use NaNs instead
        ordershape_spline      = interpolate.CubicSpline(ext_y_range, ext_ordershape, extrapolate=False)
        ordershape_errs_spline = interpolate.CubicSpline(ext_y_range, ext_ordershape_errs, extrapolate=False)

        #evalate order at centered y range
        ordershape_matrix[:, x]      = ordershape_spline(y_range)
        ordershape_matrix_errs[:, x] = ordershape_errs_spline(y_range)

        nansum = np.nansum(ordershape_matrix[:, x])
        ordershape_matrix[:, x]      /= nansum
        ordershape_matrix_errs[:, x] /= nansum

        enough_SNR[x] = np.any(window / window_errors > minSNR)


    ordershape_matrix[np.isnan(ordershape_matrix)] = 1e-10
    ordershape_matrix = np.clip(ordershape_matrix, a_min=1e-10, a_max=1)

    #ordershapes_medianfiltered = CCD_corrections._fitOrdershape2D(ordershape_matrix, weights=weights)

    ordershapes_medianfiltered = ndimage.median_filter(ordershape_matrix, size=median_width, mode='mirror', axes=1)

    """
    good_inds = np.asarray(enough_SNR).nonzero()[0]
    bad_inds  = np.asarray(~enough_SNR).nonzero()[0]

    if len(good_inds) < x_range[-1] //4:
        good_inds = np.arange(x_range[-1])
        bad_inds  = []

    for j in bad_inds:
        #get closest good index
        ind = good_inds[np.argmin(np.abs(good_inds - j))]

        ordershapes_medianfiltered[:, j] = ordershapes_medianfiltered[:, ind]
    """

    """
    ordershapes_medianfiltered = np.zeros(shape=(y_len, len(x_range)))

    for x in range(ordershape_matrix.shape[1]):
        min_ind = np.max((0, x - median_width))
        max_ind = np.min((ordershape_matrix.shape[1], x + median_width + 1))

        elements = ordershape_matrix[:, min_ind:max_ind]
        errs     = ordershape_matrix_errs[:, min_ind:max_ind]
        ordershapes_medianfiltered[:, x] = np.average(elements, weights=np.square(elements/errs), axis=1)
    """

    #filter y values where many values are nan
    #set median value to nan in that case
    for y in range(ordershape_matrix.shape[0]):
        nan_count = np.count_nonzero(np.isnan(ordershape_matrix[y,:]))

        if nan_count > 0.25 * ordershape_matrix.shape[1]:
            ordershapes_medianfiltered[y, :] = np.nan



    y_range_notnan = y_range

    #ensure that mid of ordershape is always mid of array
    while np.any(np.isnan(ordershapes_medianfiltered)):
        ordershapes_medianfiltered = ordershapes_medianfiltered[1:-1, :]
        y_range_notnan             = y_range_notnan[1:-1]


    #filter ordershape and mask outliners at the edges
    ordershapes_medianfiltered = _filterordershapematrix(ordershapes_medianfiltered)


    if datashare.reduction_parameters.plot_Ordershapes:
        fig, axs = plt.subplots()

        for x in x_range[-500:-1]:
            axs.plot(y_range, ordershape_matrix[:, x], linewidth=0.05, color='grey')
        axs.plot(y_range_notnan, ordershapes_medianfiltered[:, -250], linewidth=0.5, color='red')

        axs.set_ylim([0.8 * np.nanpercentile(ordershape_matrix, 5), 1.2 * np.nanpercentile(ordershape_matrix, 95)])


        axs.set_title('Ordershapes for ordernr {} in fiber {}'.format(order_nr, fiber_nr))
        axs.set_ylabel('Signal [a.u.]')
        axs.set_xlabel('Pixel')
        print('')

        fig.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, 'Ordershape_fiber{}_order{}.png'.format(fiber_nr, order_nr))
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            plt.show()

        plt.close()

        print('Mid y of order {} at pixel {}'.format(order_nr, Centers[len(Centers)//2]))


    return ordershapes_medianfiltered

