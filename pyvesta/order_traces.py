###############################
# Created 2026/01/12
#
# Author: Lukas Stock
#
##############################


import numpy as np
import os
import logging
from scipy import signal, ndimage, optimize
from scipy.special import erf
from multiprocessing import Pool
import matplotlib.pyplot as plt

from pyvesta import Spectra
from pyvesta import FitFunctions
from pyvesta import datashare

def init_pools(reduction_parameters, instrument, camera, current_filename):
    datashare.reduction_parameters = reduction_parameters
    datashare.instrument           = instrument
    datashare.camera               = camera
    datashare.current_filename     = current_filename

def trace_orders(Image,  npools=None, startfrom=0, endat=-1, nsigmas=20., search_steps=5,  mid_width=5):
    """
    # Trace the orders in a spectrum.
    # Use Orderdef (preferred) or flat images for this method, as science spectra with large absorption bands might cause errors
    # Assumes orders are roughly parallel to x-axis and cross-dispersion direction is roughly parallel to y-axis
    # Can use multiple threads
    #
    # :param Image: Image object, image of spectrum to trace orders in
    # :param npools: int, number of parallel processes when using multithreading. If None, will use the one defined in config (default None)
    # :param startfrom: int, pixel row to start order recognition. Can be used to exclude some areas of the image (default 0, no lower boundary)
    # :param endat: int, pixel row to end order recognition. Can be used to exclude some areas of the image. -1 means no upper boundary (default -1)
    # :param nsigmas: float, ammount of standart deviations at which a point is recognized as an outliner (default 10)
    # :param search_steps: int, step with for search of order centers in pixels (default 5)
    # :param mid_width: int, size of median window (default 5)
    #
    # :return Trace_obj: Fiber_traces object, stores the information about the trace positions
    """

    #get data from image
    image  = Image.data
    errors = Image.errors
    RON    = Image.RON
    gain   = Image.gain

    # get number of threads from config
    if npools is None:
        npools = datashare.reduction_parameters.npools


    #get parameters from instrument object
    order_separation = datashare.instrument.order_separation
    ncoef            = datashare.instrument.order_deg
    orders_sigma     = datashare.instrument.orders_sigma
    image_slicer     = datashare.instrument.image_slicer
    nr_of_fibers     = datashare.instrument.nr_of_fibers

    #mid_width is total width, used is only half of that for indices
    mid_width_half = mid_width // 2

    #cut image if requested
    if endat == -1:
        image = image[startfrom:,:]
    else:
        image = image[startfrom:endat,:]

    #calculate errors if not specified in Image
    if errors is None:
        errors = np.sqrt(np.abs(image * gain) + np.power(RON, 2.)) / gain

    #set minimal values to 0
    image[image < 0] = 0

    # Cut along middle column, median combine middle-most columns
    # to find the the maxima of the orders
    midcolumn = int(0.5*image.shape[1])
    mid_image = np.median(image[:,midcolumn - mid_width_half:midcolumn + mid_width_half +1], axis=1)
    mid_errors = np.sqrt(np.mean(np.square(errors[:,midcolumn - mid_width_half:midcolumn + mid_width_half +1]), axis=1))

    y_range = np.arange(len(mid_image))


   #get peak indices. Allow closer peaks in case of image slicer
    if not image_slicer:
        peak_inds  = signal.find_peaks(mid_image, height = nsigmas*mid_errors, distance=order_separation)[0]
    else:
        prominence = 0.2 * mid_image
        peak_inds  = signal.find_peaks(mid_image, height = nsigmas*mid_errors, distance=orders_sigma, prominence=prominence)[0]

    if image_slicer:
        heights     = mid_image[peak_inds]
        height_diff = np.diff(heights)


        #median distances between the orders
        dists = np.diff(peak_inds).astype(int)

        #get most frequent value, this is the distance between the two fibers
        (values,counts) = np.unique(dists,return_counts=True)
        ind             = np.argmax(counts)

        slicer_dist     = values[ind]

        dist_tolerance = 1      #pixel
        height_tolerance = 0.5  #relative height tolerance

        upper_peaks = []        #upper_peaks[i] > lower_peaks[i], one pair of peaks
        lower_peaks = []

        for i in range(len(dists)):
            if peak_inds[i] in upper_peaks or peak_inds[i] in lower_peaks:
                continue

            if np.abs(dists[i] - slicer_dist) <= dist_tolerance and np.abs(height_diff[i])/np.max((heights[i], heights[i+1])) < height_tolerance:
                lower_peaks.append(peak_inds[i])
                upper_peaks.append(peak_inds[i+1])


    fit_results = []
    #fit each peak with a modified gaussian

    if not image_slicer:
        #just one peak per order, fit gaussian to each peak
        fitfunc = FitFunctions.ModGauss

        for peak in peak_inds:
            #Fit with modfied Gaussian
            if peak - order_separation//2 < 0 or peak + order_separation//2 > len(mid_image):
                continue

            fit_image = mid_image[peak - order_separation//2:peak+order_separation//2]
            x = np.arange(len(fit_image)) + peak - order_separation//2

            #guess params. A, center, sigma, p
            p0 = (mid_image[peak], peak, orders_sigma, 2)
            #set boundaries. ((lower_inds), (upper_inds))
            bounds = ((0, peak-orders_sigma, orders_sigma/10, 0.1), (np.inf, peak+orders_sigma, 5*orders_sigma, 5))

            #fit
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_image, p0=p0, bounds=bounds)

            fit_results.append([popt, pconv])
    else:
        #image slicer, fit double gaussian
        fitfunc = FitFunctions.ModDoubleGauss

        for ind in range(len(upper_peaks)):

            upper_peak = upper_peaks[ind]
            lower_peak = lower_peaks[ind]   #both arrays have the same size

            #assure upper_ind >= lower_ind
            if upper_peak < lower_peak:
                temp      = lower_peak
                lower_peak = upper_peak
                upper_peak = temp

            if upper_peak + order_separation//2 > len(mid_image) or lower_peak - order_separation//2 < 0:
                continue

            mid_ind = int(0.5 * (upper_peak + lower_peak))

            fit_image = mid_image[mid_ind  - order_separation//2:mid_ind+order_separation//2 + 1]
            x = np.arange(len(fit_image)) + mid_ind - order_separation//2

            #guess params. A1, center1, sigma1, p1, A2, center2, sigma2, p2
            p0 = (mid_image[lower_peak], lower_peak, orders_sigma, 2, mid_image[upper_peak], upper_peak, orders_sigma, 2)
            #set boundaries. ((lower_inds), (upper_inds))
            bounds = ((0, lower_peak-orders_sigma, orders_sigma/10, 0.1, 0, upper_peak-orders_sigma, orders_sigma/10, 0.1), (np.inf, lower_peak+orders_sigma, 5*orders_sigma, 5, np.inf, upper_peak+orders_sigma, 5*orders_sigma, 5))

            #fit
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_image, p0=p0, bounds=bounds)

            fit_results.append([popt, pconv])

            #plt.plot(x, fit_image)
            #plt.plot(x, FitFunctions.ModGauss(x, *popt), color='red')
            #plt.show()

    #filter by amplitude
    heights = []
    centers = []
    for result in fit_results:
        popt = result[0]
        if not image_slicer:
            heights.append(popt[0])
            centers.append(popt[1])
        else:
            heights.append(0.5 * (popt[0] + popt[4]))
            centers.append(0.5 * (popt[1] + popt[5]))


    heights_copy = heights.copy()

    #remove bad peaks
    #relative height difference should be max 3 for one fiber and max 10 for multiple fibers

    if nr_of_fibers > 1:
        limit = 10
    else:
        limit = 3

    peak_ind = 0
    while peak_ind < len(fit_results):
        if peak_ind == 0:
            #first peak. Remove, if relative peak height difference to next peak is higher than three
            if np.abs(heights[peak_ind] - heights[peak_ind+1])/heights[peak_ind] > limit:
                fit_results.pop(peak_ind)
                heights.pop(peak_ind)
                continue
        elif peak_ind == len(fit_results) - 1:
            #last peak. Remove, if relative peak height difference to previous peak is higher than three
            if np.abs(heights[peak_ind] - heights[peak_ind-1])/heights[peak_ind] > limit:
                fit_results.pop(peak_ind)
                heights.pop(peak_ind)
                continue
        else:
            #all other peaks. Remove, if relative peak height difference to next or previous peak is higher than three
            if np.abs(heights[peak_ind] - heights[peak_ind-1])/heights[peak_ind] > limit or \
               np.abs(heights[peak_ind] - heights[peak_ind+1])/heights[peak_ind] > limit:
                fit_results.pop(peak_ind)
                heights.pop(peak_ind)
                continue

        #if peak did not get removed, increase index by one. This is not necessary if peak gets removed, as that automatically shrinks the list (so effective index moves forwards)
        peak_ind += 1


    #create new lists with accepted peaks
    heights_new = []
    centers_new = []
    for result in fit_results:
        popt = result[0]
        if not image_slicer:
            heights_new.append(popt[0])
            centers_new.append(popt[1])
        else:
            heights_new.append(0.5 * (popt[0] + popt[4]))
            centers_new.append(0.5 * (popt[1] + popt[5]))

    #plot, if requested
    if datashare.reduction_parameters.plot_Ordertrace:
        order_fits = np.zeros_like(mid_image)

        for r in fit_results:
            popt = r[0]
            order_fits += fitfunc(np.arange(len(mid_image)), *popt)

        plt.plot(mid_image, linewidth=1)
        plt.plot(order_fits, linewidth=1)
        #plt.plot(mid_errors, linewidth=0.5, color='red')
        plt.scatter(centers, heights_copy, color='black', s=1, zorder=2)
        plt.scatter(centers_new, heights_new, color='red', s=1, zorder=2)
        plt.xlabel('Pixel')
        plt.ylabel('Median intensity in a.u.')
        plt.title('Cross dispersion intensity at middle column')

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, "Ordertrace_centerfit.png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            plt.show()

        plt.close()

    #now start the actual orcer tracing
    #fix for jupyter notebook. It breaks the kernel when running in multiple threads, reason unknown
    if npools <= 1:
        args = [(image, errors, r, mid_width_half, order_separation, orders_sigma, ncoef, nsigmas, search_steps) for r in fit_results]

        traces = []
        j = 0
        for arg in args:
            j += 1

            #trace orders
            if not image_slicer:
                traces.append(_trace_order(arg))
            else:
                traces.append(_trace_order_image_slicer(arg))
    else:
        #use multithreading to process the order simultaneously
        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera, datashare.current_filename)) as pool:
            args = [(image, errors, r, mid_width_half, order_separation, orders_sigma, ncoef, nsigmas, search_steps) for r in fit_results]

            #trace orders
            if not image_slicer:
                traces = pool.map(_trace_order, args)
            else:
                traces = pool.map(_trace_order_image_slicer, args)

    logging.info('Finished order tracing')

    #sort out bad traces:
    traces = [t for t in traces if t['coefs'] is not None]

    #sort by median curvature of the curves by looking at the positions in the middle and at the ends
    #Improved version, now compares to local median instead of global median. global median filters out too many orders when orders are not perfectly parallel (e.g. FEROS)
    left_dists = [np.abs(np.polynomial.chebyshev.chebval(-1, t['coefs']) - np.polynomial.chebyshev.chebval(0, t['coefs'])) for t in traces]
    right_dists =[np.abs(np.polynomial.chebyshev.chebval(1, t['coefs']) - np.polynomial.chebyshev.chebval(0, t['coefs'])) for t in traces]

    filter_width = 11
    left_dists_median  = ndimage.median_filter(left_dists , size=filter_width, mode='mirror')
    right_dists_median = ndimage.median_filter(right_dists, size=filter_width, mode='mirror')

    traces = [traces[i] for i in range(len(traces)) if (np.abs((left_dists[i] - left_dists_median[i])/left_dists_median[i]) < 0.5 and np.abs((right_dists[i] - right_dists_median[i])/right_dists_median[i]) < 0.5 and traces[i]['coefs'] is not None)]

    #left_median = np.median(left_dists)
    #right_median = np.median(right_dists)

    #traces = [traces[i] for i in range(len(traces)) if (np.abs((left_dists[i] - left_median)/left_median) < 0.5 and np.abs((right_dists[i] - right_median)/right_median) < 0.5 and traces[i]['coefs'] is not None)]

    Centers = [t['Centers'] for t in traces]
    coefs = [t['coefs'] for t in traces]
    params= [t['trace_params'] for t in traces]


    #plot, if requested
    if datashare.reduction_parameters.plot_Ordertrace:
        fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(16,9))

        axs[0].imshow(image, vmin=0, vmax=1000)
        axs[1].imshow(image, vmin=0, vmax=1000)

        for c in Centers:
            all_x = [p.x for p in c][::10]
            all_y = [p.y for p in c][::10]

            axs[0].scatter(all_x, all_y, s=0.5, zorder=2)

        poly_x = np.linspace(-1.0, 1.0, image.shape[1])
        plot_x = np.arange(image.shape[1])

        for i,c in enumerate(coefs):
            if c is not None:
                #axs[1].plot(plot_x, np.polynomial.chebyshev.chebval(poly_x, c), label='Order {}'.format(i+1))
                axs[1].plot(plot_x[::10], np.polynomial.chebyshev.chebval(poly_x[::10], c))


        axs[0].set_xlabel('Pixel')
        axs[0].set_ylabel('Pixel')
        axs[1].set_xlabel('Pixel')
        axs[1].set_ylabel('Pixel')

        #axs[1].legend()

        axs[0].set_title('Found centers')
        axs[1].set_title('Fitted order positions')

        fig.suptitle('Order tracing')

        plt.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, "Ordertrace.png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            plt.show()

        plt.close()

    #create Trace_data object and add traces
    Trace_obj = Spectra.Trace_data()
    Trace_obj.add_traces(coefs, len(traces), Centers, image.shape[1]-1, fit_params=params)

    if nr_of_fibers > 1:
        Trace_obj.split_to_fibers(Image, order_multiplicity = nr_of_fibers)

    return Trace_obj


def findSingleOrder(Image, exp_center, nsigmas=3., search_steps=5, mid_width=5):
    """
    # Trace a single order. This is especially useful if one order of an instrument with multple fibers is missing
    # Use Orderdef (preferred) or flat images for this method, as science spectra with large absorption bands might cause errors
    # Assumes orders are roughly parallel to x-axis and cross-dispersion direction is roughly parallel to y-axis
    # Can use multiple threads
    #
    # :param Image: Image object, image of spectrum to trace orders in
    # :param exp_center: float, expected y position of the order at the middle pixel
    # :param nsigmas: float, ammount of standart deviations at which a point is recognized as an outliner (default 3, less than at order_traces to also detect weak orders)
    # :param search_steps: int, step with for search of order centers in pixels (default 5)
    # :param mid_width: int, size of median window (default 5)
    #
    # :return Trace_obj: Fiber_traces object, stores the information about the trace positions
    """

    #get data from image
    image  = Image.data
    errors = Image.errors
    RON    = Image.RON
    gain   = Image.gain

    #get parameters from instrument object
    order_separation = datashare.instrument.order_separation
    ncoef            = datashare.instrument.order_deg
    orders_sigma     = datashare.instrument.orders_sigma
    image_slicer     = datashare.instrument.image_slicer
    nr_of_fibers     = datashare.instrument.nr_of_fibers

    #mid_width is total width, used is only half of that for indices
    mid_width_half = mid_width // 2

    #calculate errors if not specified in Image
    if errors is None:
        errors = np.sqrt(np.abs(image * gain) + np.power(RON, 2.)) / gain

    #set minimal values to 0
    image[image < 0] = 0

    exp_center = np.round(exp_center).astype(int)

    min_y = np.max((exp_center - order_separation // 2, 0)).astype(int)
    max_y = np.min((exp_center + order_separation // 2, image.shape[1])).astype(int)

    # Cut along middle column, median combine middle-most columns
    # to find the the maxima of the orders
    midcolumn = int(0.5*image.shape[1])
    mid_image = np.median(image[min_y:max_y,midcolumn - mid_width_half:midcolumn + mid_width_half +1], axis=1)
    mid_errors = np.median(errors[min_y:max_y,midcolumn - mid_width_half:midcolumn + mid_width_half +1], axis=1)

    y_range = np.arange(min_y,max_y)

    if not image_slicer:
        #just one peak per order, fit modified gaussian
        fitfunc = FitFunctions.ModGauss

        #guess params. A, center, sigma, p
        p0 = (mid_image[exp_center - min_y], exp_center, orders_sigma, 2)
        #set boundaries. ((lower_inds), (upper_inds))
        bounds = ((0, exp_center-orders_sigma, orders_sigma/10, 0.1), (np.inf, exp_center+orders_sigma, 5*orders_sigma, 5))

        #fit
        popt, pconv = optimize.curve_fit(fitfunc, y_range, mid_image, p0=p0, bounds=bounds)

    else:
        #image slicer, fit double gaussian
        fitfunc = FitFunctions.ModDoubleGauss

        lower_peak = exp_center - orders_sigma
        upper_peak = exp_center + orders_sigma

        #guess params. A1, center1, sigma1, p1, A2, center2, sigma2, p2
        p0 = (mid_image[exp_center - min_y], lower_peak, orders_sigma, 2, mid_image[exp_center - min_y], upper_peak, orders_sigma, 2)
        #set boundaries. ((lower_inds), (upper_inds))
        bounds = ((0, lower_peak-orders_sigma, orders_sigma/10, 0.1, 0, upper_peak-orders_sigma, orders_sigma/10, 0.1), (np.inf, lower_peak+orders_sigma, 5*orders_sigma, 5, np.inf, upper_peak+orders_sigma, 5*orders_sigma, 5))

        #fit
        popt, pconv = optimize.curve_fit(fitfunc, y_range, mid_image, p0=p0, bounds=bounds)

    args = (image, errors, [popt, pconv], mid_width_half, order_separation, orders_sigma, ncoef, nsigmas, search_steps)

    #try to find order
    if not image_slicer:
        trace_params = _trace_order(args)
    else:
        trace_params = _trace_order_image_slicer(args)


    Centers = trace_params['Centers']
    coefs   = trace_params['coefs']
    params  = trace_params['trace_params']

    #bad fit, return None
    if coefs is None:
        return None
    else:

        #plot, if requested
        if datashare.reduction_parameters.plot_Ordertrace:
            fig, axs = plt.subplots(1,2, sharex=True, sharey=True, figsize=(16,9))

            axs[0].imshow(image, vmin=0, vmax=1000)
            axs[1].imshow(image, vmin=0, vmax=1000)

            all_x = np.array([p.x for p in Centers])
            all_y = np.array([p.y for p in Centers])

            normed_x = 2 * (all_x / image.shape[1]) - 1

            axs[0].scatter(all_x, all_y, s=0.5, zorder=2)

            x_range_norm = np.linspace(-1.0, 1.0, image.shape[1])
            x_range      = np.arange(image.shape[1])

            axs[1].plot(x_range, np.polynomial.chebyshev.chebval(x_range_norm, coefs))


            axs[0].set_xlabel('Pixel')
            axs[0].set_ylabel('Pixel')
            axs[1].set_xlabel('Pixel')
            axs[1].set_ylabel('Pixel')

            #axs[1].legend()

            axs[0].set_title('Found centers')
            axs[1].set_title('Fitted order positions')

            fig.suptitle('Order tracing')

            plt.tight_layout()

            if datashare.reduction_parameters.save_plots:
                filename = os.path.join(datashare.reduction_parameters.plot_dir, "Singleorder_trace.png")
                plt.savefig(filename, dpi=300)

            if datashare.reduction_parameters.show_plots:
                plt.show()

            plt.close()


        return Spectra.Trace(coefs, Centers, image.shape[1], fit_params=params)



#this method traces the orders. Can be used in multiple threads simultaneously
#use this in case of no image slicer
def _trace_order(args):
    image, errors, mid_fit, mid_width_half, order_separation, orders_sigma, ncoef, nsigmas, search_steps = args

    #just one peak per order, fit modified gaussian
    fitfunc = FitFunctions.ModGauss

    #go to the right side, starting at the x-center
    column = int(0.5*image.shape[1]) + search_steps     #already fitted mid_column
    trace_results = [mid_fit]


    Centers = [Spectra.Pixel(int(0.5*image.shape[1]), mid_fit[0][1])]  #center pixel of order

    while column + mid_width_half +1 < image.shape[1]:       #go through pixels, Centers list will get longer
        all_x = np.array([p.x for p in Centers])
        all_y = np.array([p.y for p in Centers])

        all_x_norm  = 2 * (all_x / (image.shape[1] -1)) - 1
        column_norm = 2 * (column / (image.shape[1] -1)) - 1


        #center of last column
        last_center = all_y[-1]
        last_result = trace_results[-1]

        #if enough data points: Interpolate where center is expected
        if len(trace_results) > ncoef * ncoef:
            coefs = np.polynomial.chebyshev.chebfit(all_x_norm, all_y, deg=ncoef)
            exp_center = np.round(np.polynomial.chebyshev.chebval(column_norm, coefs)).astype(int)

        else:
            #else use last center
            exp_center = np.round(last_center).astype(int)

        #if order exceeds image boundaries (in y-direction) break, do not track this order
        if exp_center - order_separation//2 < 0 or exp_center + order_separation//2 >= image.shape[0]:
            break

        #median image at current column
        fit_image = np.median(image[exp_center-order_separation//2:exp_center+order_separation//2,column-mid_width_half:column+mid_width_half +1], axis=1)
        x = np.arange(len(fit_image)) + exp_center - order_separation//2

        #check whether there is some significant signal, if not skip this pixel
        if np.max(fit_image) < nsigmas * np.median(errors[exp_center-order_separation//2:exp_center+order_separation//2,column-mid_width_half:column+mid_width_half +1]):
            column +=  search_steps
            continue

        # No image slicer, fit one gaussian

        #use last results as template
        last_amplitude = last_result[0][0]
        last_center    = last_result[0][1]
        last_sigma     = last_result[0][2]
        last_p         = last_result[0][3]

        #guess params. A, center, sigma, p
        p0 = (last_amplitude, exp_center, last_sigma, last_p)
        #set boundaries. ((lower_inds), (upper_inds))
        bounds = ((0, exp_center-orders_sigma, orders_sigma/10, 0.1), (np.inf, exp_center+orders_sigma, 5*orders_sigma, 5))

        #fit
        try:
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_image, p0=p0, bounds=bounds)
        except:
            column +=  search_steps
            continue

        amplitude = popt[0]
        center    = popt[1]
        sigma     = popt[2]
        p         = popt[3]


        if np.abs(center - exp_center) > 2 * orders_sigma or \
            np.abs((amplitude - last_amplitude)/amplitude) > 0.5 or \
            np.abs((sigma - last_sigma)/sigma) > 0.5 or \
            np.abs((p - last_p)/p) > 0.5:
            #wrong fit, reject
            pass
        else:
            #append at end
            trace_results.append([popt, pconv])
            Centers.append(Spectra.Pixel(column, center))

        #continue search at next pixel
        column +=  search_steps

    #print('finished right')

    #now go back to the middle and move left
    column = int(0.5*image.shape[1]) - search_steps     #already fitted mid_column
    while column - mid_width_half > 0:
        all_x = np.array([p.x for p in Centers])
        all_y = np.array([p.y for p in Centers])

        all_x_norm  = 2 * (all_x / (image.shape[1] -1)) - 1
        column_norm = 2 * (column / (image.shape[1] -1)) - 1


        #center of last column (here we attend at the front)
        last_center = all_y[0]
        last_result = trace_results[0]

        #if enough data points: Interpolate where center is expected
        if len(trace_results) > ncoef * ncoef:
            coefs = np.polynomial.chebyshev.chebfit(all_x_norm, all_y, deg=ncoef)
            exp_center = np.round(np.polynomial.chebyshev.chebval(column_norm, coefs)).astype(int)

        else:
            #else use last center
            exp_center = np.round(last_center).astype(int)

        #if order exceeds image boundaries (in y-direction) break, do not track this order
        if exp_center - order_separation//2 < 0 or exp_center + order_separation//2 >= image.shape[0]:
            break

        #median image at current column
        fit_image = np.median(image[exp_center-order_separation//2:exp_center+order_separation//2,column-mid_width_half:column+mid_width_half +1], axis=1)
        x = np.arange(len(fit_image)) + exp_center - order_separation//2

        #check whether there is some significant signal
        if np.max(fit_image) < nsigmas * np.median(errors[exp_center-order_separation//2:exp_center+order_separation//2,column-mid_width_half:column+mid_width_half +1]):
            column -= search_steps
            continue

        #use last results as templatelast_amplitude = last_result[0][0]
        last_amplitude = last_result[0][0]
        last_center    = last_result[0][1]
        last_sigma     = last_result[0][2]
        last_p         = last_result[0][3]

        #guess params. A, center, sigma, p
        p0 = (last_amplitude, exp_center, last_sigma, last_p)
        #set boundaries. ((lower_inds), (upper_inds))
        bounds = ((0, exp_center-orders_sigma, orders_sigma/10, 0.1), (np.inf, exp_center+orders_sigma, 5*orders_sigma, 5))

        #fit
        try:
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_image, p0=p0, bounds=bounds)
        except:
            column -= search_steps
            continue

        amplitude = popt[0]
        center    = popt[1]
        sigma     = popt[2]
        p         = popt[3]

        if np.abs(center - exp_center) > 2 * orders_sigma or \
            np.abs((amplitude - last_amplitude)/amplitude) > 0.5 or \
            np.abs((sigma - last_sigma)/sigma) > 0.5 or \
            np.abs((p - last_p)/p) > 0.5:
            #wrong fit, reject
            pass
        else:
            #append at front
            trace_results.insert(0, [popt, pconv])
            Centers.insert(0, Spectra.Pixel(column, center))

        #continue search at next pixel (move to the left)
        column -=  search_steps

    #print('finished left')

    #all Center coordinates
    all_x = np.array([p.x for p in Centers])
    all_y = np.array([p.y for p in Centers])

    all_x_norm  = 2 * (all_x / (image.shape[1] -1)) - 1

    #fit order with polynomial
    if len(all_x) >= ncoef * ncoef:
        order_coefs = np.polynomial.chebyshev.chebfit(all_x_norm, all_y, deg=ncoef)
    else:
        order_coefs = None

    median_params = {}

    median_params['sigma'] = np.median([res[0][2] for res in trace_results])
    median_params['p']     = np.median([res[0][3] for res in trace_results])

    #return fit coefficients, found order centers and median parameters of order (shape of order in cross-dispersion direction)
    return {'coefs':order_coefs, 'Centers':Centers, 'trace_params':median_params}

#this method traces the orders. Can be used in multiple threads simultaneously
#use this in case of an image slicer
def _trace_order_image_slicer(args):
    image, errors, mid_fit, mid_width_half, order_separation, orders_sigma, ncoef, nsigmas, search_steps = args

    #two peaks per order, fit double modified gaussian
    fitfunc = FitFunctions.ModDoubleGauss

    #go to the right side, starting at the x-center
    column = int(0.5*image.shape[1]) + search_steps     #already fitted mid_column
    trace_results = [mid_fit]


    Centers1 = [Spectra.Pixel(int(0.5*image.shape[1]), mid_fit[0][1])]  #center pixel of order
    Centers2 = [Spectra.Pixel(int(0.5*image.shape[1]), mid_fit[0][5])]  #center pixel of order

    while column + mid_width_half +1 < image.shape[1]:       #go through pixels, Centers list will get longer
        all_x1 = np.array([p.x for p in Centers1])
        all_y1 = np.array([p.y for p in Centers1])
        all_x2 = np.array([p.x for p in Centers2])    #should be the same as all_x1
        all_y2 = np.array([p.y for p in Centers2])

        all_x1_norm = 2 * (all_x1 / (image.shape[1] -1)) -1
        all_x2_norm = 2 * (all_x2 / (image.shape[1] -1)) -1
        column_norm = 2 * (column / (image.shape[1] -1)) -1


        #center of last column
        last_center1 = all_y1[-1]
        last_center2 = all_y2[-1]
        last_result = trace_results[-1]

        #if enough data points: Interpolate where center is expected
        if len(trace_results) > 5 * ncoef:
            coefs1 = np.polynomial.chebyshev.chebfit(all_x1_norm, all_y1, deg=ncoef)
            exp_center1 = np.round(np.polynomial.chebyshev.chebval(column_norm, coefs1)).astype(int)
            coefs2 = np.polynomial.chebyshev.chebfit(all_x2_norm, all_y2, deg=ncoef)
            exp_center2 = np.round(np.polynomial.chebyshev.chebval(column_norm, coefs2)).astype(int)

        else:
            #else use last center
            exp_center1 = np.round(last_center1).astype(int)
            exp_center2 = np.round(last_center2).astype(int)

        #assure exp_center1 > exp_center2
        if exp_center2 > exp_center1:
            temp        = exp_center2
            exp_center2 = exp_center1
            exp_center1 = temp

        #if order exceeds image boundaries (in y-direction) break, do not track this order
        if exp_center1 - order_separation//2 < 0 or exp_center1 + order_separation//2 >= image.shape[0]  or \
            exp_center2 - order_separation//2 < 0 or exp_center2 + order_separation//2 >= image.shape[0]:
            break

        mid_exp_center = np.round(0.5 * (exp_center1 + exp_center2)).astype(int)

        #median image at current column
        fit_image = np.median(image[mid_exp_center-order_separation//2:mid_exp_center+order_separation//2 +1,column-mid_width_half:column+mid_width_half +1], axis=1)
        x = np.arange(len(fit_image)) + mid_exp_center - order_separation//2

        #check whether there is some significant signal, if not skip this pixel
        if np.max(fit_image) < nsigmas * np.median(errors[mid_exp_center-order_separation//2:mid_exp_center+order_separation//2 +1,column-mid_width_half:column+mid_width_half +1]):
            column +=  search_steps
            continue

        # image slicer, fit double gaussian

        #use last results as template
        last_amplitude1 = last_result[0][0]
        last_center1    = last_result[0][1]
        last_sigma1     = last_result[0][2]
        last_p1         = last_result[0][3]
        last_amplitude2 = last_result[0][4]
        last_center2    = last_result[0][5]
        last_sigma2     = last_result[0][6]
        last_p2         = last_result[0][7]

        #guess params. A1, center1, sigma1, p1, A2, center2, sigma2, p2
        p0 = (last_amplitude1, exp_center1, last_sigma1, last_p1, last_amplitude2, exp_center2, last_sigma2, last_p2)
        #set boundaries. ((lower_inds), (upper_inds))
        bounds = ((0, exp_center1-orders_sigma, orders_sigma/10, 0.1, 0, exp_center2-orders_sigma, orders_sigma/10, 0.1), (np.inf, exp_center1+orders_sigma, 5*orders_sigma, 5, np.inf, exp_center2+orders_sigma, 5*orders_sigma, 5))

        #fit
        try:
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_image, p0=p0, bounds=bounds)
        except:
            column +=  search_steps
            continue

        amplitude1 = popt[0]
        center1    = popt[1]
        sigma1     = popt[2]
        p1         = popt[3]
        amplitude2 = popt[4]
        center2    = popt[5]
        sigma2     = popt[6]
        p2         = popt[7]

        if (center1 - exp_center1) > 5 * orders_sigma or \
            np.abs((amplitude1 - last_amplitude1)/amplitude1) > 0.5 or \
            np.abs((sigma1 - last_sigma1)/sigma1) > 0.5 or \
            np.abs((p1 - last_p1)/p1 > 0.5) or \
            (np.abs(center2 - exp_center2) > 5 * orders_sigma or \
            np.abs((amplitude2 - last_amplitude2)/amplitude2) > 0.5 or \
            np.abs((sigma2 - last_sigma2)/sigma2) > 0.5 or \
            np.abs((p2 - last_p2)/p2) > 0.5):
            #wrong fit, reject
            pass
        else:
            #append at end
            trace_results.append([popt, pconv])
            Centers1.append(Spectra.Pixel(column, center1))
            Centers2.append(Spectra.Pixel(column, center2))

        #continue search at next pixel
        column +=  search_steps

    #print('finished right')

    #now go back to the middle and move left
    column = int(0.5*image.shape[1]) - search_steps     #already fitted mid_column
    while column - mid_width_half > 0:
        all_x1 = np.array([p.x for p in Centers1])
        all_y1 = np.array([p.y for p in Centers1])
        all_x2 = np.array([p.x for p in Centers2])    #should be the same as all_x1
        all_y2 = np.array([p.y for p in Centers2])

        all_x1_norm = 2 * (all_x1 / (image.shape[1] -1)) -1
        all_x2_norm = 2 * (all_x2 / (image.shape[1] -1)) -1
        column_norm = 2 * (column / (image.shape[1] -1)) -1

        #center of last column, we attend at front, so first element
        last_center1 = all_y1[0]
        last_center2 = all_y2[0]
        last_result = trace_results[0]

        #if enough data points: Interpolate where center is expected
        if len(trace_results) > 5 * ncoef:
            coefs1 = np.polynomial.chebyshev.chebfit(all_x1_norm, all_y1, deg=ncoef)
            exp_center1 = np.round(np.polynomial.chebyshev.chebval(column_norm, coefs1)).astype(int)
            coefs2 = np.polynomial.chebyshev.chebfit(all_x2_norm, all_y2, deg=ncoef)
            exp_center2 = np.round(np.polynomial.chebyshev.chebval(column_norm, coefs2)).astype(int)

        else:
            #else use last center
            exp_center1 = np.round(last_center1).astype(int)
            exp_center2 = np.round(last_center2).astype(int)

        #assure exp_center1 > exp_center2
        if exp_center2 > exp_center1:
            temp        = exp_center2
            exp_center2 = exp_center1
            exp_center1 = temp

        #if order exceeds image boundaries (in y-direction) break, do not track this order
        if exp_center1 - order_separation//2 < 0 or exp_center1 + order_separation//2 >= image.shape[0]  or \
            exp_center2 - order_separation//2 < 0 or exp_center2 + order_separation//2 >= image.shape[0]:
            break

        mid_exp_center = np.round(0.5 * (exp_center1 + exp_center2)).astype(int)

        #median image at current column
        fit_image = np.median(image[mid_exp_center-order_separation//2:mid_exp_center+order_separation//2 +1,column-mid_width_half:column+mid_width_half +1], axis=1)
        x = np.arange(len(fit_image)) + mid_exp_center - order_separation//2

        #check whether there is some significant signal, if not skip this pixel
        if np.max(fit_image) < nsigmas * np.median(errors[mid_exp_center-order_separation//2:mid_exp_center+order_separation//2 +1,column-mid_width_half:column+mid_width_half +1]):
            column -=  search_steps
            continue
        # image slicer, fit double gaussian

        #use last results as template
        last_amplitude1 = last_result[0][0]
        last_center1    = last_result[0][1]
        last_sigma1     = last_result[0][2]
        last_p1         = last_result[0][3]
        last_amplitude2 = last_result[0][4]
        last_center2    = last_result[0][5]
        last_sigma2     = last_result[0][6]
        last_p2         = last_result[0][7]

        #guess params. A1, center1, sigma1, p1, A2, center2, sigma2, p2
        p0 = (last_amplitude1, exp_center1, last_sigma1, last_p1, last_amplitude2, exp_center2, last_sigma2, last_p2)
        #set boundaries. ((lower_inds), (upper_inds))
        bounds = ((0, exp_center1-orders_sigma, orders_sigma/10, 0.1, 0, exp_center2-orders_sigma, orders_sigma/10, 0.1), (np.inf, exp_center1+orders_sigma, 5*orders_sigma, 5, np.inf, exp_center2+orders_sigma, 5*orders_sigma, 5))

        #fit
        try:
            popt, pconv = optimize.curve_fit(fitfunc, x, fit_image, p0=p0, bounds=bounds)
        except:
            column -=  search_steps
            continue

        amplitude1 = popt[0]
        center1    = popt[1]
        sigma1     = popt[2]
        p1         = popt[3]
        amplitude2 = popt[4]
        center2    = popt[5]
        sigma2     = popt[6]
        p2         = popt[7]

        if (center1 - exp_center1) > 5 * orders_sigma or \
            np.abs((amplitude1 - last_amplitude1)/amplitude1) > 0.5 or \
            np.abs((sigma1 - last_sigma1)/sigma1) > 0.5 or \
            np.abs((p1 - last_p1)/p1 > 0.5) or \
            (np.abs(center2 - exp_center2) > 5 * orders_sigma or \
            np.abs((amplitude2 - last_amplitude2)/amplitude2) > 0.5 or \
            np.abs((sigma2 - last_sigma2)/sigma2) > 0.5 or \
            np.abs((p2 - last_p2)/p2) > 0.5):
            #wrong fit, reject
            pass
        else:
            #append at front
            trace_results.insert(0, [popt, pconv])
            Centers1.insert(0, Spectra.Pixel(column, center1))
            Centers2.insert(0, Spectra.Pixel(column, center2))

        #continue search at next pixel (move to the left)
        column -=  search_steps

    #print('finished left')

    #calculate centers between both image slicer traces
    all_x1 = np.array([p.x for p in Centers1])
    all_y1 = np.array([p.y for p in Centers1])
    all_x2 = np.array([p.x for p in Centers2])   #should be the same as all_x1
    all_y2 = np.array([p.y for p in Centers2])

    Centers = [Spectra.Pixel(0.5 * (all_x1[i] + all_x2[i]), 0.5 * (all_y1[i] + all_y2[i])) for i in range(len(all_x1))]

    #all Center coordinates
    all_x = np.array([p.x for p in Centers])
    all_y = np.array([p.y for p in Centers])

    normed_x = 2 * (all_x / image.shape[1]) - 1

    #fit order with polynomial
    if len(all_x1) >= 5 * ncoef:
        order_coefs = np.polynomial.chebyshev.chebfit(normed_x, all_y, deg=ncoef)
    else:
        order_coefs = None

    median_params = {}

    median_params['sig1']   = np.median([res[0][2] for res in trace_results])
    median_params['sig2']   = np.median([res[0][6] for res in trace_results])
    median_params['p1']     = np.median([res[0][3] for res in trace_results])
    median_params['p2']     = np.median([res[0][4] for res in trace_results])

    #return fit coefficients, found order centers and median parameters of order (shape of order in cross-dispersion direction)
    return {'coefs':order_coefs, 'Centers':Centers, 'trace_params':median_params}

