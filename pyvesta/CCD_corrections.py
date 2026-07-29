import numpy as np
import matplotlib.pyplot as plt
import logging
import astroscrappy

from pyvesta import Spectra
from pyvesta import datashare

from numpy.polynomial.chebyshev import chebval
from scipy.sparse.linalg import lsqr, LinearOperator
from scipy import interpolate, ndimage


def _fitBackground_masked(image, weights, degx, degy, batch_width=256, damp=1e-4):
    """
    # Fit a 2D Chebyshev polynomial to an image to calculate the background /scattered light via LSQR.
    # Diffraction orders are masked by setting weights for those pixels to 0.
    #
    # This method will never calculate the full coefficient matrix and is therefore much more memory efficient than direct fitting.
    #
    # :param image: 2D numpy array of shape (H,W), contains the image to fit
    # :param weights: 2D numpy array of same shape as image, weights for each pixel. Mask pixels with weights of 0. All other pixels can have weights > 0, no need for binary weights
    # :param degx: int, polynomial degree in x direction
    # :param degy: int, polynomial degree in y direction
    # :param batch_width: int, number of pixel rows that are calculated at once. Larger values will increase speed but also memory usage (default 256)
    # :param damp: float, Tikhonov damping for LSQR. Increase if fit is noisy
    #
    # :returns coeffs: 2D numpy array of shape (degy+1, degx+1), coefficient array
    # :returns background: 2D numpy array of same shape as image, fitted background
    """

    H, W = image.shape
    px, py = degx+1, degy+1     #number of coefficients in each direction
    ncoeffs = px * py           #number of total coefficients

    #make sure weights are correct
    if weights.shape != image.shape:
        logging.warning('invalid weights, assume constant weights for background substraction. This might cause errors, as orders are no longer masked!')
        weights = np.ones_like(image)

    #use interaval [-1,1] as basis for chebychev polynimials
    x = np.linspace(-1.0, 1.0, W)
    y = np.linspace(-1.0, 1.0, H)

    #calculate pure chebyshev values for x and y. Use np.eye in combination with chebval to get individual chebyshev functions
    eye_x = np.eye(px)
    eye_y = np.eye(py)
    Tx    = np.array([chebval(x, eye_x[k]) for k in range(px)])
    Ty    = np.array([chebval(y, eye_y[j]) for j in range(py)])

    #Least square solves (sqrt(w) * (model - data))^2, so we need sqrt of weights.
    sq_weights = np.sqrt(np.maximum(weights, 0.0))

    #Model is defined by Matrix A width shape (H, W, coeffs) and model = A * c, where c is the flattened coefficient matrix.
    #For large pixtures and many coefficients, A is too big to be held in memory
    # However, all we need for LSQR is A * c for the forwards and A^T * r for the backwards direction
    # -> Use LinearOperator to simulate A without actually building it

    #forward direction, sqrt(W) * A * c
    def matvec(c):
        #reshape flattened coeff matrix
        C = c.reshape(py, px)

        #output array
        out = np.zeros((H, W)).astype(np.float64)

        #calculate batch_width rows at once
        for i0 in np.arange(0, H, batch_width).astype(int):
            #end of current batch
            i1 = np.min([i0 + batch_width, H])

            #Chebyshev polynoimials in y direction for current batch
            ty = Ty[:, i0:i1]

            #evaluate model by matrix calculation
            # model[row, col] = sum_(j,k) C[j,k] * Ty[j, row] * Tx[k, col] = Ty^T @ C @ Tx
            model = (ty.T @ C) @ Tx

            #apply weights
            out[i0:i1] = sq_weights[i0:i1] * model

        #flatten output array
        return out.ravel()

    #backward direction, sqrt(W) * A^T * r
    def rmatvec(r):
        #apply weights and reshape r
        R = sq_weights * r.reshape(H,W)

        #output array sqrt(w) * A^T * r
        Atr = np.zeros((py, px)).astype(np.float64)

        #calculate batch_width rows at once
        for i0 in np.arange(0, H, batch_width).astype(int):
            #end of current batch
            i1 = np.min([i0 + batch_width, H])

            #Chebyshev polynomials in y direction for current batch
            ty = Ty[:, i0:i1]

            #R at current batch
            Rb = R[i0:i1]

            Atr += ty @ Rb @ Tx.T

        #flatten output array
        return Atr.ravel()

    #Use LinearOperator to avoid building A
    N_obs = H * W
    A_op = LinearOperator(shape=(N_obs, ncoeffs), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    #Right hand side: sqrt(w) * y
    rhs = (sq_weights * image).ravel()

    #Solve with LSQR
    result = lsqr(A_op, rhs, damp=damp, iter_lim=500, show=False)

    coeffs_flat = result[0]
    coeffs      = coeffs_flat.reshape(py, px)

    #Evaluate full background model
    background = _eval_background(coeffs, Ty, Tx, H, W, batch_width)

    #clip background
    background = np.maximum(background, 0.0)

    return coeffs, background

def _eval_background(coeffs, Ty, Tx, H, W, batch_width=256):
    # evaluate Chebyshev model of background fit

    #output array
    bg = np.zeros(shape=(H,W)).astype(float)

    #go through batches
    #calculate batch_width rows at once
    for i0 in np.arange(0, H, batch_width).astype(int):
            #end of current batch
            i1 = np.min([i0 + batch_width, H])

            #Chebyshev polynoimials in y direction for current batch
            ty = Ty[:, i0:i1]

            #evaluate model by matrix calculation
            # model[row, col] = sum_(j,k) C[j,k] * Ty[j, row] * Tx[k, col] = Ty^T @ C @ Tx
            bg[i0:i1] = (ty.T @ coeffs) @ Tx

    return bg


def CreateBackground(image, Trace_data, degx=12, degy=20, buffer_size = 100):
    """
    # Create background image. Mask found orders to model only background.
    # Fit background with chebyshev polynomials
    #
    #
    # :param image: Image object, contains the image to fit
    # :param Trace_data: Trace_data object, contains the information about order positions and width
    # :param degx: int, polynomial degree in x (dispersion) direction (default 8)
    # :param degy: int, polynomial degree in y (cross-dispersion) direction (default 8)
    # :param buffer_size: int, pixel buffer at boundary orders to avoid interference from undetected orders (default 100)
    #
    # :return Background: Image object, contains the fitted background
    """

    data = image.data.copy()

    weights = np.ones_like(data)

    x_range = np.arange(data.shape[1]).astype(int)

    #mask all orders
    for fiber_nr in range(Trace_data.nr_of_fibers()):
        Fiber_trace = Trace_data.traces[fiber_nr]

        for ord_nr in range(Fiber_trace.nr_of_orders):
            trace = Fiber_trace.traces[ord_nr]

            trace.compute_centers(x_range)

            centers = trace.Centers
            sigma = np.round(trace.sigma).astype(int)

            min_inds = np.round(np.clip(centers - 3 * sigma, a_min=0, a_max = None)).astype(int)
            max_inds = np.round(np.clip(centers + 3 * sigma + 1, a_min = None, a_max=data.shape[0]-1)).astype(int)


            for x in x_range:
                weights[min_inds[x]:max_inds[x], x] = 0

    # create buffer at lowest and highest order. There might be not detected orders, which can disturb the background. Set weights close to edge orders to zero
    for x in x_range:
        zero_inds = np.where(weights[:,x] == 0)[0]

        min_ind = np.max((0, np.min(zero_inds) - buffer_size))
        max_ind = np.min((data.shape[0] - 1, np.max(zero_inds) + buffer_size + 1))

        weights[min_ind:np.min(zero_inds), x] = 0
        weights[np.max(zero_inds):max_ind, x] = 0

    #old method. Highly recommended to use new variant
    #coeffs = _backgroundfit(data, weights, degx, degy)
    #background = _evaluate_background(data, coeffs)
    coeffs, background = _fitBackground_masked(data, weights, degx, degy)


    if datashare.reduction_parameters.plot_Backgroundfit:
        fig, axs = plt.subplots(2,3, sharex=True, sharey=True)

        weighted_image = data * weights

        axs[0,0].imshow(data, vmax=np.nanpercentile(data, 95))
        axs[0,1].imshow(weights)
        axs[0,2].imshow(data - background, vmax = np.nanpercentile(data - background, 95))
        axs[1,0].imshow(weighted_image, vmax = np.nanpercentile(weighted_image, 95))
        axs[1,1].imshow(background)
        axs[1,2].imshow(weights * (data - background), vmax=np.nanpercentile(weights * (data - background), 95))

        axs[0,0].set_title('Image')
        axs[0,1].set_title('Weights')
        axs[0,2].set_title('Image - background')
        axs[1,0].set_title('Weighted image')
        axs[1,1].set_title('Background')
        axs[1,2].set_title('Weighted image - background')

        plt.tight_layout()

        if datashare.reduction_parameters.save_plots:
            filename = os.path.join(datashare.reduction_parameters.plot_dir, os.path.splitext(datashare.current_filename)[0] + "_Backgroundfit.png")
            plt.savefig(filename, dpi=300)

        if datashare.reduction_parameters.show_plots:
            print('') #needed to show plot in Jupyter Notebook
            plt.show()

        plt.close()


    return Spectra.Image(data=background, errors=None, header=image.header)

def filter_cosmics(image, **kwargs):
    """
    # Use the astroscrappy package to detect and remove cosmics in the images
    # astroscrappy is based on the algoritm of van Dokkum 2001, PASP, 113, 789, 1420 (http://adsabs.harvard.edu/abs/2001PASP..113.1420V)
    #
    # DO NOT use this on ThAr images, as this might filter some wanted ThAr emission lines.
    #
    # :param Image: Image object, contains the image to process
    # :param **kwargs: Additional parameters, passed over to astroscrappy.detect_cosmics()
    #
    # :return new_image: Image object with filtered flux
    """

    #filter cosmics using astro
    cosmics_inds, filtered_data = astroscrappy.detect_cosmics(image.data, gain=image.gain, readnoise=image.RON, satlevel=datashare.camera.get_maxcount(image.header), **kwargs)

    #FIXME: How to handle errors?

    return Spectra.Image(filtered_data, errors=image.errors, gain=image.gain, RON=image.RON, header=image.header)




def IsImageEmpty(data, biasmedian, biasrms, fac = 10):
    """
    # Returns True if image has no significant flux above bias level
    #
    # :param data: 2D numpy array, image to check
    # :param biasmedian: float, median of bias image
    # :param biasrms: float, RMS of bias signal
    # :param fac: float, minimal SNR of signal above biasmedian to be detected as significant (default 10)
    #
    # :returns result: boolean, whether image is empty
    """
    yshape, xshape = data.shape

    #check just middle part of image
    mid_data = data[yshape//4:-yshape//4, xshape//4:-xshape//4]

    #check if mean of middle image is above threshold
    return np.nanmean(mid_data) - biasmedian < fac * biasrms


def InterpolateBadRows(Image):
    """
    # Interpolate bad / dead camera rows by using neighbor pixels
    # This will only use the direct neighbors, therefore more than one dead row side by side will cause errors
    # The dead rows have to de identified MANUALLY and added in instruments.py
    #
    # :param Image: Image object, contains the image
    #
    # :return Image: Image object, Image with interpolated data
    """

    #no bad rows
    if datashare.camera.get_badrows(Image.header) is None or len(datashare.camera.get_badrows(Image.header)) == 0:
        return Image

    Image = Image.copy()

    data   = Image.data
    errors = Image.errors

    for bad_row in datashare.camera.get_badrows(Image.header):

        bad_row = bad_row - 1   #convert "normal" indices to software indices

        if bad_row == 0:
            data[0, :] = data[1,:]

            if errors is not None:
                errors[0, :] = 2 * errors[1,:] #larger errors due to interpolation

        elif bad_row == data.shape[0] - 1:
            data[-1,:] = data[-2,:]

            if errors is not None:
                errors[-1, :] = 2 * errors[-2,:] #larger errors due to interpolation

        else:
            data[bad_row, :] = 0.5 * (data[bad_row - 1, :] + data[bad_row + 1, :])

            if errors is not None:
                errors[bad_row, :] = errors[bad_row - 1, :] + errors[bad_row + 1, :] #larger errors due to interpolation

    Image.data   = data
    Image.errors = errors

    return Image

def CreateFlatImage(masterflat, Trace_data, median_width=100, min_SNR=10, maxdeviation=0.2):
    """
    # Create a Flat Image from a masterflat frame. This flat image has values around one.
    # The pixelwise flat values will be calculated by comparing the median spacial profile of a order with the spactial profile at the actual pixel position.
    # Pixels with low SNR and without relevant signal will have values of one.
    # Cannot be run in multiple threads, as all traces nee to write data to one array.
    #
    # Inspired by https://ui.adsabs.harvard.edu/abs/2002A%26A...385.1095P/abstract
    #
    # :param masterflat: Image object, contains the masterflat spectrum image
    # :param Trace_data: Trace_data object, contains the information about the order traces
    # :param median_width; int, width of median filter (default 100).
    # :param min_SNR: float, all pixels with a SNR lower than this value will have a value of one (default 10)
    # :param maxdeviation: float, between 0 and 1. All values with deviations larger than this value will also get a value of 1, as larger deviations are most likely defects and not flat correlated (default 0.2 (= 20%))
    #
    # :return flatimage: Image object, contains the flat image
    """


    image  = masterflat.data
    errors = masterflat.errors

    flat_data = np.ones_like(image).astype(float)


    #pixel range
    x_range = np.arange(image.shape[1]).astype(int)

    for fiber_nr in range(Trace_data.nr_of_fibers()):
        Fiber_trace = Trace_data.traces[fiber_nr]

        for trace in Fiber_trace.all_traces():
            y_len = np.round(6 * trace.sigma).astype(int)       #3 sigma in both directions

            #assure y_len is odd
            if y_len % 2 == 0:
                y_len += 1

            y_range = np.arange(y_len)

            trace.compute_centers(x_range)
            Centers   = trace.Centers

            ordershape_matrix =  np.zeros(shape=(y_len, len(x_range))).astype(float)
            ordershape_sum    =  np.ones(shape=len(x_range)).astype(float)

            weights = np.ones_like(x_range)

            #iteration over all pixels
            #first calculate shape of this order
            for x in x_range:
                center = Centers[x]

                min_idx_y = np.max((0, np.round(center - 3 * trace.sigma))).astype(int)
                max_idx_y = np.min((image.shape[0], np.round(center + 3 * trace.sigma))).astype(int)

                #get image window
                window       = image[min_idx_y:max_idx_y+1, x]
                error_window = errors[min_idx_y:max_idx_y+1, x]

                weights[x] = np.abs(np.sum(window)/np.sum(error_window))


                #we want to shift individual pixels sp, that the center always is exactly the center of y_range
                mid_y = center - min_idx_y

                order_shift = mid_y - y_len/2.

                ext_y_range = np.arange(0, max_idx_y - min_idx_y +1) - order_shift

                #interpolate ordershape. Do not extrapolate, but use NaNs instead
                ordershape_spline = interpolate.CubicSpline(ext_y_range, window, extrapolate=False)

                #evalate order at centered y range
                ordershape_matrix[:, x] = ordershape_spline(y_range)
                nansum = np.nansum(ordershape_matrix[:, x])

                ordershape_sum[x] = nansum

                ordershape_matrix[:, x] /= nansum


            ordershape_matrix[np.isnan(ordershape_matrix)] = 1e-10
            ordershape_matrix = np.clip(ordershape_matrix, a_min=1e-10, a_max=1)

            ordershape_fit = _fitOrdershape2D(ordershape_matrix, weights=weights)


            #print('')
            """
            fig = plt.figure(figsize=(6,6))
            axs = fig.add_subplot(2, 1, 2, projection='3d')

            Y = np.arange(ordershape_matrix.shape[0]) - ordershape_matrix.shape[0]/2.
            X = np.arange(ordershape_matrix.shape[1])

            XX, YY = np.meshgrid(X,Y)

            axs.plot_surface(XX, YY, ordershape_matrix)
            axs.plot_surface(XX, YY, ordershape_fit)

            fig.tight_layout()

            plt.show()
            plt.close('all')
            """

            norm_x = np.linspace(-1, 1, ordershape_matrix.shape[0])

            test_ordershape = ordershape_matrix[:, ordershape_matrix.shape[1]//2]

            """
            fig, axs = plt.subplots()

            axs.plot(norm_x, test_ordershape, color='black')

            for deg in range(10,11):
                coeffs = np.polynomial.chebyshev.chebfit(norm_x, test_ordershape, deg=deg)
                axs.plot(norm_x, np.polynomial.chebyshev.chebval(norm_x, coeffs), label='deg {}'.format(deg))

            axs.legend()

            plt.show()
            """

            #interpolate sum of ordershape
            #normalize x coordinates

            cheb_x_range =2 * x_range / np.max(x_range) - 1

            #coeffs   = np.polynomial.chebyshev.chebfit(cheb_x_range, ordershape_sum, deg=7)
            #eval_sum = np.polynomial.chebyshev.chebval(cheb_x_range, coeffs)

            eval_sum = ndimage.median_filter(ordershape_sum, size=101, mode='nearest')

            #plt.title('Sum Eval')
            #plt.plot(cheb_x_range, ordershape_sum)
            #plt.plot(cheb_x_range, eval_sum)
            #plt.show()

            #plt.plot(x_range, ordershape_sum)
            #plt.plot(x_range, eval_sum, color='red')
            #plt.show()

            """
            median_ordershapes = np.zeros(shape=(ordershape_matrix.shape[0], nsections))
            basis_points       = np.zeros(nsections).astype(int)


            for i in range(nsections):
                first_ind  = int(i     * ordershape_matrix.shape[1] / nsections)
                second_ind = int((i+1) * ordershape_matrix.shape[1] / nsections)

                basis_points[i] = int(0.5 * (first_ind + second_ind))

                median_ordershapes[:, i] = np.nanmedian(ordershape_matrix[:, first_ind:second_ind], axis=1)


            ordershapes_interpolated = np.zeros_like(ordershape_matrix)

            x_range_int = np.linspace(basis_points[0], basis_points[-1], num=basis_points[-1] - basis_points[0] +1, endpoint=True)

            for y in range(ordershapes_interpolated.shape[0]):
                interpolator = interpolate.PchipInterpolator(basis_points, median_ordershapes[y, :])
                #interpolator = interpolate.interp1d(basis_points, median_ordershapes[y, :])

                ordershapes_interpolated[y, basis_points[0]:basis_points[-1] +1] = interpolator(x_range_int)

                #extrapolate edges from first and last basis points
                ordershapes_interpolated[y, :basis_points[0]]  = ordershapes_interpolated[y, basis_points[0]]
                ordershapes_interpolated[y, basis_points[-1]:] = ordershapes_interpolated[y, basis_points[-1]]
            """

            #ordershapes_interpolated = ndimage.median_filter(ordershape_matrix, size=median_width, mode='nearest', axes=1)
            ordershapes_interpolated = ordershape_fit


            #filter y values where many values are nan
            #set median value to nan in that case
            for y in range(ordershape_matrix.shape[0]):
                nan_count = np.count_nonzero(np.isnan(ordershape_matrix[y,:]))

                if nan_count > 0.25 * ordershape_matrix.shape[1]:
                    ordershapes_interpolated[y, :] = np.nan

            #ensure that mid of ordershape is always mid of array
            while np.any(np.isnan(ordershapes_interpolated)):
                ordershapes_interpolated = ordershapes_interpolated[1:-1, :]

            #again iterate over all pixels
            #now calculate flat data
            for x in x_range:
                center = Centers[x]

                min_idx_y = np.max((0, np.round(center - 3 * trace.sigma))).astype(int)
                max_idx_y = np.min((image.shape[0], np.round(center + 3 * trace.sigma))).astype(int)

                #get image window
                window       = image[min_idx_y:max_idx_y+1, x].astype(float)
                error_window = errors[min_idx_y:max_idx_y+1, x].astype(float)
                flat_window  = flat_data[min_idx_y:max_idx_y+1, x].astype(float)

                #get shift between current pixel and median ordershape
                order_shift = (center - min_idx_y) - (ordershapes_interpolated.shape[0] / 2.)

                ordershape_spline = interpolate.CubicSpline(np.arange(ordershapes_interpolated.shape[0]), ordershapes_interpolated[:, x], extrapolate=False)

                ext_y_range     = np.arange(0, max_idx_y - min_idx_y +1).astype(float) - order_shift
                ordershape_eval = ordershape_spline(ext_y_range).astype(float)

                not_nan_inds = np.array(np.asarray(~np.isnan(ordershape_eval)).nonzero())

                #scale ordershape_eval and window to same median

                #get rough factor betweeen ordershape and window via median.
                #fac        = window[not_nan_inds] / ordershape_eval[not_nan_inds]
                #median_fac = np.nanmedian(fac)

                #residuals  = fac - median_fac
                #rms        = np.sqrt(np.sum(np.square(residuals)))

                #indices used to calculate factor between ordershape and window. Exclude outliners
                #fac_inds   = np.asarray(residuals <= 3 * rms).nonzero()
                #fac_inds   = not_nan_inds[fac_inds]

                #ordershape_eval *= np.nansum(window[fac_inds]) / np.nansum(ordershape_eval[fac_inds])
                ordershape_eval *= eval_sum[x]

                good_inds = np.asarray((window[not_nan_inds]/np.clip(error_window[not_nan_inds], a_min=1e-10, a_max=np.inf) > min_SNR) & (np.abs(window[not_nan_inds] - ordershape_eval[not_nan_inds]) / np.clip(ordershape_eval[not_nan_inds], a_min=1e-10, a_max=np.inf) < maxdeviation)).nonzero()

                good_inds= not_nan_inds[good_inds]


                flat_window[good_inds] = window[good_inds] / np.clip(ordershape_eval[good_inds], a_min=1e-10, a_max=np.inf)

                #just make sure that we do not have any big deviations
                flat_window[np.abs(flat_window - 1) > maxdeviation] = 1

                #transfer flat_window back to flat_data
                flat_data[min_idx_y:max_idx_y+1, x] = flat_window


                #plot at middle pixel, if requested
                if datashare.reduction_parameters.plot_FlatImage and x == x_range[-1]//2:
                    plt.plot(window, label='window')
                    plt.plot(ordershape_eval, label='median ordershape')

                    plt.legend()

                    if datashare.reduction_parameters.save_plots:
                        filename = os.path.join(datashare.reduction_parameters.plot_dir, "Flatimage.png")
                        plt.savefig(filename, dpi=300)

                    if datashare.reduction_parameters.show_plots:
                        print('') #needed to show plot in Jupyter Notebook
                        plt.show()

                    plt.close()

    flatimage = Spectra.Image(flat_data, errors=np.zeros_like(flat_data))


    return flatimage


def _2dfit(ordershape_matrix, degx, degy, weights=None):
    x_norm = np.linspace(-1.0, 1.0, ordershape_matrix.shape[1])
    y_norm = np.linspace(-1.0, 1.0, ordershape_matrix.shape[0])

    vander_x = np.polynomial.chebyshev.chebvander(x_norm, degx)
    vander_y = np.polynomial.chebyshev.chebvander(y_norm, degy)

    ordershape_matrix_copy = ordershape_matrix.copy().T

    if weights is None:
        w_vander_x = vander_x
        w_matrix   = ordershape_matrix_copy
    else:
        w = np.maximum(weights, 0)

        sw = np.sqrt(w)[:, np.newaxis]
        w_vander_x = sw * vander_x
        w_matrix   = sw * ordershape_matrix_copy

    #print(w_vander_x.shape, w_matrix.shape, vander_y.shape)
    #print(np.linalg.pinv(w_vander_x).shape, w_matrix.shape, np.linalg.pinv(vander_y).T.shape)

    #solve equation using pseudo-invariant
    coeffs = np.linalg.pinv(w_vander_x) @ w_matrix @ np.linalg.pinv(vander_y).T

    #return fitted ordershapes
    result = vander_x @ coeffs @ vander_y.T

    #norm result at each pixel
    sums = np.sum(result, axis=1)
    result /= sums[:, np.newaxis]

    return result.T

def _fitOrdershape2D(ordershape_matrix, weights=None, nsigma_clip=3.0, niter_max=10):
    """
    # Fit Ordershape in 2D in dispersion and cross-dispersion direction to avoid bad ordershapes at low SNR regions
    #
    #
    """

    if weights is None:
        w_start = np.ones(ordershape_matrix.shape[1])
    else:
        w_start = np.array(weights).astype(float).copy()

    w = w_start.copy()

    for it in range(1, niter_max+1):
        fit = _2dfit(ordershape_matrix, datashare.instrument.ordershape_dispdeg,  datashare.instrument.ordershape_crossdispdeg, weights=weights)

        residuals = ordershape_matrix - fit
        res_rms   = np.sqrt(np.mean(np.square(residuals), axis=0))

        used_inds   = np.asarray(w > 0).nonzero()[0]
        med_rms     = np.median(res_rms[used_inds])
        med_rms_sig = np.median(np.abs(res_rms[used_inds] - med_rms))
        sigma       = 1.4826 * med_rms_sig if med_rms_sig > 0 else res_rms[used_inds].std()

        is_outliner = np.asarray(res_rms > med_rms + nsigma_clip * sigma).nonzero()[0]

        #new weights
        w_new = w_start.copy()
        w_new[is_outliner] = 0

        if np.array_equal(w, w_new):
            break

        w = w_new

    #clip negative values
    fit = np.maximum(fit, 1e-10)

    return fit



