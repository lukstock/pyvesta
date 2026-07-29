import numpy as np
from scipy import interpolate, ndimage
import logging
from astroquery.simbad import Simbad
import astropy.coordinates as coord
import astropy.units as u
from astropy.time import Time

from pyvesta import Spectra
from pyvesta import datashare


def filter_outliners(flux, errors = None, median_width=5, nsigmas=5, niter=5):
    """
    # Filter outliners (hot / dead pixels, dead rows, cosmics etc.) in flux by comparing it with the local median. Large deviations from the median will be replaced by the median
    #
    # :param flux: numpy 1D array, spectral flux
    # :param errors: numpy 1D array, same size as flux, contains the errors of flux. If not None, will set the errors of interpolated pixels to inf (default None)
    # :param median_width: int, width of median filter, minimum of 3 (default 5)
    # :param nsigmas: float, threshold of filtering in multiples of local rms to median (default 5)
    # :param niter: int, how often the filter routine will run (default 5)
    #
    # :return new_flux: numpy 1D array, filtered flux
    # :return new_errors: numpy 1D array, errors of filtered flux
    """

    #median_width must be odd and min 3
    if median_width % 2 == 0:
        median_width += 1
    if median_width < 3:
        median_width = 3

    if niter < 1:
        niter = 1

    #set median filter width. Ignore current value for median calculation
    footprint   = np.ones(median_width)
    footprint[median_width//2] = 0

    new_flux = flux.copy()

    if errors is not None:
        new_errors = errors.copy()
    else:
        new_errors = None

    for _ in range(niter):
        #calculate median
        median_flux = ndimage.median_filter(new_flux, footprint=footprint, mode='nearest')

        residuals = new_flux - median_flux

        #calculate local rms
        res_square = np.square(residuals)
        #again ignore current value
        window = np.ones(median_width)/float(median_width -1)
        window[median_width // 2] = 0

        local_rms = np.sqrt(np.convolve(res_square, window, mode='same'))

        #search for bad indices
        bad_inds = np.where(np.abs(residuals) > nsigmas * local_rms)

        #replace flux
        new_flux[bad_inds] = median_flux[bad_inds]

        #replace errors, if available
        if new_errors is not None:
            new_errors[bad_inds] = np.inf
        else:
            new_errors = None

    return new_flux, new_errors

def  _combine_orders(fluxes, wavs, errors = None, min_SNR = 10, fill = 'zero', **kwargs):
    """
    Combine multiple orders of an echelle spectrum
    Idea from https://spectrum.readthedocs.io/en/latest/_modules/spectrum/coadd.html / wave_little_interpol
    Some issues fixed:
        Orders don't need to overlap
        Gaps between orders are interpolated/ set to zero / set to nan

    This algorithm merges echelle orders to one continous spectrum.
    Where the orders overlap the orders are interpolated, e.g.:
        aaaaaaa
             bbbbbbbbb
                    cccccccc
                                dddddddd
    The adding at the overlapping regions will be error-weighted, if errors are available.
    In the ranges the orders do not overlap the original wavelength grid will be kept. If there are gaps between orders
    (here between c and d) they are per default set to c, wavelength bin size will be mean of the ones at the front / end.


    Note: The resulting wavelenth grid is **not** equally spaced! Also orders are not allowed to totally overlap (each order has to have a part without overlapping)
    Parameters
    ----------
    fluxes: list of 1d arrays
        list of input fluxes
    wavs:   list of 1d arrays
        list of input wavelengths
    errors: list of 1d array or None
        list of flux errors. Error weighted coadding if errors are available
    fill: 'zero', 'nan' or 'interpolate'
        how to fill the gaps between orders, 'zero' by default
    min_SNR: float
        pixels with SNR lower than this threshold are ignored. Only used if errors is not None. Default 10

    **kwargs are passed to scipy.interpolate.interpol1d

    Returns
    -------
    fluxout: ndarray
        flux of merged spectrum
    waveout: ndarray
        wavelengths of merged spectrum
    errout: ndarray
        flux errors of merged spectrum
    """


    if len(fluxes) != len(wavs):
        raise ValueError('fluxes and wavelengths have to have the same size')
    if errors is not None and len(errors) != len(fluxes):
        raise ValueError('fluxes and errors habe to have the same size')

    #just one order, return as is
    if len(fluxes) == 1:
        err = errors[0] if errors is not None else None
        return fluxes[0], wavs[0], err

    fluxes = np.array(fluxes)
    wavs   = np.array(wavs)

    if errors is not None:
        errors = np.array(errors)

    #minimal and maximal wavelengths of each order
    mins = np.array([min(wavs[i][fluxes[i] > 0]) for i in range(len(wavs))])
    maxs = np.array([max(wavs[i][fluxes[i] > 0]) for i in range(len(wavs))])

    if np.any(np.argsort(mins) != np.arange(len(wavs))):
        raise ValueError('List of wavelengths must be sorted in increasing order.')
    if np.any(np.argsort(maxs) != np.arange(len(wavs))):
        raise ValueError('List of wavelengths must be sorted in increasing order.')

    min_wav = np.min(mins[mins > 0])
    max_wav = np.max(maxs)

    #last_wav is the wavelength we already visited, the spectrum is already merged for wave < last_wav
    last_wav = min_wav

    #these are the limits where either a order ends or the next order starts
    limits   = np.sort(np.append(mins, maxs))

    waveout = []
    fluxout = []
    errout = []


    #go though wavelengths
    while last_wav < max_wav:
        #get next limit
        next_wav = _NextGreater(last_wav, limits)

        wav_parts  = []
        flux_parts = []
        err_parts  = []

        #check all orders if they have wavelengths in the current wavelength section
        for i in range(len(wavs)):
            #valid spectral range of this order
            if (mins[i] <= last_wav and maxs[i] >= next_wav):
                #use only region where flux is not 0 (mostly at the edges flux is set to 0)
                matched_inds = np.where((wavs[i] <= next_wav) & (wavs[i] >= last_wav) & (np.abs(fluxes[i]) > 0))[0]

                if len(matched_inds) >= 3:       #min 3 points for interpolation, else too few points to be relevant
                    wav_parts.append(wavs[i][matched_inds])
                    flux_parts.append(fluxes[i][matched_inds])

                    if errors is not None:
                        err_parts.append(errors[i][matched_inds])


        #check if overlap or not
        if len(flux_parts) == 0:
            ##### no data in this region ####
            #get mean wavelength bin size of last / first 10% of leading and following order

            #index of last_wav and next_wav
            i0 = np.argmin(np.abs(maxs - last_wav))
            i1 = np.argmin(np.abs(mins - next_wav))

            #wavelength parts used for interpolations
            lead_wav_part = wavs[i0][:-int(0.1 * len(wavs[i0]))]
            foll_wav_part = wavs[i1][:int(0.1 * len(wavs[i1]))]

            #get bin sizes
            #In overlap region patch in a linear scale with slightly different step.
            dw = mins[i1] - maxs[i0]
            step = 0.5*(np.mean(np.diff(lead_wav_part)) + np.mean(np.diff(foll_wav_part)))
            n_steps = int(dw / step + 0.5)

            new_wavs = np.linspace(maxs[i0] + step, mins[i1] - step, n_steps - 1)

            #flux
            if fill == 'zero':
                new_flux = [0 for n in range(1, n_steps)]
                new_errs = [0 for n in range(1, n_steps)]
            elif fill == 'nan':
                new_flux = [np.nan for n in range(1, n_steps)]
                new_errs = [np.nan for n in range(1, n_steps)]
            elif fill == 'interpolate':
                lead_flux_part = fluxes[i0][:-int(0.1 * len(wavs[i0]))]
                foll_flux_part = fluxes[i1][:int(0.1 * len(wavs[i1]))]

                flux_parts = np.append(lead_flux_part, foll_flux_part)
                wav_parts  = np.append(lead_wav_part, foll_wav_part)

                new_flux = _interpolate(flux_parts, wav_parts, new_wavs, **kwargs)
                new_flux = np.array(new_flux)

                #TODO: ERROR ESTIMATION!? Currently interpolating
                if errors is not None:
                    lead_err_part = errors[i0][:-int(0.1 * len(wavs[i0]))]
                    foll_err_part = errors[i1][:int(0.1 * len(wavs[i1]))]

                    err_parts = np.append(lead_err_part, foll_err_part)
                    new_errs = _interpolate(err_parts, wav_parts, new_wavs, **kwargs)

            else:
                raise ValueError('fill must be either \'zero\', \'nan\' or \'interpolate\'')

            waveout.append(new_wavs)
            fluxout.append(new_flux)

            if errors is not None:
                errout.append(new_errs)

        if len(flux_parts) == 1:
            ### only one spectrum ###

            waveout.append(wav_parts[0])
            fluxout.append(flux_parts[0])

            if errors is not None:
                errout.append(err_parts[0])

        elif len(flux_parts) > 0:
            #### overlap region ####

            #calculate flux in overlap region
            #check if flux error is given

            if errors is not None:
                #use error-weighted coadding
                new_flux, new_wav, new_errors = _coadd_errorweighted(flux_parts, wav_parts, err_parts, min_SNR=min_SNR)

                waveout.append(new_wav)
                fluxout.append(new_flux)
                errout.append(new_errors)

            else:
                #use simple coadding, no errors
                new_flux, new_wav = _coadd_simple(flux_parts, wav_parts)

                waveout.append(new_wav)
                fluxout.append(new_flux)


        last_wav = next_wav

    #append fluxes, wavelengths and errors to one array
    fluxout = np.hstack(fluxout)
    waveout = np.hstack(waveout)

    if errors is None:
        #no errors, so set them to -1
        errout = np.zeros(fluxout.shape) - 1.0
    else:
        errout = np.hstack(errout)

    """
    #filter errors in wavelength computation:
    while True:
        bad_inds = np.where(np.diff(waveout) < 0)[0] + 1

        if len(bad_inds) == 0:
            break

        fluxout = np.delete(fluxout, bad_inds)
        waveout = np.delete(waveout, bad_inds)
        errout  = np.delete(errout,  bad_inds)
    """

    return fluxout, waveout, errout


def _NextGreater(value, array):
    """
    # Get minimal entry v from array so that v > value. If no such value is found return nan
    # Important! assert array is sorted!
    #
    # :param value: float, test value
    # :param array: list or np.array, array to search next greater value
    #
    # :param v: float or nan, next greater value
    """

    for v in array:
        if v > value:
            return v


    return np.nan


def _coadd_simple(fluxes, wavs, new_wav_grid = None, **kwargs):
    """
    Simple coadding of spectra in the same spectral range
    Idea from https://spectrum.readthedocs.io/en/latest/_modules/spectrum/coadd.html

    All spectra are interpolated to 'new_wav_grid'. If 'new_wav_grid' is None the wavelength grid from wavs[0] is used.
    Errors are ignored.

    Parameters
    ----------
    fluxes: list of 1d arrays
        fluxes of the spectra
    wavs: list of 1d arrays
        wavelength grid of the spectra
    new_wav_grid: 1d array or 'None'
        new wavelength grid, if 'None' the first one from 'wavs' is used

    **kwargs are passed through to scipy.interpolate.interp1d

    Returns
    -------
    fluxout: ndarray
        coadded flux
    wavout: ndarray
        wavelength grid of fluxout, same as new_wav_grid if available

    """

    if not len(fluxes) == len(wavs):
        raise ValueError('fluxes and wavs have to have same size!')

    #use wavelengths of first order if no wav_grid is given
    if new_wav_grid is None:
        new_wav_grid = np.array(wavs[0])
    else:
        new_wav_grid = np.array(new_wav_grid)

    #remove too low / too high values from new_wav
    inds_too_low  = np.where(new_wav_grid < np.max(np.min(wavs, axis=0)))[0]
    inds_too_high = np.where(new_wav_grid > np.min(np.max(wavs, axis=0)))[0]

    bad_inds = np.append(inds_too_low, inds_too_high)

    new_wav_grid = np.delete(new_wav_grid, bad_inds, axis=0)


    new_fluxes = []

    #interpolate fluxes
    for i in range(len(fluxes)):
        f = fluxes[i]
        w = wavs[i]

        #interpolate using linear splines
        new_f = _interpolate(f, w, new_wav_grid, kind='linear')

        new_fluxes.append(new_f)


    #use masked arrays to ignore NANs etc.
    fluxes = np.stack(new_fluxes)
    fluxes  = np.ma.fix_invalid(fluxes)
    #build mean
    fluxout  = fluxes.mean(axis = 0).filled(fill_value = np.nan)

    return fluxout, new_wav_grid

def _coadd_errorweighted(fluxes, wavs, errors, new_wav_grid = None, min_SNR=10, **kwargs):
    """
    Simple coadding of spectra in the same spectral range
    Idea from https://spectrum.readthedocs.io/en/latest/_modules/spectrum/coadd.html

    All spectra are interpolated to 'new_wav_grid'. If 'new_wav_grid' is None the wavelength grid from the order with the best SNR ist used.
    coadding based on errors.

    Parameters
    ----------
    fluxes: list of 1d arrays
        fluxes of the spectra
    wavs: list of 1d arrays
        wavelength grid of the spectra
    errors: list of 1d arrays
        flux errors of the spectra
    new_wav_grid: 1d array or 'None'
        new wavelength grid, if 'None' the first one from 'wavs' is used
    min_SNR: float
        pixels with SNR lower than this threshold are ignored. Default 10

    **kwargs are passed through to scipy.interpolate.interp1d

    Returns
    -------
    fluxout: ndarray
        coadded flux
    wavout: ndarray
        wavelength grid of fluxout, same as new_wav_grid if available
     errout: ndarray
        errors of fluxout

    """

    if len(fluxes) == 0:
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

    if (not len(fluxes) == len(wavs)):
        raise ValueError('fluxes and wavs have to have same size!')
    if (not len(fluxes) == len(errors)):
        raise ValueError('fluxes and errors have to have same size!')

    #use wavelength grid from order with best SNR if no wave_grid was given
    if new_wav_grid is None:
        SNRs = [np.nanmedian(fluxes[i]/errors[i])  for i in range(len(fluxes))]
        best_order = np.argmax(SNRs)

        new_wav_grid = np.array(wavs[best_order])
    else:
        new_wav_grid = np.array(new_wav_grid)

    #remove too low / too high values from new_wav
    inds_too_low  = np.where(new_wav_grid <= np.max([np.min(w) for w in wavs]))[0]
    inds_too_high = np.where(new_wav_grid >= np.min([np.max(w) for w in wavs]))[0]

    bad_inds = np.append(inds_too_low, inds_too_high)

    new_wav_grid = np.delete(new_wav_grid, bad_inds, axis=0)

    new_fluxes = []
    new_errs   = []

    #interpolate fluxes and errors
    #TODO: Maybe better error estimations
    for i in range(len(fluxes)):
        f = fluxes[i]
        w = wavs[i]
        e = errors[i]

        #already correct wavelength scale
        if len(w) == len(new_wav_grid) and np.all(w == new_wav_grid):
            new_fluxes.append(f)
            new_errs.append(e)
        else:
            #interpolate flux and errors using linear splines
            new_f = _interpolate(f, w, new_wav_grid, kind='linear')
            new_e = _interpolate(e, w, new_wav_grid, kind='linear')

            #smooth errors, avoids distortions from single too small errors
            new_e = ndimage.median_filter(new_e, size=21)

            new_fluxes.append(new_f)
            new_errs.append(new_e)

    #stack flux and errors in one matrix
    fluxes = np.stack(new_fluxes)
    errors = np.stack(new_errs)

    #weights matrix
    weights = np.power(errors, -2.)

    #set weights to zero for values with no errors
    weights[np.isnan(errors)] = 0

    #set weights to zero in case of too low SNR
    weights[np.where(np.abs(fluxes/errors) < min_SNR)] = 0

    #check for pixels with just zero weights.
    #First try to just mask values with no errors
    #If still no weights for that pixel replace those weights with one (may result in bad combinations, but still better than a nan value)
    bad_pixels = np.where(np.sum(weights, axis=0) <= 0)

    weights[:, bad_pixels] =  np.power(errors[:, bad_pixels], -2.)
    weights[:, bad_pixels][np.isnan(errors[:, bad_pixels])] = 0

    #check again
    bad_pixels = np.where(np.sum(weights, axis=0) <= 0)
    weights[:, bad_pixels] = 1

    #finally combine the orders
    flux   = np.average(fluxes, axis=0, weights=weights)
    error  = 1./np.sqrt(np.sum(weights, axis=0))

    # check if all single measurements are within 3 error ranges to combined flux
    # if not, just use flux with highest SNR
    #if np.any(np.abs(fluxes - flux[None, :]) > 3 * errors):
    #    maxSNR_inds = np.argmax(fluxes/errors, axis=0)
    #    cols = np.arange(fluxes.shape[1])
    #
    #    flux  = fluxes[maxSNR_inds, cols]
    #    error = errors[maxSNR_inds, cols]

    return flux, new_wav_grid, error

def _interpolate(flux, wav, new_wav, **kwargs):
    """
    # Interpolate flux from wav to new_wav
    #
    # :param flux: 1D numpy array, flux of spectrum
    # :param wav: 1D numpy array, old wavelength range (same size as flux)
    # :param new_wav: 1D numpy array, new wavelength range. Must not exceed old wavelength range in both directions
    # :param kwargs: additional parameters, these are passed over to scipy.interpolate.interp1d
    #
    # :param interpol_flux: 1D numpy array, interpolated flux (same size as new_wav)
    """
    wav     = np.array(wav)
    new_wav = np.array(new_wav)

    if (max(new_wav) > max(wav)) or (min(new_wav) < min(wav)):
        raise ValueError('new_wav interval is greater than wav interval\nWe do only interpolation and no extrapolation!')

    #create interpolator
    interpol = interpolate.interp1d(wav, flux, **kwargs)

    #evaluate
    return interpol(new_wav)


def getBaryCorr(header):
    """
    # Get barycentric velocity of telescope and barycentric MJD to correct for movement, roation and position of earth
    #
    # :param header: header of exposure (dict like)
    #
    # :return BCMJD: float, barycentric MJD of exposure
    # :return RV_corr: float, barycentric velocity in km/s
    """

    #no header, cannot do anything
    if header is None:
        return np.nan, np.nan

    try:
        #try to get barycentric correction

        #get observatory location
        observatory = coord.EarthLocation(lat=datashare.instrument.lat*u.deg, lon=datashare.instrument.lon*u.deg, height=datashare.instrument.alt*u.m)

        RA, DEC = datashare.instrument.getStarCoordinates(header)

        coords = coord.SkyCoord(RA, DEC, unit=(u.hourangle, u.deg),frame='icrs')

        # time
        mjd    = datashare.instrument.getMJD(header)
        MJDUTC = Time(mjd, format='mjd', scale='utc', location=observatory)

        #get BCMJD
        ltt_bary      = MJDUTC.light_travel_time(coords)
        barycorr_time = MJDUTC.tdb + ltt_bary

        BCMJD         = barycorr_time.mjd

        #now heliocentric RV correction
        barycorr = coords.radial_velocity_correction(obstime=MJDUTC)
        RV_corr  = barycorr.to(u.km/u.s).value

        return BCMJD, RV_corr

    except Exception as e:
        logging.warning('Barycentric velocity calculation failed with exception \'{}\''.format(e))
        return np.nan, np.nan
