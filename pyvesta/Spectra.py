import numpy as np
import copy
from scipy import interpolate, optimize
from scipy.ndimage import median_filter
from scipy.linalg import solve_banded
from multiprocessing import Pool

import os
import glob
import logging

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyvesta import CCD_corrections
from pyvesta import wavelength_calibration
from pyvesta import spectrum_operations
from pyvesta import order_traces
from pyvesta import FitFunctions
from pyvesta import iocomp
from pyvesta import instruments
from pyvesta import datashare

def _gauss(x, mu, sigma):
    return np.exp(-np.square(x - mu) / np.square(sigma))

class Constants:
    c = 299792.458      #speed of light in km/s

def init_pools(reduction_parameters, instrument, camera):
    datashare.reduction_parameters = reduction_parameters
    datashare.instrument           = instrument
    datashare.camera               = camera

class GeneralOrder:
    ################
    # General class to save data from one spectral order (raw or calibrated)
    # To be specialized
    #################

    def __init__(self, ordernr = 0):
        self.ordernr = ordernr

    def copy(self):
        return copy.deepcopy(self)

class Target:
    ###############
    # Class to store information about the target star
    #
    ###############

    def __init__(self, name, coordinates=None):
        self.name = name
        self.coordinates=coordinates

class ObservationMode:
    """
    # Just a dummy class to identify different kinds of observations, mainly light and calibrations
    """

    LIGHT    = 0
    BIAS     = 1
    DARK     = 2
    FLAT     = 3
    ORDERDEF = 4
    THAR     = 5

class GeneralSpectrum:
    ##############
    # General class to save spectral orders (raw or calibrated)
    # To be specialized
    #############
    def __init__(self):
        self._orders = []

        self._order_type = None
        self.header      = None

    def addOrder(self, order, index=-1):
        assert np.issubdtype(type(index), np.integer)

        if np.abs(index) >= len(self._orders):
            index = -1

        if not isinstance(order, self._order_type):
            raise ValueError('order has wrong type! Should be {}, but is {}.'.format(str(type(order).__name__), str(self._order_type.__name__)))

        if index > 0:
            self._orders.insert(index, order)
        else:
            self._orders.append(order)

    def allOrders(self):
        return [order.copy() for order in self._orders]

    def nr_of_orders(self):
        return len(self._orders)

    def cut_to_good_values(self, threshold = 5):
        for order in self._orders:
            order.cut_to_good_values(threshold=threshold)


    def removeOrder(self, index):
        assert np.issubdtype(type(index), np.integer)

        if np.abs(index) >= len(self._orders):
            raise ValueError('index out of bounds!')

        del self._orders[index]

        for ordernr, order in self._orders:
            order.ordernr = ordernr

    def reverse_orders(self):
        self._orders = self._orders[::-1]

        for i in range(len(self._orders)):
            self._orders[i].ordernr = (len(self._orders) -1) - self._orders[i].ordernr

    def reverse_pixels(self):
        for order in self._orders:
            order.reverse_pixels()


    def __getitem__(self, index):
        assert np.issubdtype(type(index), np.integer)

        if np.abs(index) > len(self._orders):
            raise ValueError('index out of bounds!')

        return self._orders[index]

    def __setitem__(self, index, value):
        assert np.issubdtype(type(index), np.integer)

        if np.abs(index) >= len(self._orders):
            raise ValueError('index out of bounds!')

        if not isinstance(value, self._order_type):
            raise ValueError('order has wrong type! Should be {}, but is {}.'.format(str(type(value).__name__), str(self._order_type.__name__)))


        self._orders[index] = value

    def __add__(self, other):
        assert isinstance(other, type(self))

        if self.nr_of_orders() != other.nr_of_orders():
            raise ValueError('spectra do not have the same number of orders!')

        new_spectrum = type(self)()
        new_spectrum.header = self.header

        for order_nr in range(self.nr_of_orders()):
            new_order = self[order_nr] + other[order_nr]
            new_spectrum.addOrder(new_order)

        return new_spectrum

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        assert isinstance(other, type(self))

        if self.nr_of_orders() != other.nr_of_orders():
            raise ValueError('spectra do not have the same number of orders!')

        new_spectrum = type(self)()
        new_spectrum.header = self.header

        for order_nr in range(self.nr_of_orders()):
            new_order = self[order_nr] * other[order_nr]
            new_spectrum.addOrder(new_order)

        return new_spectrum

    def __truediv__(self, other):
        return self * (~other)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other


    def __neg__(self):
        new_spectrum = type(self)()
        new_spectrum.header = self.header

        for order_nr in range(self.nr_of_orders()):
            new_order = -self[order_nr]
            new_spectrum.addOrder(new_order)

        return new_spectrum

    def __invert__(self):
        new_spectrum = type(self)()
        new_spectrum.header = self.header

        for order_nr in range(self.nr_of_orders()):
            new_order = ~self[order_nr]
            new_spectrum.addOrder(new_order)

        return new_spectrum

    def copy(self):
        return copy.deepcopy(self)


class Telescope:
    def __init__(self, lat=0, lon=0, alt=0, name=''):
        self.lat  = lat
        self.lon  = lon
        self.alt  = alt
        self.name = name

    def from_header(header):
        if header is None:
            return None

        telescope = Telescope()

        if 'TELESCOPE NAME' in header:
            telescope.name = header['TELESCOPE NAME']
        if 'TELESCOPE LAT'  in header and 'TELESCOPE LON'  in header and \
           'TELESCOPE ALT' in header:

            try:
                telescope.lat  = float(header['TELESCOPE LAT'])
                telescope.lon  = float(header['TELESCOPE LON'])
                telescope.alt  = float(header['TELESCOPE ALT'])
            except Exception as e:
                return None
        else:
            return None


        return telescope

class SolutionHandler:
    def __init__(self, base_dir, suffix=''):
        self.base_dir     = base_dir
        self.solution_dir = None
        self.suffix       = suffix

        self.solutions    = []
        self.filenames    = []

    def sort_by_index(self, index, reverse=False):
        self.solutions = helpers.sort_by_index(self.solutions, index, reverse=reverse)
        self.filenames = helpers.sort_by_index(self.filenames, index, reverse=reverse)

    def makeDirectories(self):
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        self.solution_dir = os.path.join(self.base_dir, 'wavelength_solutions')

        if not os.path.exists(self.solution_dir):
            os.mkdir(self.solution_dir)

    def saveSolution(self, Solution, filename, modify_filename=True, log=True):
        if modify_filename:
            filename = os.path.basename(filename)
            if filename[:-5] != '.hdf5':
                filename += '.hdf5'

            filename = filename.replace('.hdf5', self.suffix + '.hdf5')

            save_filename = os.path.join(self.solution_dir, filename)
        else:
            save_filename = filename

        if (not save_filename in self.filenames) and log:
            self.solutions.append(Solution)
            self.filenames.append(save_filename)

        Solution.save(save_filename)

    def loadSolution(self, filename, modify_filename = True, log=True):
        if modify_filename:
            filename = os.path.basename(filename)
            if filename[-5:] != '.hdf5':
                filename += '.hdf5'

            filename = filename.replace('.hdf5', self.suffix + '.hdf5')

            save_filename = os.path.join(self.solution_dir, filename)
        else:
            save_filename = filename

        Solution = FinalWavelengthSolution.from_file(save_filename)

        if (not save_filename in self.filenames) and log:
            self.solutions.append(Solution)
            self.filenames.append(save_filename)

        return Solution

    def SolutionExists(self, filename, modify_filename = True):
        if modify_filename:
            filename = os.path.basename(filename)
            if filename[-5:] != '.hdf5':
                filename += '.hdf5'

            filename = filename.replace('.hdf5', self.suffix + '.hdf5')

            new_filename = os.path.join(self.solution_dir, filename)
        else:
            new_filename = filename

        return os.path.exists(new_filename)

    def logExisting(self, filename, **kwargs):
        self.loadSolution(filename, **kwargs)

    def makeReference(self,  reference_filename='', shift_method='cubic'):
        """
        # Create a reference wavelenth solution with an interpolator, that interpolates the radial velocity shift over the night
        #
        # :param reference_filename: str, filename of reference wavelength slolution. If not '' will save reference wavelength solution to this file (default '')
        # :param shift_method: str, must be one of ['median', 'linear', 'cubic', 'smoothed', 'temperature']. See create_shifts_interpolator for more details (default 'cubic')
        #
        # :return Reference_Solution: FinalWavelengthSolution object, reference wavelength solution with an interpolator. Can be used to calculate the wavelengths at any time during the night
        """

        #group ThAr measurements together
        self.solutions, master_ind, combined_mjds, combined_shifts, shift_errors, combined_temps, temp_errors = wavelength_calibration.ThArGroupMedian(self.solutions, deltaT=10)

        #get reference wavelength solution
        Reference_Solution = self.solutions[master_ind].copy()

        """
        solution_errors = []

        for solution in self.solutions:
            solution_errors.append(solution.rms)


        solution_errors = np.array(solution_errors)

        solution_errors = solution_errors[solution_errors <= 5 * np.nanmedian(solution_errors)]

        #shift_rms       = np.sqrt(np.sum(np.square(shifts))) / len(shifts)
        shift_rms       = (np.max(shifts) - np.min(shifts))/2.
        total_shift_rms = np.sqrt(np.square(shift_rms * len(shifts)) + np.sum(np.square(solution_errors))) / len(solution_errors)
        final_rms       = np.sqrt(np.square(total_shift_rms) + np.square(Reference_Solution.rms))
        """

        #create interpolator
        Reference_Solution.create_shifts_interpolator(combined_mjds, combined_shifts, method=shift_method, temps=combined_temps, rms=shift_errors)

        #residuals
        residuals = combined_shifts - Reference_Solution.v_shift_interpolator(combined_mjds)

        #plot, if requested
        if datashare.reduction_parameters.plot_WaveReference:
            temp_mjds = np.linspace(np.min(combined_mjds), np.max(combined_mjds), 1000)

            #TODO: Scatter Mastersolution seperately and mark this to make it more clear

            fig, axs = plt.subplots(2, figsize=(15,7))
            axs[0].errorbar(combined_mjds - np.min(combined_mjds), combined_shifts, yerr=shift_errors, fmt='o', label='ThAr measurements')
            axs[0].plot(temp_mjds - np.min(combined_mjds), Reference_Solution.v_shift_interpolator(temp_mjds), label='Interpolation')
            axs[0].set_ylabel('Shifts to master solution in km/s')
            axs[0].set_xlabel('MJD - {}'.format(np.round(np.min(combined_mjds), 2)))
            axs[0].legend()

            axs[1].errorbar(combined_mjds - np.min(combined_mjds), residuals , yerr=shift_errors, fmt='o')
            axs[1].set_ylabel('Residuals to interpolation')
            axs[1].set_xlabel('MJD - {}'.format(np.round(np.min(combined_mjds), 2)))

            plt.tight_layout()

            if datashare.reduction_parameters.save_plots:
                filename = os.path.join(datashare.reduction_parameters.plot_dir, "Wavelenghtinterpolation.png")
                plt.savefig(filename, dpi=300)

            if datashare.reduction_parameters.show_plots:
                plt.show()

            plt.close()


        #calculate and print final wavelength RMS
        weights = np.power(shift_errors, -2)        #weights = 1./error^2
        final_rms = np.sqrt((np.sum(weights * np.square(residuals) + np.sum(weights * np.square(shift_errors))))/ np.sum(weights))


        Reference_Solution.final_rms = final_rms

        logging.info('Final RMS is {} km/s'.format(np.around(final_rms, 3)))

        #save reference wavelength solution
        if reference_filename != '':
            #do not add reference solution to own solution list
            if reference_filename[-5:] != '.hdf5':
                reference_filename += '.hdf5'

            Reference_Solution.save(reference_filename)

        return Reference_Solution

class FileHandler:
    def __init__(self):
        self.filenames = {
            ObservationMode.FLAT    : [],
            ObservationMode.ORDERDEF: [],
            ObservationMode.BIAS    : [],
            ObservationMode.DARK    : [],
            ObservationMode.THAR    : [],
            ObservationMode.LIGHT   : []
            }

        self.directories = {
            ObservationMode.FLAT    : '',
            ObservationMode.ORDERDEF: '',
            ObservationMode.BIAS    : '',
            ObservationMode.DARK    : '',
            ObservationMode.THAR    : '',
            ObservationMode.LIGHT   : ''
            }

        self.mjds ={
            ObservationMode.FLAT    : [],
            ObservationMode.ORDERDEF: [],
            ObservationMode.BIAS    : [],
            ObservationMode.DARK    : [],
            ObservationMode.THAR    : [],
            ObservationMode.LIGHT   : []
            }

        self.dir_standard_names = {
            ObservationMode.FLAT    : 'flat',
            ObservationMode.ORDERDEF: 'orderdef',
            ObservationMode.BIAS    : 'bias',
            ObservationMode.DARK    : 'dark',
            ObservationMode.THAR    : 'thar',
            ObservationMode.LIGHT   : 'light'
            }


    def sortbyMJD(self):
        for key in self.filenames.keys():
            if len(self.mjds[key]) > 1:
                sorted_inds = np.argsort(self.mjds[key])

                self.filenames[key] = [self.filenames[key][i] for i in sorted_inds]


class RawHandler(FileHandler):
    def __init__(self, index = 0):
        super().__init__()

        #used if fits file has multiple hdus / images (e.g. HARPS)
        self.index = index


    def _loadGeneric(self, directory, exposure_type):
        if not os.path.exists(directory):
            raise ValueError('Directory doesn\'t exist')

        for fits_filename in glob.glob(os.path.join(directory, '*.fits')):
            header = iocomp.getFitsHeader(fits_filename)

            header_exptype = datashare.instrument.getExpMode(header)

            if header_exptype == exposure_type:
                self.filenames[exposure_type].append(fits_filename)

                mjd = datashare.instrument.getMJD(header)

                if mjd == 0 or np.isnan(mjd):
                    self.mjds[exposure_type].append(0)
                else:
                    self.mjds[exposure_type].append(mjd)

            #self.filenames[exposure_type] = list(np.unique(self.filenames[exposure_type]))

        self.directories[exposure_type] = directory

    def loadFlats(self, flat_dir):
        self._loadGenereric(flat_dir, ObservationMode.FLAT)

    def loadOrderdef(self, orderdef_dir):
        self._loadGeneric(orderdef_dir, ObservationMode.ORDERDEF)

    def loadThAr(self, ThAr_dir):
        self._loadGeneric(Thar_dir, ObservationMode.THAR)

    def loadBias(self, Bias_dir):
        self._loadGeneric(Bias_dir, ObservationMode.BIAS)

    def loadDark(self, Dark_dir):
        self._loadGeneric(Dark_dir, ObservationMode.DARK)

    def loadLight(self, Light_dir):
        self._loadGeneric(Light_dir, ObservationMode.LIGHT)


    def autoLoad(self, base_dir):
        subdirectories = glob.glob(os.path.join(base_dir, '*/'))

        #go through sub-dirs
        if len(subdirectories) > 0:
            for subdir in subdirectories:
                #open first file
                all_filenames = glob.glob(os.path.join(subdir, '*.fits'))
                if len(all_filenames) < 1:
                    continue
                else:
                    #check files till we find one with correct header
                    for filename in all_filenames:
                        header = iocomp.getFitsHeader(filename)
                        if header is not None and datashare.instrument.getObjectName(header) != '':
                            break

                    if header is None:
                        continue

                    exposure_type = datashare.instrument.getExpMode(header)

                    if exposure_type is not None and self.directories[exposure_type].lower() == '':
                        self._loadGeneric(subdir, exposure_type)

        #all files in one directory, no subdirectories. Search all files in base_dir
        else:
            for exposure_type in self.directories.keys():
                self._loadGeneric(base_dir, exposure_type)


    def generateDarks(self, master_darks):
        # get all exposure times of light frames

        light_exposuretimes = []
        for fits_filename in self.filenames[ObservationMode.LIGHT]:
            header = iocomp.getFitsHeader(fits_filename)

            try:
                exptime = datashare.instrument.getExptime(header)

                if exptime != 0 and not np.isnan(exptime):
                    light_exposuretimes.append(exptime)

            except:
                pass

        #nothing to do, we know nothing about the light frames
        if len(light_exposuretimes) < 1:
            return master_darks

        light_exposuretimes = np.sort(light_exposuretimes)


        # get all dark exposure times
        dark_exposuretimes = []
        for fits_filename in self.filenames[ObservationMode.DARK]:
            header = iocomp.getFitsHeader(fits_filename)

            try:
                exptime = datashare.instrument.getExptime(header)

                if exptime != 0 and not np.isnan(exptime):
                    dark_exposuretimes.append(exptime)
            except:
                pass

        dark_exposuretimes = np.sort(dark_exposuretimes)

        #get unique exposure times
        u_lightexp = np.sort(np.unique(light_exposuretimes))
        u_darkexp  = np.sort(np.unique(dark_exposuretimes))

        #no darks
        if len(u_darkexp) < 1:
            #generate empty dark files
            #open one light file to get image size

            data, errors, header = iocomp.image_from_file_fits(filename, transpose=datashare.instrument.transpose_image, index=self.index)

            #generate empty image
            dark_data   = np.zeros_like(data)
            dark_errors = np.zeros_like(errors)

            Dark_image = Image(dark_data, errors=dark_errors)

            master_darks = {}
            for exptime in u_lightexp:
                master_darks[exptime] = Dark_image.copy()

            return master_darks


        #find missing exposure times
        exptimes_missing = np.array([e for e in u_lightexp if e not in u_darkexp])

        #there are darks for every light frame, nothing to do
        if len(exptimes_missing) < 1:
            return master_darks

        #get the longest dark exposure time with at least 3 exposures (to remove cosmics etc.)

        longest_darkexp = -1
        for exptime in u_darkexp[::-1]:
            if len(np.where(dark_exposuretimes == exptime)[0]) >= 3 and exptime in master_darks.keys():
                longest_darkexp = exptime
                break

        # no dark has >= 3 exposures, just use the longest one
        if longest_darkexp < 0:
            longest_darkexp = u_darkexp[-1]

        #get masterdark to this exposuretime
        masterdark = master_darks[longest_darkexp]

        #rescale masterdark to missing exposure times
        for missing_exp in exptimes_missing:
            new_masterdark = masterdark.copy()
            new_masterdark.header = datashare.instrument.setExpTime(new_masterdark.header, missing_exp)
            new_masterdark.data   = new_masterdark.data   * (missing_exp / longest_darkexp)
            new_masterdark.errors = new_masterdark.errors * (missing_exp / longest_darkexp)

            master_darks[missing_exp] = new_masterdark

        return master_darks

    def saveImage(self, image, filename):
        if image.header is None:
            #no information about the image, assume Light
            frame_type = ObservationMode.LIGHT
        else:
            frame_type = datashare.instrument.getExpMode(image.header)

            if frame_type not in self.filenames.keys():
                #invalid frame type, assume again light
                frame_type = ObservationMode.LIGHT

        final_filename = os.path.join(self.directories[frame_type], filename)

        if final_filename not in self.filenames[frame_type]:
            self.filenames[frame_type].append(final_filename)

        iocomp.save_image_hdf5(filename, image)


    def MedianCombine(self, frame_type):
        if frame_type not in [ObservationMode.BIAS, ObservationMode.DARK, ObservationMode.ORDERDEF, ObservationMode.FLAT, ObservationMode.THAR]:
            raise ValueError('Invalid frame_type!')

        if frame_type == ObservationMode.THAR:
            logging.warning("Building the median of multiple ThAr exposures. This should be only used for PSF determination and never for wavelength calibration!")

        if frame_type != ObservationMode.DARK:
            datalist   = []
            errorlist  = []
            headerlist = []

            for filename in self.filenames[frame_type]:
                data, errors, header = iocomp.image_from_file_fits(filename, transpose=datashare.instrument.transpose_image, index=self.index)

                if errors is None:
                    errors = np.sqrt(np.abs(data * datashare.camera.get_gain(header)) + np.power(datashare.camera.get_RON(header), 2.)) / datashare.camera.get_gain(header)

                if len(datalist) > 0:
                    if data.shape != datalist[0].shape or errors.shape != data.shape:
                        continue

                datalist.append(data)
                errorlist.append(errors)
                headerlist.append(header)

            if len(datalist) < 1:
                return None

            datalist = np.stack(datalist)
            errorlist = np.stack(errorlist)

            #sort out exposures with wrong exposure
            median_sum = np.median(np.sum(datalist, axis=(1,2)))

            good_inds = np.asarray(np.abs(np.sum(datalist, axis=(1,2)) - median_sum)/median_sum < 0.1).nonzero()

            datalist  = datalist[good_inds]
            errorlist = errorlist[good_inds]



            median_data   = np.median(datalist, axis=0)
            #median_errors = 1.253 * np.sqrt(np.mean(np.square(errorlist), axis=0))
            #median_errors = np.sqrt(np.mean(np.square(datalist - median_data), axis=0))# + np.mean(np.square(errorlist), axis=0))
            median_errors = np.median(errorlist, axis=0) / np.sqrt(datalist.shape[0])

            median_image = Image(median_data, errors=median_errors, gain=1, RON=0, header=headerlist[0])

            return median_image

        else:
            #dark frames, do the same as above, but for each exposure time


            datalist   = []
            errorlist  = []
            headerlist = []

            for filename in self.filenames[frame_type]:
                data, errors, header = iocomp.image_from_file_fits(filename, transpose=datashare.instrument.transpose_image, index=self.index)

                if errors is None:
                    errors = np.sqrt(np.abs(data * datashare.camera.get_gain(header)) + np.power(datashare.camera.get_RON(header), 2.)) / datashare.camera.get_gain(header)

                if len(datalist) > 0:
                    if data.shape != datalist[0].shape or errors.shape != data.shape:
                        continue

                datalist.append(data)
                errorlist.append(errors)
                headerlist.append(header)

            if len(datalist) < 1:
                return None

            datalist = np.stack(datalist)
            errorlist = np.stack(errorlist)

            #get all exposure times
            exposure_times = np.array([datashare.instrument.getExptime(header) if header is not None else -1 for header in headerlist ])

            unique_exptimes = np.unique(exposure_times)
            unique_exptimes = unique_exptimes[unique_exptimes > 0]

            if len(unique_exptimes) < 1:
                return None

            master_darks = {}

            for exptime in unique_exptimes:
                inds = np.where(exposure_times == exptime)[0]

                median_data   = np.median(datalist[inds,:,:], axis=0)
                median_errors = 1.253 * np.sqrt(np.sum(np.square(errorlist[inds,:,:]), axis=0)) / len(inds)

                median_image = Image(median_data, errors=median_errors, gain=1, RON=0, header=headerlist[0])

                master_darks[exptime] = median_image

            return master_darks

    def number_of(self, mode):
        assert mode in self.filenames.keys()

        return len(self.filenames[mode])


    def getBasename(self, mode, nr):
        assert mode in self.filenames.keys()
        assert np.abs(nr) < len(self.filenames[mode])

        return os.path.splitext(os.path.basename(self.filenames[mode][nr]))[0]


    def getFile(self, mode, nr):
        assert mode in self.filenames.keys()
        assert np.abs(nr) < len(self.filenames[mode])

        filename = self.filenames[mode][nr]

        datashare.current_filename = os.path.splitext(os.path.basename(filename))[0]

        return Image.from_file(filename, index=self.index)

    def getAllFiles(self, mode):
        assert mode in self.filenames.keys()

        Files = []

        for i in range(len(self.filenames[mode])):
            Files.append(self.getFile(mode, i))

        return Files

    def removeEmptyFiles(self, masterbias, **kwargs):
        biasmedian = np.nanmedian(masterbias.data)
        biasrms    = np.sqrt(np.sum(np.square(masterbias.data - biasmedian))) / masterbias.data.size

        self._updateEmpty(ObservationMode.FLAT    , biasmedian, biasrms, **kwargs)
        self._updateEmpty(ObservationMode.ORDERDEF, biasmedian, biasrms, **kwargs)
        self._updateEmpty(ObservationMode.THAR    , biasmedian, biasrms, **kwargs)
        self._updateEmpty(ObservationMode.LIGHT   , biasmedian, biasrms, **kwargs)


    def _updateEmpty(self, mode, biasmedian, biasrms, **kwargs):
        old_filenames = self.filenames[mode]
        new_filenames = []
        new_mjds      = []

        for filename in old_filenames:
            data, errors, header = iocomp.image_from_file_fits(filename, transpose=datashare.instrument.transpose_image, index=self.index)

            if not CCD_corrections.IsImageEmpty(data, biasmedian, biasrms, **kwargs):
                new_filenames.append(filename)

                mjd = datashare.instrument.getMJD(header)

                if mjd == 0 or np.isnan(mjd):
                    new_mjds.append(0)
                else:
                    new_mjds.append(mjd)

        self.filenames[mode] = new_filenames
        self.mjds[mode]      = new_mjds


class ExtractedHandler(FileHandler):
    def __init__(self, suffix='_ext', default_extension='.hdf5'):
        super().__init__()

        self.suffix            = suffix
        self.default_extension = default_extension
        self.base_directories  = {}


    def makeBaseDirectories(self, base_dir):
        # automatically make subdirectories

        if not os.path.exists(base_dir):
            os.mkdir(base_dir)

        subdirectories = {
            ObservationMode.FLAT    : 'Flat',
            ObservationMode.ORDERDEF: 'Orderdef',
            ObservationMode.BIAS    : 'Bias',
            ObservationMode.DARK    : 'Dark',
            ObservationMode.THAR    : 'ThAr',
            ObservationMode.LIGHT   : 'Light'
            }

        for key in subdirectories.keys():
            subdir = os.path.join(base_dir, subdirectories[key])

            if not os.path.exists(subdir):
                os.mkdir(subdir)

            self.directories[key]      = subdir
            self.base_directories[key] = subdir

    def makeOwnDirectories(self, base_directories = None):
        if type(base_directories) is dict and len(base_directories) == len(self.directories):
            self.base_directories = base_directories

        if len(self.base_directories) != len(self.directories):
            if base_directories is None:
                raise ValueError('Base directories are not set yet!')
            else:
                raise ValueError('Invalid base directories!')


        own_directory_name = self.suffix.replace(' ', '_').strip('_')

        for key in self.base_directories.keys():
            subdir = os.path.join(self.base_directories[key], own_directory_name)

            if not os.path.exists(subdir):
                os.mkdir(subdir)

            self.directories[key] = subdir

    def getBaseDirectories(self):
        return self.base_directories

    def saveSpectrum(self, spectrum, filename, frame_type = None):
        if frame_type is None or frame_type not in self.filenames.keys():
            if spectrum[0].header is None:
                #no information about spectrum, assume Light
                frame_type = ObservationMode.LIGHT
            else:
                frame_type = datashare.instrument.getExpMode(spectrum[0].header)

                if frame_type not in self.filenames.keys():
                    #invalid frame type, assume again light
                    frame_type = ObservationMode.LIGHT

        final_filename = self._makeNewFilename(filename, frame_type)

        if final_filename not in self.filenames[frame_type]:
            self.filenames[frame_type].append(final_filename)

            try:
                mjd = datashare.instrument.getMJD(spectrum[0].header)

                if mjd == 0 or np.isnan(mjd):
                    self.mjds[frame_type].append(0)
                else:
                    self.mjds[frame_type].append(mjd)

            except:
                self.mjds[frame_type].append(0)


        spectrum.save(final_filename)

        return final_filename

    def exists(self,original_filename, frame_type):
        extracted_filename = self._makeNewFilename(original_filename, frame_type)

        return os.path.exists(extracted_filename)

    def logExisting(self, original_filename, frame_type):
        extracted_filename = self._makeNewFilename(original_filename, frame_type)

        if not extracted_filename in self.filenames[frame_type]:
            self.filenames[frame_type].append(extracted_filename)

            headerlist = iocomp.load_header_hdf5(extracted_filename)

            if len(headerlist) > 0:
                mjd = datashare.instrument.getMJD(headerlist[0])

                if mjd == 0 or np.isnan(mjd):
                    self.mjds[frame_type].append(0)
                else:
                    self.mjds[frame_type].append(mjd)
            else:
                 self.mjds[frame_type].append(0)


    def loadExtracted(self, original_filename, frame_type):
        extracted_filename = self._makeNewFilename(original_filename, frame_type)
        return SpectraList.load(extracted_filename)


    def _makeNewFilename(self, filename, frame_type):
        new_filename  = filename + self.suffix + self.default_extension

        return os.path.join(self.directories[frame_type], new_filename)

class ThArPeaks:
    def __init__(self):
        self._peaks = {}
        self._errs  = {}

        self.widths = []

    def addPeaks(self, order, peaks, errs=None):
        self._peaks[order] = peaks

        if errs is not None:
            self._errs[order] = errs
        else:
            self._errs[order] = np.ones_like(peaks) * np.nan

    def getPeaks(self, order):
        if order not in self._peaks.keys():
            return []

        return self._peaks[order]


    def getErrs(self, order):
        if order not in self._errs.keys():
            return []

        return self._errs[order]


    def getOrders(self):
        return list(self._peaks.keys())

    def __getitem__(self, index):
        return self.getPeaks(index)

    def allPeaks(self):
        all_peaks  = []
        all_orders = []

        for o in list(self._peaks.keys()):
            for p in self._peaks[o]:
                all_peaks.append(p)
                all_orders.append(o)

        return np.array(all_peaks), np.array(all_orders)

    def setWidths(self, widths):
        widths = np.array(widths)

        widths[np.isnan(widths)] = np.nanmedian(widths)

        self.widths = widths
    def getWidths(self):
        return self.widths

class Overlap:
    def __init__(self, red_ordnr, blue_ordnr, red_pixel, blue_pixel):
        assert np.abs(red_ordnr - blue_ordnr) == 1.

        self.red_ordnr  = red_ordnr
        self.blue_ordnr = blue_ordnr
        self.red_pixel  = red_pixel
        self.blue_pixel = blue_pixel

    def orders(self):
        return (self.red_ordnr, self.blue_ordnr)

class Overlaps:
    def __init__(self):
        self.overlaps = []
        self.orders_with_overlaps = []

        self.maxpix = None
        self.minpix = None

    def addOverlap(self, overlap):
        assert isinstance(overlap, Overlap)

        self.overlaps.append(overlap)

        red_ordnr, blue_ordnr = overlap.orders()

        if red_ordnr not in self.orders_with_overlaps:
            self.orders_with_overlaps.append(red_ordnr)
        if blue_ordnr not in self.orders_with_overlaps:
            self.orders_with_overlaps.append(blue_ordnr)

    def addOverlapList(self, overlaps):
        assert type(overlaps) is list

        for overlap in overlaps:
            self.addOverlap(overlap)

    def subsample(self, indices):
        if len(indices) < 1:
            return None

        if np.max(np.abs(indices)) >= len(self.overlaps):
            raise ValueError('Invalid indices')

        new_overlaps = [self.overlaps[int(i)] for i in indices]

        new_Overlaps = Overlaps()

        new_Overlaps.addOverlapList(new_overlaps)

        return new_Overlaps

    def fromSameOverlap(self, red_ordnr, blue_ordnr):
        overlaps = [o for o in self.overlaps if (o.red_ordnr == red_ordnr and o.blue_ordnr == blue_ordnr)]

        return overlaps

    def setPixExtrema(self, minpix, maxpix):
        self.minpix = int(minpix)
        self.maxpix = int(maxpix)

    def minOrder(self):
        return np.min(self.orders_with_overlaps).astype(int)

    def maxOrder(self):
        return np.max(self.orders_with_overlaps).astype(int)

    def allPixels(self):
        red_pixels  = [o.red_pixel  for o in self.overlaps]
        blue_pixels = [o.blue_pixel for o in self.overlaps]

        return np.array(red_pixels), np.array(blue_pixels)

    def allOrders(self):
        red_orders  = [o.red_ordnr  for o in self.overlaps]
        blue_orders = [o.blue_ordnr for o in self.overlaps]

        return np.array(red_orders), np.array(blue_orders)

    def __len__(self):
        return len(self.overlaps)

class WavelengthSolution:
    def __init__(self, m0, orderlim, pixlim):
        self.m0       = m0
        self.Overlaps = None


        self.min_order, self.max_order = orderlim
        self.min_pix  , self.max_pix = pixlim

        self.coeffs = np.nan

        self.v_shift_interpolator = None
        self.interpolator_method  = None

        self.ThAr_header = None

    def eval_wavelengths(self, ordernr, pixels, header=None):
        normed_ordernr = wavelength_calibration._normalize_coordinates(ordernr, self.min_order, self.max_order)
        normed_pixels  = wavelength_calibration._normalize_coordinates(pixels, self.min_pix, self.max_pix)

        shift = self.getVShift(header)

        return wavelength_calibration.eval_wave_coeffs(self.coeffs, self.m0, normed_ordernr, normed_pixels, ordernr) * (1 + shift/Constants.c)

    def getVShift(self, header):
        if self.v_shift_interpolator is None or header is None :
            shift = 0
        else:
            #negative velocities are blue-shifts
            if self.interpolator_method == 'temperature':
                temp = datashare.instrument.getTemperature(header)

                if temp != 0:
                    shift = float(self.v_shift_interpolator(temp))  #velocity shift in km/s
                else:
                    shift = 0

            else:
                try:
                    mjd = datashare.instrument.getMJD(header)

                    if mjd != 0 or not np.isnan(mjd):
                        shift = float(self.v_shift_interpolator(mjd))  #velocity shift in km/s
                    else:
                        shift = 0
                except:
                    shift = 0

        return shift


    def copy(self):
        return copy.deepcopy(self)

    def apply_scale(self, scale):
        self.coeffs *= scale

class WavelengthSolution_Spline:
    def __init__(self, m0, orderlim, pixlim):
        self.m0       = m0
        self.Overlaps = None


        self.spline = None

        self.min_order, self.max_order = orderlim
        self.min_pix  , self.max_pix = pixlim

        self.pix_shift = 0

        self.v_shift_interpolator = None
        self.interpolator_method  = None

        self.ThAr_header = None

    def eval_wavelengths(self, ordernr, pixels, header=None):
        shift = self.getVShift(header)

        assert np.all(self.min_order <= ordernr) and np.all(ordernr <= self.max_order)
        assert np.all(self.min_pix   <= pixels - self.pix_shift) and np.all(pixels - self.pix_shift <= self.max_pix)

        if type(self.spline) is list:
            return self.spline[ordernr - self.min_order](pixels - self.pix_shift) * (1 + shift/Constants.c)
        else:
            return self.spline(np.ones_like(pixels) * ordenr, pixels - self.pix_shift) * (1 + shift/Constants.c)

    def getVShift(self, header):
        if self.v_shift_interpolator is None or header is None:
            shift = 0
        else:
            #negative velocities are blue-shifts
            if self.interpolator_method == 'temperature':
                temp = datashare.instrument.getTemperature(header)

                if temp != 0:
                    shift = float(self.v_shift_interpolator(temp))  #velocity shift in km/s
                else:
                    shift = 0
            else:
                try:
                    mjd = datashare.instrument.getMJD(header)

                    if mjd != 0 or not np.isnan(mjd):
                        shift = float(self.v_shift_interpolator(mjd))  #velocity shift in km/s
                    else:
                        shift = 0
                except:
                    shift = 0

        return shift


    def copy(self):
        return copy.deepcopy(self)

class OverlapWavelengthSolution(WavelengthSolution):
    def __init__(self, m0, Overlaps, pixlim):
        orderlim = (Overlaps.minOrder(), Overlaps.maxOrder())

        super().__init__(m0, orderlim, pixlim)
        self.nord = datashare.instrument.nord_overlaps
        self.npix = datashare.instrument.npix_overlaps

        self.Overlaps = Overlaps

        self.coeffs, self.rms, self.used_Overlaps, self.not_used_Overlaps = \
            wavelength_calibration.coeffs_from_overlaps(self.m0, self.Overlaps, self.min_pix, self.max_pix, self.min_order, self.max_order, self.nord, self.npix)

class FinalWavelengthSolution(WavelengthSolution):
    def __init__(self, coeffs, m0, lineatlas, found_lines, orderlim, useful_orderlim, pixlim, allPeaks, rms, mjd):
        super().__init__(m0, orderlim, pixlim)
        self.nord = coeffs.shape[0] -1
        self.npix = coeffs.shape[1] -1

        self.coeffs = coeffs

        self.min_useful_order, self.max_useful_order = useful_orderlim

        self.allPeaks      = allPeaks

        self.lines_matched = found_lines    #list of MatchedLine objects
        self.lineatlas     = lineatlas
        self.rms           = rms            #rms in km/s
        self.mjd           = mjd            #MJD of ThAr exposure

        self.final_rms     = None           # will be set after comparing all different ThAr frames



    def save(self, filename):
        iocomp.save_final_wavesolution(filename, self)

    def from_file(filename):
        return iocomp.load_final_wavesolution(filename)


    def apply_pixel_shift(self, shift):
        #shift solutions in pixels
        new_solution = copy.deepcopy(self)

        new_solution.min_pix += shift
        new_solution.max_pix += shift

        return new_solution

    def apply_RV_shift(self, shift):
        #TODO!!!
        pass

    def create_shifts_interpolator(self, mjds, shifts, method='median', temps=None, rms=None, lam=None):
        """
        # Create a shift interpolator to interpolate the radial velocity drift of the wavelength solutions over the night.
        # Does not return anything, as the interpolator is stored within this object
        #
        # Methods:
        #   median: just use median of all shifts as a constant shift
        #   linear: interpolate shifts with linear splines using the MJDs
        #   cubic: interpolate shifts with cubic splines using the MJDs (default)
        #   smoothed: use a smooth spline to interpolate between the shifts using the MJDs. This method will NOT exactly go through the reference points (the measured wavelength solutions)
        #   temperature: try to fit a linear correlation between temperature and shift. Might not be very stable, still in testing! This is not a spline interpolation, but a linear fit.
        #
        # :param mjds: list of float, list with MJDs of all wavelength solutions
        # :param shifts: list of float, list with radial velocity shifts of all wavelength solutions relative to this solution
        # :param method: string, method how to interpolate. Must be one of ['median', 'linear', 'cubic', 'smoothed', 'temperature'] (default 'median')
        # :param temps: list of floats or None, instrument temperatures for all wavelength solutions. Only used of method=='temperature' (default None)
        # :param rms: list of floats or None, radial velocity RMS of all wavelength solutions, only used if method == 'smoothed' (default None)
        # :param lam: float or None, smoothing parameter, only used if method == 'smoothed'. Greater lam leeds to a smoother fit (default None)
        #
        # :return None
        """


        if method == 'median':
            median_shift = np.nanmedian(shifts)

            #always return median_shift
            self.v_shift_interpolator = lambda mjd: median_shift * np.ones_like(mjd)

        elif method == 'linear':
            if mjds is None:
                raise ValueError('MJDs must be given when using method \'linear\'!')

            self.v_shift_interpolator = interpolate.make_interp_spline(mjds, shifts, k=1)

        elif method == 'cubic':
            if mjds is None:
                raise ValueError('MJDs must be given when using method \'cubic\'!')

            self.v_shift_interpolator = interpolate.make_interp_spline(mjds, shifts, k=np.min((3, len(mjds) - 1)))

        elif method == 'smoothed':
            if rms is None:
                weights = np.ones_like(shifts)
            else:
                weights = np.power(rms, -2.)        # weight = 1/error^2

            if lam is None:
                lam = 0.005  #empirical value

            self.v_shift_interpolator = interpolate.make_smoothing_spline(mjds, shifts, w=weights, lam=lam)

        elif method == 'temperature':
            if temps is None:
                raise ValueError('temperatures must be given when using method \'temperature\'!')

            #fit linear model
            #guess parameters, a, b
            p0 = (0, np.median(shifts))

            #fit
            popt, _ = optimize.curve_fit(FitFunctions.linear, temps, shifts, p0=p0)

            #b + a*x - median
            self.v_shift_interpolator = lambda temp: popt[1] + popt[0] * temp - median

        else:
            raise ValueError('Invalid method!')

        self.interpolator_method = method


    def getPeakWidths(self, ordernr):
        return self.allPeaks.getWidths()[ordernr]

class FinalWavelengthSolution_Spline1D(WavelengthSolution_Spline):
    def __init__(self, spline, m0, lineatlas, found_lines, orderlim, useful_orderlim, pixlim, allPeaks, rms, mjd):
        super().__init__(m0, orderlim, pixlim)

        self.spline = spline

        self.min_useful_order, self.max_useful_order = useful_orderlim

        self.allPeaks      = allPeaks

        self.lines_matched = found_lines    #list of MatchedLine objects
        self.lineatlas     = lineatlas
        self.rms           = rms            #rms in km/s
        self.mjd           = mjd            #MJD of ThAr exposure

        self.pix_shift     =  0             # pixel shift to first calculation

        self.final_rms     = None           # will be set after comparing all different ThAr frames



    def save(self, filename):
        #TODO!!!
        pass

    def from_file(filename):
        #TODO!!!
        pass

    def apply_pixel_shift(self, shift):
        #shift solutions in pixels
        new_solution = copy.deepcopy(self)

        new_solution.pix_shift += shift

        return new_solution

    def apply_RV_shift(self, shift):
        #TODO!!!
        pass

    def create_shifts_interpolator(self, mjds, shifts, method='median', temps=None, rms=None, lam=None):
        if method == 'median':
            median_shift = np.nanmedian(shifts)

            #always return median_shift
            self.v_shift_interpolator = lambda mjd: median_shift * np.ones_like(mjd)

        elif method == 'linear':
            if mjds is None:
                raise ValueError('MJDs must be given when using method \'linear\'!')

            self.v_shift_interpolator = interpolate.make_interp_spline(mjds, shifts, k=1)

        elif method == 'cubic':
            if mjds is None:
                raise ValueError('MJDs must be given when using method \'cubic\'!')

            self.v_shift_interpolator = interpolate.make_interp_spline(mjds, shifts, k=3)

        elif method == 'smoothed':
            if rms is None:
                weights = np.ones_like(shifts)
            else:
                weights = np.power(rms, -2.)        # weight = 1/error^2

            if lam is None:
                lam = 0.005  #empirical value

            self.v_shift_interpolator = interpolate.make_smoothing_spline(mjds, shifts, w=weights, lam=lam)

        elif method == 'temperature':
            if temps is None:
                raise ValueError('temperatures must be given when using method \'temperature\'!')

            #fit linear model
            #guess parameters, a, b
            p0 = (0, np.median(shifts))

            #fit
            popt, _ = optimize.curve_fit(FitFunctions.linear, temps, shifts, p0=p0)

            #b + a*x - median
            self.v_shift_interpolator = lambda temp: popt[1] + popt[0] * temp - median

        else:
            raise ValueError('Invalid method!')

        self.interpolator_method = method


    def getPeakWidths(self, ordernr):
        return self.allPeaks.getWidths()[ordernr]


class MatchedLine:
    def __init__(self, order, pix, wavelength):
        self.order      = order
        self.pixel      = pix
        self.wavelength = wavelength

class PSF_Plotter:
    ##############
    # Stores the information of the PSF in one order
    # Can plot the 2D-Image of one spectral line as it would look like on the CCD
    ##############

    def __init__(self, function, params=None, is_const=True):
        assert callable(function)


        self.function = function
        self.params   = params
        self.is_const = is_const    #whether the PSF is constant along the order or not

    def plot(self, x):
        if isinstance(self.params, dict):
            return self.function(x, **self.params)
        elif isinstance(self.params, list):
            return self.function(x, *self.params)
        else:
            return self.function(x)

class Tilt_PSF_Plotter(PSF_Plotter):
    #############
    # simple class, just assuming a tilted line as the PSF
    ############

    def __init__(self, tilt, orders_sigma = 3):
        self.tilt = tilt
        self.orders_sigma = orders_sigma

    def _createLine(self, x):
        ndots = 100

        #if tilt is constant, we can calculate the line earlier
        if not callable(self.tilt):
            tilt = self.tilt
        else:
            tilt = self.tilt(x)

        #get relevant pixels

        window_width_x = np.min(1, np.ceil(3* self.orders_sigma * np.abs(np.sin(tilt)))).astype(int)
        window_width_y = np.min(1, np.ceil(3* self.orders_sigma * np.abs(np.cos(tilt)))).astype(int)

        line_weights = np.zeros(shape=(window_width_y, window_width_x))

        if tilt == 0:
            line_weights[0, :] = _gauss(np.arange(line_weights.shape[1], line_weights.shape[1]/2., self.order_sigma))

        else:
             #x and y coordinates of the line
             mid_x = (0.5 + window_width_x//2 )   #set middle x to the mid of the pixel (+0.5)
             mid_y = (0.5 + window_width_y//2)
             fac_x = np.sin(mean_tilt) * window_width_x/ndots
             fac_y = np.cos(mean_tilt) * window_width_y/ndots


             line_x = lambda t: mid_x + t * fac_x
             line_y = lambda t: mid_y + t * fac_y

             #line parameter
             dots = np.arange(ndots) - ndots/2.

             xs, ys = np.array(line_x(dots)), np.array(line_y(dots))

             valid_inds = np.where(valid_index(xs, ys, line_weights.shape[1]-1, line_weights.shape[0]-1))[0]

             xs = xs[valid_inds]
             ys = ys[valid_inds]

             for (x_l,y_l) in zip(xs,ys):
                     line_weights[int(y_l),int(x_l)] += 1

            # apply exponential decrease
            # TODO!!!!!


class ExtractionWeights:
    ##############
    # Class to store weights for weighted extraction algorithm
    #
    #
    ##############
    def __init__(self, nr_of_fibers, nr_of_orders, nr_of_pixels):
        self.weights_array     = np.zeros(shape=(nr_of_fibers, nr_of_orders, nr_of_pixels), dtype=object) * np.nan
        self.boundaries_array  = np.zeros(shape=(nr_of_fibers, nr_of_orders, nr_of_pixels, 4), dtype=int)
        self.ordershapes_list  = [[None for _ in range(nr_of_orders)] for _ in range(nr_of_fibers)]       #list instead of array as length of ordershape may vary

        self.nr_of_fibers = nr_of_fibers
        self.nr_of_orders = nr_of_orders
        self.nr_of_pixels = nr_of_pixels



    def set_weights(self, fiber, order, x, weights):
        assert isinstance(weights, np.ndarray)
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)
        assert (x     >= 0 and x     < self.nr_of_pixels)

        self.weights_array[fiber, order, x] = weights

    def set_weights_order(self, fiber, order, weights):
        assert isinstance(weights, np.ndarray)
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)
        assert weights.shape[0] == self.nr_of_pixels

        self.weights_array[fiber, order, :] = weights

    def get_weights(self, fiber, order, x):
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)
        assert (x     >= 0 and x     < self.nr_of_pixels)

        if np.all(self.weights_array[fiber, order, x] is np.nan):
            raise ValueError('weights for fiber {}, order {}, pixel {} not set yet!'.format(fiber, order, x))
        else:
            return self.weights_array[fiber, order, x]

    def get_weights_order(self, fiber, order):
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)

        if np.all(self.weights_array[fiber, order, :] is np.nan):
            raise ValueError('weights for fiber {}, order {} not set yet!'.format(fiber, order))
        else:
            return self.weights_array[fiber, order, :]

    def set_boundaries(self, fiber, order, x, boundaries):
        #boundaries = (min_x, min_y, max_x, max_y)

        assert len(boundaries) == 4
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)
        assert (x     >= 0 and x     < self.nr_of_pixels)

        #ensure boundaries are ints
        self.boundaries_array[fiber, order, x, :] = boundaries

    def set_boundaries_order(self, fiber, order, boundaries):
        #boundaries = (min_x, min_y, max_x, max_y)

        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)
        assert boundaries.shape == (self.nr_of_pixels, 4)

        #ensure boundaries are ints
        self.boundaries_array[fiber, order, :, :] = boundaries


    def get_boundaries(self, fiber, order, x):
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)
        assert (x     >= 0 and x     < self.nr_of_pixels)

        if np.all(self.boundaries_array[fiber, order, x] is np.nan):
            raise ValueError('boundaries for fiber {}, order {}, pixel {} not set yet!'.format(fiber, order, x))
        else:
            return tuple(self.boundaries_array[fiber, order, x, :])

    def get_boundaries_order(self, fiber, order):
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)

        if np.all(self.boundaries_array[fiber, order, :, :] is np.nan):
            raise ValueError('boundaries for fiber {}, order {} not set yet!'.format(fiber, order))
        else:
            return self.boundaries_array[fiber, order, :, :]


    def set_ordershape(self, fiber, order, ordershape):
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)

        self.ordershapes_list[fiber][order] = ordershape

    def get_ordershape(self, fiber, order):
        assert (fiber >= 0 and fiber < self.nr_of_fibers)
        assert (order >= 0 and order < self.nr_of_orders)

        return self.ordershapes_list[fiber][order]

    def save(self, filename):
        iocomp.save_weights(filename, self)

    def load(filename):
        return iocomp.load_weights(filename)

class RawSpectralOrder(GeneralOrder):
    def __init__(self, pixels, flux, errors=None, ordernr=0):
        super().__init__(ordernr)

        #pixels is just an int array, no list of "Pixel" objects!
        self.pixels = pixels
        self.flux   = flux
        self.errors = errors

        if self.errors is None:
            self.errors = np.zeros_like(self.flux)

        if not len(self.pixels) == len(self.flux) == len(self.errors):
            raise ValueError('Invalid shapes!')


    def reverse_pixels(self):
        self.flux = self.flux[::-1]

    def good_values(self, threshold = 5):
        return np.where(self.flux/self.errors > threshold)[0]

    def cut_to_good_values(self, threshold = 5):
        good_values = self.good_values(threshold)

        if len(good_values) < 2:
            return

        self.pixels = self.pixels[np.min(good_values):np.max(good_values)]
        self.flux   = self.flux[np.min(good_values):np.max(good_values)]
        self.errors = self.errors[np.min(good_values):np.max(good_values)]

        #remove remaining bad values
        self.flux[np.where(self.flux/self.errors < threshold)] = 0

    def applyWaveSolution(self, WaveSolution, ordernr, v_shift = 0, header=None):
        if (header is not None and datashare.instrument is not None) and v_shift == 0:
            wavelengths = WaveSolution.eval_wavelengths(ordernr, self.pixels, header=header)
        else:
            wavelengths = WaveSolution.eval_wavelengths(ordernr, self.pixels)
            wavelengths *= (1 + v_shift / Constants.c)      #apply shift


        return SpectralOrder(wavelengths, self.flux.copy(), errors=self.errors.copy(), ordernr=self.ordernr)

    def __add__(self, other):
        assert isinstance(other, type(self))


        #easy case
        if np.array_equal(self.pixels, other.pixels):
            new_flux = self.flux + other.flux
            new_errs = np.sqrt(np.power(other.errors, 2.) + np.power(self.errors, 2.))

            return type(self)(self.pixels, new_flux, errors=new_errs)
        else:
            if (np.min(self.pixels) > np.max(other.pixels) or np.max(self.pixels) < np.min(other.pixels)):
                raise ValueError('Pixel arrays do not match!')

            min_pix = np.max((np.min(self.pixels), np.min(other.pixels)))
            max_pix = np.min((np.max(self.pixels), np.max(other.pixels)))

            new_pixels = np.arange(min_pix, max_pix)

            self_flux_interp  = interpolate.interp1d(self.pixels, self.flux, kind='linear')
            self_errs_interp  = interpolate.interp1d(self.pixels, self.errors, kind='linear')
            other_flux_interp = interpolate.interp1d(other.pixels, other.flux, kind='linear')
            other_errs_interp = interpolate.interp1d(other.pixels, other.errors, kind='linear')

            new_flux = self_flux_interp(new_pixels) + other_flux_interp(new_pixels)
            new_errs = np.sqrt(np.power(self_errs_interp(new_pixels), 2.) + np.power(other_errs_interp(new_pixels), 2.))

            return type(self)(new_pixels, new_flux, errors=new_errs)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        assert isinstance(other, type(self))

        #easy case
        if np.array_equal(self.pixels, other.pixels):
            new_flux = self.flux * other.flux
            new_errs = np.sqrt(np.power(self.flux * other.errors, 2.) + np.power(other.flux * self.errors, 2.))

            return type(self)(self.pixels, new_flux, errors=new_errs)
        else:
            if (np.min(self.pixels) > np.max(other.pixels) or np.max(self.pixels) < np.min(other.pixels)):
                raise ValueError('Pixel arrays do not match!')

            min_pix = np.max((np.min(self.pixels), np.min(other.pixels)))
            max_pix = np.min((np.max(self.pixels), np.max(other.pixels)))

            new_pixels = np.arange(min_pix, max_pix+1)

            self_flux_interp  = interpolate.interp1d(self.pixels, self.flux, kind='linear')
            self_errs_interp  = interpolate.interp1d(self.pixels, self.errors, kind='linear')
            other_flux_interp = interpolate.interp1d(other.pixels, other.flux, kind='linear')
            other_errs_interp = interpolate.interp1d(other.pixels, other.errors, kind='linear')

            new_flux_self  = self_flux_interp(new_pixels)
            new_flux_other = other_flux_interp(new_pixels)
            new_flux       = new_flux_self * new_flux_other
            new_errs       = np.sqrt(np.power(new_flux_other * self_errs_interp(new_pixels), 2.) + np.power(new_flux_self * other_errs_interp(new_pixels), 2.))

            return type(self)(new_pixels, new_flux, errors=new_errs)

    def __truediv__(self, other):
        return self * (~other)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other


    def __neg__(self):
        return type(self)(self.pixels, -self.flux, errors=self.errors)

    def __invert__(self):
        new_flux = np.power(self.flux, -1)
        new_flux[self.flux <= 0] = 0
        new_errs = np.abs(self.errors * np.power(self.flux, -2.))

        return type(self)(self.pixels, new_flux, errors=new_errs)

class SpectralOrder(GeneralOrder):
    def __init__(self, wave, flux, errors=None, ordernr=0):
        super().__init__(ordernr)

        self.wave   = wave
        self.flux   = flux
        self.errors = errors

        if self.errors is None:
            self.errors = np.zeros_like(self.flux)

        if not len(self.wave) == len(self.flux) == len(self.errors):
            raise ValueError('Invalid shapes!')

    def reverse_pixels(self):
        self.flux = self.flux[::-1]

    def good_values(self, threshold = 5):
        return np.where(self.flux/self.errors > threshold)[0]

    def cut_to_good_values(self, threshold = 5):
        good_values = self.good_values(threshold)

        if len(good_values) < 2:
            return

        self.wave   = self.wave[np.min(good_values):np.max(good_values)]
        self.flux   = self.flux[np.min(good_values):np.max(good_values)]
        self.errors = self.errors[np.min(good_values):np.max(good_values)]

    def loglinear_wavelengths(self, oversampling=2, **kwargs):
        min_wav = np.min(self.wave)
        max_wav = np.max(self.wave)

        min_log_wav = np.log10(min_wav)
        max_log_wav = np.log10(max_wav)

        npoints = int(oversampling * len(self.wave))

        new_log_wavs = np.linspace(min_log_wav, max_log_wav, num=npoints)

        new_wave = np.power(10., new_log_wavs)

        new_wave = new_wave[(new_wave >= min_wav) & (new_wave <= max_wav)]  #filter for calculation errors

        """
        flux_interpolator  = interpolate.interp1d(self.wave, self.flux, bounds_error = False, **kwargs)
        error_interpolator = interpolate.interp1d(self.wave, self.errors, bounds_error = False, **kwargs)
        new_flux           = flux_interpolator(new_wave)
        new_errs           = error_interpolator(new_wave)

        return type(self)(new_wave, new_flux, errors=new_errs)
        """

        return self.interpolate_to_new_wavs(new_wave, **kwargs)


    def interpolate_to_new_wavs(self, new_wave, **kwargs):
        min_wav     = np.min(self.wave)
        max_wav     = np.max(self.wave)
        new_min_wav = np.min(new_wave)
        new_max_wav = np.max(new_wave)

        if new_min_wav < min_wav or new_max_wav > max_wav:
            raise ValueError('Invalid bounds! Bounds exceed wavelength range!')


        #check, if we need to interpolate (increase resolution) or build averages (reduce resolution)
        #reduce wavelength range to new wavelengths
        temp_order = self.return_order_between(new_wave[0], new_wave[1])

        npoints_new = len(new_wave)
        npoints_old = len(temp_order.wave)

        flux_interpolator  = interpolate.interp1d(self.wave, self.flux, bounds_error = False, **kwargs)
        error_interpolator = interpolate.interp1d(self.wave, self.errors, bounds_error = False, **kwargs)


        if npoints_new > npoints_old:   #interpolate, increase resolution
            new_flux   = flux_interpolator(new_wave)
            new_errs   = error_interpolator(new_wave)
        else:   #build averages, reduce resolution
            new_flux = np.zeros(npoints_new)
            new_errs = np.zeros(npoints_new)

            #build wave boundaries for averaging
            wave_boundaries = new_wave + 0.5 * np.diff(new_wave)
            wave_boundaries = np.insert(wave_boundaries, 0, 2 * new_wave[0] - new_wave[1])     #new_wave[0] - np.diff(new_wave)[0]
            wave_boundaries = np.insert(wave_boundaries, -1, 2 * new_wave[-1] - new_wave[-2])

            for pix_ind in range(npoints_new):
                left_boundary  = wave_boundaries[pix_ind]
                right_boundary = wave_boundaries[pix_ind+1]

                left_ind  = np.searchsorted(self.wave, left_boundary, side='left')
                right_ind = np.searchsorted(self.wave, right_boundary, side='right')

                if right_ind > left_ind:    #normal case
                    weights = np.power(self.errors[left_ind:right_ind], -2.)

                    new_flux[pix_ind] = np.average(self.flux[left_ind, right_ind], weights=weights)
                    new_errs[pix_ind] = 1. / np.sqrt(np.sum(weights))

                else: #no data between wave_boundaries
                    new_flux[pix_ind]   = flux_interpolator(new_wave[pix_ind])
                    new_errs[pix_ind]   = error_interpolator(new_wave[pix_ind])


        return type(self)(new_wave, new_flux, errors=new_errs)

    def return_between(self, min_wav, max_wav):
        inds = np.where((self.wave >= min_wav) & (self.wave <= max_wav))[0]

        return self.flux[inds]

    def return_order_between(self, min_wav, max_wav):
        inds = np.where((self.wave >= min_wav) & (self.wave <= max_wav))[0]

        return type(self)(self.wave[inds], self.flux[inds], errors=self.errors[inds])


    def square_flux(self):
        new_flux = np.power(self.flux, 2)
        new_errs = np.abs(2 * self.flux * self.errors)

        return type(self)(self.wave, new_flux, errors=new_errs)

    def contnorm(self, Continuum):
        min_wave = np.min(self.wave)
        max_wave = np.max(self.wave)
        #continuum does not cover whole spectral range, do nothing
        if np.min(Continuum.wave) > min_wave or np.max(Continuum.wave) < max_wave:
            return self.copy()

        cont_interpolator = interpolate.CubicSpline(Continuum.wave, Continuum.flux)

        new_flux = self.flux   / cont_interpolator(self.wave)
        new_errs = self.errors / cont_interpolator(self.wave)
        new_wave = self.wave

        return SpectralOrder(new_wave, new_flux, errors=new_errs, ordernr=self.ordernr)

    def filter_outliners(self, **kwargs):
        new_flux, new_errors = spectrum_operations.filter_outliners(self.flux, errors=self.errors, **kwargs)

        return SpectralOrder(self.wave, new_flux, errors=new_errors, ordernr=self.ordernr)

    def plot_data(self):
        return self.wave, self.flux

    def applyShift(self, v_shift):
        #self.wave *= (1 + v_shift / Constants.c)                                  # not relativistic
        self.wave *= np.sqrt((Constants.c + v_shift)/(Constants.c - v_shift))      # relativistic

    def ToVacuum(self):
        self.wave = wavelength_calibration.ToVacuum(self.wave)

    def ToAir(self):
        self.wave = wavelength_calibration.ToAir(self.wave)

    def append(self, order):
        assert isinstance(order, SpectralOrder)

        if (np.max(order.wave) >= np.min(self.wave) or np.min(order.wave) <= np.max(self.wave)):
            raise ValueError('orders should not overlap!')

        if np.max(order.wave) < np.min(self.wave):
            self.flux   = np.append(order.flux  , self.flux)
            self.wave   = np.append(order.wave  , self.wave)
            self.errors = np.append(order.errors, self.errors)

        else:
            self.flux   = np.append(self.flux  , order.flux)
            self.wave   = np.append(self.wave  , order.wave)
            self.errors = np.append(self.errors, order.errors)

        sort_inds   = np.argsort(self.wave)
        self.wave   = self.wave[sort_inds]
        self.flux   = self.flux[sort_inds]
        self.errors = self.errors[sort_inds]


    def __add__(self, other):
        assert isinstance(other, type(self))

        #easy case
        if np.array_equal(self.wave, other.wave):
            new_flux = self.flux * other.flux
            new_errs = np.sqrt(np.power(self.flux * other.errors, 2.) + np.power(other.flux * self.errors, 2.))

            return type(self)(self.wave, new_flux, errors=new_errs)
        else:
            if (np.min(self.pixels) > np.max(other.pixels) or np.max(self.pixels) < np.min(other.pixels)):
                raise ValueError('Wavelength arrays do not match!')

            min_wav = np.max((np.min(self.wave), np.min(other.wave)))
            max_wav = np.min((np.max(self.wave), np.max(other.wave)))

            new_wave = np.linspace(min_wav, max_wav, np.max((len(self.wave), len(other.wave))))

            self_flux_interp  = interpolate.interp1d(self.wave, self.flux, kind='linear')
            self_errs_interp  = interpolate.interp1d(self.wave, self.errors, kind='linear')
            other_flux_interp = interpolate.interp1d(other.wave, other.flux, kind='linear')
            other_errs_interp = interpolate.interp1d(other.wave, other.errors, kind='linear')

            new_flux = self_flux_interp(new_wave) + other_flux_interp(new_wave)
            new_errs = np.sqrt(np.power(self_errs_interp(new_wave), 2.) + np.power(other_errs_interp(new_wave), 2.))

            return type(self)(new_wave, new_flux, errors=new_errs)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        assert isinstance(other, type(self))

        #easy case
        if np.array_equal(self.wave, other.wave):
            new_flux = self.flux * other.flux
            new_errs = np.sqrt(np.power(self.flux * other.errors, 2.) + np.power(other.flux * self.errors, 2.))

            return type(self)(self.wave, new_flux, errors=new_errs)
        else:
            if (np.min(self.wave) > np.max(other.wave) or np.max(self.wave) < np.min(other.wave)):
                raise ValueError('Wavelength arrays do not match!')

            min_wav = np.max((np.min(self.wave), np.min(other.wave)))
            max_wav = np.min((np.max(self.wave), np.max(other.wave)))

            new_wave = np.linspace(min_wav, max_wav, np.max((len(self.wave), len(other.wave))))

            self_flux_interp  = interpolate.interp1d(self.wave, self.flux, kind='linear')
            self_errs_interp  = interpolate.interp1d(self.wave, self.errors, kind='linear')
            other_flux_interp = interpolate.interp1d(other.wave, other.flux, kind='linear')
            other_errs_interp = interpolate.interp1d(other.wave, other.errors, kind='linear')

            new_flux_self  = self_flux_interp(new_wave)
            new_flux_other = other_flux_interp(new_wave)
            new_flux       = new_flux_self * new_flux_other
            new_errs       = np.sqrt(np.power(new_flux_other * self_errs_interp(new_wave), 2.) + np.power(new_flux_self * other_errs_interp(new_wave), 2.))

            return type(self)(new_wave, new_flux, errors=new_errs)

    def __truediv__(self, other):
        return self * (~other)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other


    def __neg__(self):
        return type(self)(self.wave, -self.flux, errors=self.errors)

    def __invert__(self):
        new_flux = np.power(self.flux, -1)
        new_flux[self.flux <= 0] = 0

        new_errs = np.abs(self.errors * np.power(self.flux, -2.))

        return type(self)(self.wave, new_flux, errors=new_errs)



class RawSpectrum(GeneralSpectrum):
    def __init__(self):
        super().__init__()

        self._order_type = RawSpectralOrder

    def getPixBounds(self):
        max_list = []
        min_list = []

        for o in self._orders:
            max_list.append(np.nanmax(o.pixels))
            min_list.append(np.nanmin(o.pixels))

        return np.nanmin(min_list), np.nanmax(max_list)

    def applyWaveSolution(self, WaveSolution, v_shift=0):
        WaveSpectrum = Spectrum()

        for ordernr, order in enumerate(self._orders):
            waveOrder = order.applyWaveSolution(WaveSolution, ordernr, header = self.header, v_shift=v_shift)
            WaveSpectrum.addOrder(waveOrder)

        WaveSpectrum.header = self.header

        if WaveSpectrum.header is not None:
            WaveSpectrum.header['HIERARCH WAVESOLUTION MJD']       = WaveSolution.mjd
            WaveSpectrum.header['HIERARCH WAVESOLUTION RMS']       = WaveSolution.rms
            WaveSpectrum.header['HIERARCH WAVESOLUTION TOTAL RMS'] = WaveSolution.final_rms if WaveSolution.final_rms is not None else -1
            WaveSpectrum.header['HIERARCH WAVESOLUTION VSHIFT']    = v_shift if v_shift != 0 else WaveSolution.getVShift(self.header)

            WaveSpectrum.header['HIERARCH WAVESOLUTION MIN USEFUL WAVE'] = np.min((np.min(WaveSpectrum[WaveSolution.min_useful_order].wave), np.min(WaveSpectrum[WaveSolution.max_useful_order].wave)))
            WaveSpectrum.header['HIERARCH WAVESOLUTION MAX USEFUL WAVE'] = np.max((np.max(WaveSpectrum[WaveSolution.min_useful_order].wave), np.max(WaveSpectrum[WaveSolution.max_useful_order].wave)))


        return WaveSpectrum


class Spectrum(GeneralSpectrum):
    def __init__(self):
        super().__init__()

        self._order_type = SpectralOrder

        #reference frame, air or vacuum
        self.reference   = 'AIR'

    def applyShift(self, v_shift):
        for order in self._orders:
            order.applyShift(v_shift)

    def loglinear_wave(self, oversampling=2, **kwargs):
        new_Spectrum = Spectrum()

        for order in self._orders:
            new_order = order.loglinear_wavelengths(oversampling=oversampling, **kwargs)
            new_Spectrum.addOrder(new_order)

        new_Spectrum.header = self.header

        if new_Spectrum.header is not None:
            new_Spectrum.header['HIERARCH WAVELENGTH OVERSAMPLING'] = oversampling

        return new_Spectrum


    def ToVacuum(self):
        if self.reference == 'AIR':
            for i in range(len(self._orders)):
               self._orders[i].ToVacuum()

            self.reference = 'VACUUM'


    def ToAir(self):
        if self.reference == 'VACUUM':
            for i in range(len(self._orders)):
                self._orders[i].ToVacuum()

            self.reference = 'AIR'

    def mergeOrders(self, **kwargs):
        fluxes = []
        wavs   = []
        errors = []

        for order in self._orders:
            fluxes.append(order.flux)
            wavs.append(order.wave)
            errors.append(order.errors)

        fluxes = np.array(fluxes)
        wavs   = np.array(wavs)
        errors = np.array(errors)

        #sort by wavelength, wavelengths should increase

        median_wavs = [np.nanmedian(wave) for wave in wavs]
        sort_inds   = np.argsort(median_wavs)

        fluxes = fluxes[sort_inds]
        wavs   = wavs[sort_inds]
        errors = errors[sort_inds]

        new_fluxes, new_wavs, new_errors = spectrum_operations._combine_orders(fluxes, wavs, errors=errors, **kwargs)

        merged_order    = SpectralOrder(new_wavs, new_fluxes, errors=new_errors, ordernr=0)
        merged_spectrum = Spectrum()
        merged_spectrum.addOrder(merged_order)

        merged_spectrum.header = self.header

        return merged_spectrum

    def contnorm_orders(self, Continuum):
        new_Spectrum = Spectrum()

        for order in self._orders:
            new_order = order.contnorm(Continuum[0])
            new_Spectrum.addOrder(new_order)

        new_Spectrum.header = self.header

        return new_Spectrum

    def filter_outliners(self, **kwargs):
        new_Spectrum = Spectrum()

        for order in self._orders:
            new_order = order.filter_outliners(**kwargs)
            new_Spectrum.addOrder(new_order)

        new_Spectrum.header = self.header

        return new_Spectrum


class SpectraList:
    ################
    # A class to store many spectra, e.g. the spectra from two fibers of an eshelle spectrograph
    ################

    def __init__(self, spectra = None):
        if spectra is None:
            spectra = []

        assert type(spectra) is list

        self._Spectra = spectra

    def addSpectrum(self, spectrum):
        assert isinstance(spectrum, GeneralSpectrum)
        self._Spectra.append(spectrum)

    def remove_spectrum(self, index):
        if np.abs(index) >= len(self._Spectra):
            raise ValueError('index out of bounds!')

        del self._Spectra[index]

    def reverse_orders(self):
        for spec in self._Spectra:
            spec.reverse_orders()

    def reverse_pixels(self):
        for spec in self._Spectra:
            spec.reverse_pixels()

    def __getitem__(self, index):
        if np.abs(index) >= len(self._Spectra):
            raise ValueError('index out of bounds!')

        return self._Spectra[index]

    def __setitem__(self, index, value):
        assert isinstance(spectrum, GeneralSpectrum)

        if np.abs(index) >= len(self._Spectra):
            raise ValueError('index out of bounds!')

        self._Spectra[index] = value

    def save(self, filename, save_type='hdf5'):
        if save_type == 'hdf5':
            iocomp.save_spectra_hdf5(filename, self)
        elif save_type == 'fits':
            iocomp.save_spectra_fits(filename, self)
            pass
        else:
            raise ValueError('invalid save_type, must be \'hdf5\' or \'fits\'!')

    def load(filename):
        if filename[-5:] == '.hdf5':
            return iocomp.load_spectra_hdf5(filename)
        elif filename[-5:] == '.fits':
            return iocomp.load_spectra_fits(filename)
        else:
            raise ValueError('invalid filename, must end with .hdf5 or .fits!')

    def copy(self):
        return copy.deepcopy(self)

    def nr_of_spectra(self):
        return len(self._Spectra)

    def cut_to_good_values(self, threshold = 5):
        for spec in self._Spectra:
            spec.cut_to_good_values(threshold=threshold)

    def applyWaveSolution(self, WaveSolutions, v_shift=0):
        if isinstance(WaveSolutions, WavelengthSolution) or istinstance(WaveSolutions, WavelengthSolution_Spline):
            WaveSolutions = [WaveSolutions] * self.nr_of_spectra()
        elif type(WaveSolutions) is list:
            if len(WaveSolutions) != self.nr_of_spectra():
                raise ValueError('WaveSolutions should have length {}, but has length {}!'.format(self.nr_of_spectra(), len(WaveSolutions)))

        new_SpectraList = SpectraList([])

        for i in range(self.nr_of_spectra()):
            new_spec = self._Spectra[i].applyWaveSolution(WaveSolutions[i], v_shift=v_shift)
            new_SpectraList.addSpectrum(new_spec)

        return new_SpectraList

    def contnorm_orders(self, Continuum):
        if Continuum.nr_of_spectra() == self.nr_of_spectra():
            Continuum_list = [Continuum[i] for i in range(Continuum.nr_of_spectra())]
        else:
            Continuum_list = [Continuum[0] for i in range(self.nr_of_spectra())]

        new_SpectraList = SpectraList([])

        for i in range(self.nr_of_spectra()):
            new_spec = self._Spectra[i].contnorm_orders(Continuum_list[i])
            new_SpectraList.addSpectrum(new_spec)

        return new_SpectraList



    def ToVacuum(self):
        for i in range(self.nr_of_spectra()):
            if isinstance(self._Spectra[i], Spectrum):
                self._Spectra[i].ToVacuum()

    def ToAir(self):
        for i in range(self.nr_of_spectra()):
            if isinstance(self._Spectra[i], Spectrum):
                self._Spectra[i].ToAir()


    def loglinear_wave(self, **kwargs):
        new_spectra = [spec.loglinear_wave(**kwargs) if isinstance(spec, Spectrum) else spec.copy() for spec in self._Spectra]

        return SpectraList(spectra=new_spectra)

    def mergeOrders(self, **kwargs):
        new_spectra = [spec.mergeOrders(**kwargs) if isinstance(spec, Spectrum) else spec.copy() for spec in self._Spectra]

        return SpectraList(spectra=new_spectra)

    def filter_outliners(self, **kwargs):
        new_spectra = [spec.filter_outliners(**kwargs) for spec in self._Spectra]

        return SpectraList(spectra=new_spectra)


    def __add__(self, other):
        assert isinstance(other, type(self))

        if self.nr_of_spectra() != other.nr_of_spectra():
            raise ValueError('Spectralists do not have the same number of spectra!')

        new_spectra = [self[spec_nr] + other[spec_nr] for spec_nr in range(self.nr_of_spectra())]

        return SpectraList(spectra=new_spectra)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        assert isinstance(other, type(self))

        if self.nr_of_spectra() != other.nr_of_spectra():
            raise ValueError('Spectralists do not have the same number of spectra!')

        new_spectra = [self[spec_nr] * other[spec_nr] for spec_nr in range(self.nr_of_spectra())]

        return SpectraList(spectra=new_spectra)

    def __truediv__(self, other):
        return self * (~other)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other


    def __neg__(self):
        new_spectra = [-self[spec_nr] for spec_nr in range(self.nr_of_spectra())]

        return SpectraList(spectra=new_spectra)

    def __invert__(self):
        new_spectra = [~self[spec_nr] for spec_nr in range(self.nr_of_spectra())]

        return SpectraList(spectra=new_spectra)

    def __len__(self):
        return self.nr_of_spectra()


class Pixel:
    """
        Represents one pixel on the CCD
    """
    def __init__(self, x, y, value=0):
        self.x     = x
        self.y     = y
        self.value = value

    def copy(self):
        return copy.deepcopy(self)

    def to_array(self):
        return np.array([self.x, self.y, self.value])

    def from_array(arr):
        if len(arr) == 3:
            return Pixel(arr[0], arr[1], value=arr[2])
        elif len(arr) == 2:
             return Pixel(arr[0], arr[1], value=0)
        else:
            raise ValueError('Invalid array!')

class Image:
    def __init__(self, data, errors=None, gain=1, RON=0, header=None):
        self.data   = data
        self.gain   = gain        #camera gain
        self.RON    = RON         #camera readout noise
        self.header = header

        if errors is None:
            self.errors =  np.sqrt(np.abs(data * gain) + np.power(RON, 2.)) / np.abs(gain)
        else:
            assert errors.shape == self.data.shape

            self.errors = errors


    def clip_negatives(self):
        #clip negative values to small (not zero) value
        self.data = np.clip(self.data, a_min=1e-10, a_max=None)

    def from_file(filename, raw=True, index=0):
        assert isinstance(filename, str)

        #only transpose raw images, not masterflat etc.
        if datashare.instrument is None or raw != True:
            transpose = False
        else:
            transpose = datashare.instrument.transpose_image

        if filename[-5:] == '.fits':
            data, errors, header = iocomp.image_from_file_fits(filename, transpose=transpose, index=index)

            if datashare.camera is None:
                RON = 0
                gain = 1
            else:
                RON = datashare.camera.get_RON(header)
                gain = datashare.camera.get_gain(header)


            if errors is None:
                errors = np.sqrt(np.abs(data * gain) + np.power(RON, 2.)) / gain

            return Image(data, errors=errors, gain=gain, RON=RON, header=header)
        elif filename[-5:] == '.hdf5':
            data, errors, gain, RON, header = iocomp.image_from_file_hdf5(filename)

            return Image(data, errors=errors, gain=gain, RON=RON, header=header)

        else:
            raise ValueError('invalid filename, has to end with .fits or .hdf5')

    def save(self, filename):
        assert isinstance(filename, str)

        if filename[-5:] == '.fits':
            iocomp.save_image_fits(filename, self)

        elif filename[-5:] == '.hdf5':
            iocomp.save_image_hdf5(filename, self)

        else:
            raise ValueError('invalid filename, has to end with .fits or .hdf5')

    def __add__(self, other):
        assert isinstance(other, Image)

        if not self.data.shape == other.data.shape:
            raise ValueError('images do not have the same shape!')

        gain = self.gain if self.gain == other.gain else 1
        RON  = self.RON  if self.RON  == other.RON  else 0

        new_data   = self.data + other.data
        new_errors = np.sqrt(np.power(self.errors, 2.) + np.power(other.errors, 2.))

        new_Image = Image(new_data, errors=new_errors, gain=gain, RON=RON, header=self.header)

        return new_Image

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        assert isinstance(other, Image)

        if not self.data.shape == other.data.shape:
            raise ValueError('images do not have the same shape!')

        gain = self.gain if self.gain == other.gain else 1
        RON  = self.RON  if self.RON  == other.RON  else 0

        new_data   = self.data * other.data
        new_errors = np.sqrt(np.power(other.data * self.errors, 2.) + np.power(self.data * other.errors, 2.))

        new_Image = Image(new_data, errors=new_errors, gain=gain, RON=RON, header=self.header)

        return new_Image

    def __truediv__(self, other):
        return self * (~other)

    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __idiv__(self, other):
        return self / other


    def __neg__(self):
        return Image(-self.data, errors=self.errors, gain=self.gain, RON=self.RON, header=self.header)

    def __invert__(self):
        return Image(np.power(self.data, -1.), errors=np.abs(self.errors/np.power(self.data, 2.)), gain=self.gain, RON=self.RON, header=self.header)

    def copy(self):
        return copy.deepcopy(self)

class Trace:
    """
        Stores the information of the Trace of one single order (one "line" on the chip)
    """
    def __init__(self, pol_coefficients, order_positions, maxpix, Centers=None, pix_range=None, fit_params=None):
        self.pol_coefficients = pol_coefficients
        self.order_positions  = order_positions         #found Centers, list of Pixels
        self.maxpix           = maxpix                  #maximum pixel value, used for chebyshev normalization
        self.Centers          = Centers                 #computed Centers
        self.pix_range        = pix_range
        self.tilt             = 0                       #mean tilt of the spectral lines in this order, in rad. Can be a scalar (constant for all pixels) or a array with a tilt for all pixels in pix_range
        self.tilt_err         = np.inf                  #error of tilt, can also be a scalar or an array
        self.fit_params       = fit_params
        self.line_plotter     = None

        if fit_params is not None:
            if 'sigma' in fit_params.keys():    #no image slicer
                self.sigma = fit_params['sigma']
            elif 'sig1' in fit_params.keys() and 'sig2' in fit_params.keys(): #image slicer
                self.sigma = fit_params['sig1'] + fit_params['sig2']
        else:
            self.sigma = -1

    def compute_centers(self, pix_range):
        pix_range_norm = (2 * np.array(pix_range) / self.maxpix) - 1

        self.Centers   = np.polynomial.chebyshev.chebval(pix_range_norm, self.pol_coefficients)
        self.pix_range = pix_range

    def center_at_pixel(self, x):
        x_norm = (2 * np.array(x) / self.maxpix) - 1

        return np.polynomial.chebyshev.chebval(x_norm, self.pol_coefficients)

    def copy(self):
        return copy.deepcopy(self)

class Fiber_traces:
    """
        Stores the informations for the trace of one fiber
        Necessary if order_multiplicity > 1 (e.g. one object and one sky fiber)
    """
    def __init__(self, filename = None):
        self.traces = []
        self.nr_of_orders = 0
        self.Centers = None
        self.filename = filename
        self.filename_of_spec = None
        self.parent_filename = None
        self.type = None            #whether object or sky/cali

    def __len__(self):
        return self.nr_of_orders

    def add_trace(self, pol_coefficients, order_positions, maxpix, Centers=None, pix_range=None, fit_params=None):
        self.traces.append(Trace(pol_coefficients, order_positions, maxpix, Centers=Centers, pix_range=pix_range, fit_params=fit_params))
        self.nr_of_orders += 1


    def add_trace_class(self, trace):
        if not isinstance(trace, Trace):
            raise ValueError('trace must be a Trace object!')

        self.traces.append(trace)
        self.nr_of_orders += 1

    def insert_trace_class(self, trace, index):
        if not isinstance(trace, Trace):
            raise ValueError('trace must be a Trace object!')

        if index < 0 or index > len(self.traces):
            raise ValueError('Invalid index!')

        self.traces.insert(index, trace)
        self.nr_of_orders += 1


    def get_trace(self, index):
        if np.abs(index) >= len(self.traces):
            raise ValueError('index must be >0 and <{}!'.format(len(self.traces)))

        return self.traces[index].copy()

    def all_traces(self):
        return [tr.copy() for tr in self.traces]

    def remove_trace(self, index):
        if np.abs(index):
            raise ValueError('index must be >0 and <{}!'.format(len(self.traces)))

        self.traces.pop(index)
        self.nr_of_orders -= 1

    def __setitem__(self, string, value):
        if string == 'filename':
            self.filename = value
        elif string == 'filename_of_spec':
            self.filename_of_spec = value
        elif string == 'parent_filename':
            self.parent_filename = value
        elif string == 'type':
            self.type = value
        else:
            raise ValueError('Input None or not part of Fiber_traces.')

    def compute_centers(self, pix_range):
        for tr in self.traces:
            tr.compute_centers(pix_range)

    def allCenters(self):
        return [tr.Centers for tr in self.traces]

    def all_order_positions(self):
        return [tr.order_positions for tr in self.traces]

    def all_pol_coefficients(self):
        return [tr.pol_coefficients for tr in self.traces]

    def setCenters(self, Centers):
        if len(Centers) != self.nr_of_orders:
            raise ValueError('length of Centers must be {}!'.format(self.nr_of_orders))

        for i in range(self.nr_of_orders):
            self.traces[i].Centers = Centers[i]

    def set_pol_coefficients(self, pol_coefficients):
        if len(pol_coefficients) != self.nr_of_orders:
            raise ValueError('length of pol_coefficients must be {}!'.format(self.nr_of_orders))

        for i in range(self.nr_of_orders):
            self.traces[i].pol_coefficients = pol_coefficients[i]

    def set_order_positions(self, order_positions):
        if len(order_positions) != self.nr_of_orders:
            raise ValueError('length of order_positionss must be {}!'.format(self.nr_of_orders))

        for i in range(self.nr_of_orders):
            self.traces[i].order_positions = order_positions[i]


   # def save(self, filename=None):
   #     """
   #         Save the data to a hdf5 file, as dictionary.
   #         TODO: Not working yet!!!
   #     """
   #     if filename == None:
   #         filename = self.filename
   #     if filename == None:
   #         raise ValueError('No output filename given!')
   #     elif splitext(filename)[-1] == ('' or not '.hdf5'):
   #         filename += '.hdf5'
   #     data = self.to_dict()
   #     #comp_io.save_dict_hdf5(data, filename)

   # def load(self, filename=None):
   #     """
   #         Load the data from a hdf5 file, as dictionary
   #         TODO: not working yet!
   #     """
   #     if filename == None:
   #         filename = self.filename
   #
   #     if filename == None:
   #         raise ValueError('No input filename given!')
   #
   #     else:
   #         data = comp_io.load_dict_hdf5(filename)
   #         if type(data) == dict:
   #             self.from_dict(data)
   #         else:
   #             raise ValueError('Data of type {} is no Trace_data object!'.format(type(data)))


    def __str__(self):
        return '<Fiber_traces: filename: {},'.format(self.filename) + \
                '\n\t     Corresponding spectrum: {},'.format(self.filename_of_spec) + \
                '\n\t     Number of orders: {})>'.format(self.nr_of_orders) + \
                '\n\t     type: {}'.format(self.type) + \
                '>'

    def to_dict(self):
        order_positions   = [tr.order_positions   for tr in self.traces]
        Centers           = [tr.Ceenters          for tr in self.traces]
        pol_coefficitents = [tr.pol_coefficitents for tr in self.traces]
        pix_range         = [tr.pix_range         for tr in self.traces]
        params            = [tr.fit_param         for tr in self.traces]

        data = {
                'pol_coefficients': pol_coefficients,
                'nr_of_orders': self.nr_of_orders,
                'order_positions': order_positions,
                'maxpix': self.maxpix,
                'Centers': Centers,
                'pix_range': pix_range,
                'fit_params':params,
                'filename': self.filename,
                'filename_of_spec': self.filename_of_spec,
                'parent_filename': self.parent_filename,
                'type': self.type,
                }

        return data

    def from_dict(self, data=None):
        if type(data) != dict:
            raise ValueError('Input needs to be a dictionary!')
        else:
            keys = ['filename', 'filename_of_spec', 'parent_filename', 'type']

            try:
                for i in range(np.max((len(data['order_positions']), len(data['Centers']), len(data['pix_range']), len(data['pol_coefficitents']), len(data['fit_params'])))):
                    self.add_trace(data['pol_coefficients'][i], data['order_positions'][i], data['maxpix'], Centers=data['Centers'][i], pix_range = data['pix_range'][i], fit_params=fit_params[i])
            except:
                pass

            for key in keys:
                try:

                    self[key] = data[key]
                except:
                    print('Key {} not in dictionary.'.format(key))

    def copy(self):
        return copy.deepcopy(self)


class Trace_data:
    """
        Store the information on the traces of the orders.
    """
    def __init__(self, filename = None):
        self.traces = []            #list of fiber_traces objects
        self.nr_of_orders = 0      #total number of orders
        self.order_multiplicity = 1
        self.filename = filename
        self.filename_of_spec = None
        self.daughter_filenames = None #list of filenames of fiber_traces objects
        self.ron_flat = None
        self.gain_flat = None
        self.ron_bias = None
        self.gain_bias = None

    def add_traces(self, pol_coefficients, nr_of_orders, order_positions, maxpix, fit_params=None):
        Traces = Fiber_traces()

        if not (nr_of_orders == len(pol_coefficients) == len(order_positions)):
            raise ValueError('Wrong input data shapes!')

        if fit_params is not None and len(fit_params) == nr_of_orders:
            for i in range(nr_of_orders):
                Traces.add_trace(pol_coefficients[i], order_positions[i], maxpix, fit_params=fit_params[i])
        else:
            for i in range(nr_of_orders):
                Traces.add_trace(pol_coefficients[i], order_positions[i], maxpix)

        self.nr_of_orders       += nr_of_orders

        self.traces.append(Traces)

    def add_traces_obj(self, Traces):
        if not isinstance(Traces, Fiber_traces):
            raise ValueError('Traces must be a Fiber_traces object!')

        self.traces.append(Traces)
        self.nr_of_orders += Traces.nr_of_orders

    def getMinDistances(self, fibernr, ordernr, pixrange, image_height):
        """
        # Get minimum pixel distances below and above the requested order to the neighboring orders
        """

        mean_pix = np.mean(pixrange)

        current_trace     = self.traces[fibernr].traces[ordernr]
        current_midcenter = current_trace.center_at_pixel(mean_pix)

        fiber_nrs = []
        order_nrs = []
        centers   = []

        for fib_nr in range(len(self.traces)):
            for ord_nr in range(len(self.traces[fib_nr].traces)):
                if fib_nr != fibernr and ord_nr != ordernr:
                    fiber_nrs.append(fib_nr)
                    order_nrs.append(ord_nr)
                    centers.append(self.traces[fib_nr].traces[ord_nr].center_at_pixel(mean_pix))

        fiber_nrs = np.array(fiber_nrs)
        order_nrs = np.array(order_nrs)
        centers   = np.array(centers)


        current_centers = current_trace.center_at_pixel(pixrange)

        #lower inds
        lower_inds = np.asarray(centers < current_midcenter).nonzero()[0]

        if len(lower_inds) > 0:
            nearest_lower_ind = lower_inds[np.argmin(np.abs(centers[lower_inds] - current_midcenter))]

            lower_trace = self.traces[fiber_nrs[nearest_lower_ind]].traces[order_nrs[nearest_lower_ind]]

            lower_centers = lower_trace.center_at_pixel(pixrange)

            lower_dist = np.min(np.abs(lower_centers - current_centers))

        else:
            lower_dist = np.floor(np.min(current_centers)).astype(int)

        #upper inds
        upper_inds = np.asarray(centers > current_midcenter).nonzero()[0]

        if len(upper_inds) > 0:
            nearest_upper_ind = upper_inds[np.argmin(np.abs(centers[upper_inds] - current_midcenter))]

            upper_trace = self.traces[fiber_nrs[nearest_upper_ind]].traces[order_nrs[nearest_upper_ind]]

            upper_centers = upper_trace.center_at_pixel(pixrange)

            upper_dist = np.min(np.abs(upper_centers - current_centers))
        else:
            upper_dist = np.floor(image_height - np.max(current_centers)).astype(int)


        return lower_dist, upper_dist


    def split_to_fibers(self, Image, order_multiplicity = 1):
        """
        # Split detected orders in single fibers
        #
        """

        order_multiplicity = int(order_multiplicity)


        if order_multiplicity == 1:
            self.traces[0].filename = self.filename.replace('.hdf5', '_obj.hdf5')
            self.daughter_filenames = [self.traces[0].filename]

            self.traces[0].type = 'OBJECT'

            logging.info("Only one fiber, not splitting in different traces")

        elif order_multiplicity < 1:
            raise ValueError("order_multiplicity must be equal or larger than 1, but is {}".format(order_multiplicity))

        elif len(self.traces) < 1:
             raise ValueError("No traces stored")

        elif len(self.traces) > 1:
            #TODO: Maybe do this better
            logging.info("Traces already splitted, do nothing")

        else:
            self.order_multiplicity = order_multiplicity

            if self.order_multiplicity == 2:
                image = Image.data

                x_range = np.arange(image.shape[1])

                self.traces[0].compute_centers(x_range)

                Centers = self.traces[0].allCenters()

                #median distances between the orders
                dists = [np.abs(np.median(Centers[i][:] - Centers[i+1][:])) for i in range(len(Centers) - 1)]

                #round values
                dists = np.around(dists).astype(int)


                #get most frequent value, this is the distance between the two fibers
                (values,counts) = np.unique(dists,return_counts=True)
                ind=np.argmax(counts)

                fiber_dist = values[ind]

                #median signal at each order
                heights = [np.median([image[np.round(Centers[i][j]).astype(int), j] for j in np.arange(image.shape[1])]) for i in range(len(Centers))]

                height_diff = np.diff(heights)


                #range around fiber_dist, to avoid errors
                dist_tolerance = 1

                height_tolerance = 0.75  #relative height tolerance

                upper_inds = []        #upper_inds > lower_inds, one pair of orders
                lower_inds = []

                for i in range(len(dists)):
                    if i in upper_inds or i in lower_inds:
                        continue

                    if np.abs(dists[i] - fiber_dist) <= dist_tolerance and np.abs(height_diff[i])/np.max((heights[i], heights[i+1])) < height_tolerance:
                        lower_inds.append(i)
                        upper_inds.append(i+1)


                #check lonely orders
                lonely_inds = [i for i in range(len(Centers)) if (i not in lower_inds and i not in upper_inds)]

                #try to find companions for lonely orders
                for ind in lonely_inds:
                    mid_pix = np.round(image.shape[1] / 2.).astype(int)

                    #first or last order, most likely companion order very weak. Just skip this order
                    #TODO: Maybe keep it? May break things later
                    if ind == 0 or ind == len(Centers) - 1:
                        pass
                    else:
                        #companion order most likely is in larger gap to prev/next order
                        # dists[i] = Center[i] - Center[i+1]

                        #next gap is larger
                        if dists[ind] - dists[ind -1] > 0:
                            exp_center = Centers[ind][mid_pix] + fiber_dist
                        else:
                            exp_center = Centers[ind][mid_pix] - fiber_dist

                        new_trace = order_traces.findSingleOrder(Image, exp_center)

                        #bad fit, reject new order and also do not use lonely order
                        if new_trace is None:
                            pass

                        lonely_trace = self.traces[0].get_trace(ind)
                        lonely_trace.compute_centers(x_range)

                        new_trace.compute_centers(x_range)

                        #traces differ too much, reject
                        if np.any(np.abs(np.abs(lonely_trace.Centers - new_trace.Centers) - fiber_dist) > dist_tolerance):
                            pass
                        else:
                            #add trace
                            self.traces[0].add_trace_class(new_trace)

                            if exp_center > Center[ind, mid_pix]:
                                upper_inds.append(self.traces[0].nr_of_orders - 1)
                                lower_inds.append(ind)
                            else:
                                upper_inds.append(ind)
                                lower_inds.append(self.traces[0].nr_of_orders - 1)

                upper_inds = np.array(upper_inds)
                lower_inds = np.array(lower_inds)

                #sort traces by pixels
                upper_centers = [np.median(self.traces[0].get_trace(ind).Centers) for ind in upper_inds]

                sort_inds = np.argsort(upper_centers)

                #upper_inds[i] belongs to lower_inds[i]
                upper_traces = [self.traces[0].get_trace(ind) for ind in upper_inds[sort_inds]]
                lower_traces = [self.traces[0].get_trace(ind) for ind in lower_inds[sort_inds]]

                #distinguish object and comparison fiber. Usually the object fiber is brighter
                upper_height = np.median([np.median([image[np.round(trace.Centers[j]).astype(int), j] for j in np.arange(image.shape[1])]) for trace in upper_traces])
                lower_height = np.median([np.median([image[np.round(trace.Centers[j]).astype(int), j] for j in np.arange(image.shape[1])]) for trace in lower_traces])

                if upper_height > lower_height:
                    obj_traces = upper_traces
                    sky_traces = lower_traces
                else:
                    obj_traces = lower_traces
                    sky_traces = upper_traces

                Trace_obj = Fiber_traces()
                Trace_sky = Fiber_traces()

                #assign values to Fiber_traces objects
                for trace in obj_traces:
                    Trace_obj.add_trace_class(trace)
                for trace in sky_traces:
                    Trace_sky.add_trace_class(trace)

                #TODO: Is that still necessary?
                if self.filename is not None:
                    Trace_obj.filename = self.filename.replace('.hdf5', '_obj.hdf5')
                    Trace_sky.filename = self.filename.replace('.hdf5', '_sky.hdf5')

                Trace_obj.parent_filename = self.filename
                Trace_sky.parent_filename = self.filename

                Trace_obj.filename_of_spec = self.filename_of_spec
                Trace_sky.filename_of_spec = self.filename_of_spec

                Trace_obj.type = 'OBJECT'
                Trace_sky.type = 'COMPARISON'

                self.nr_of_orders = Trace_obj.nr_of_orders + Trace_sky.nr_of_orders

                self.daughter_filenames = [Trace_obj.filename, Trace_sky.filename]
                self.traces = [Trace_obj, Trace_sky]

                logging.info("Splitted traces into object and sky/calib traces")

                #plot, if requested
                if datashare.reduction_parameters.plot_Fibersplit:
                    fig, ax = plt.subplots(figsize=(16,9))

                    ax.imshow(image, vmin=0, vmax=1000, cmap='gray')

                    for trace in obj_traces:
                        ax.plot(x_range, trace.Centers, color='red', zorder=2)

                    for trace in sky_traces:
                        ax.plot(x_range, trace.Centers, color='orange', zorder=2)

                    ax.set_xlabel('Pixel')
                    ax.set_ylabel('Pixel')

                    legend_elements = [Line2D([0], [0], color='red', ls='-',lw=2, label='Object fiber'), \
                                       Line2D([0], [0], color='orange', ls='-',lw=2, label='Comparison fiber')]

                    ax.legend(handles=legend_elements, loc='upper left')

                    ax.set_title('Orders splitted to fibers')

                    plt.tight_layout()

                    if datashare.reduction_parameters.save_plots:
                        filename = os.path.join(datashare.reduction_parameters.plot_dir, "Fibersplit.png")
                        plt.savefig(filename, dpi=300)

                    if datashare.reduction_parameters.show_plots:
                        plt.show()

                    plt.close()


            else:
                logging.info("case order_multiplicity > 2 not yet implemented, doing nothing")




    def set_Centers(self, Centers, tr_nr = 0):
        if len(self.traces) > tr_nr:
            self.traces[tr_nr].setCenters(Centers)

        else:
            raise ValueError("tr_nr >  len(self.traces), {} > {}".format(tr_nr, len(self.traces)))


    def set_pol_coefficients(self, pol_coefficients, tr_nr = 0):
        if len(self.traces) > tr_nr:
            self.traces[tr_nr].set_pol_coefficients(pol_coefficients)

        else:
            raise ValueError("tr_nr >  len(self.traces), {} > {}".format(tr_nr, len(self.traces)))

    def set_order_positions(self, order_positions, tr_nr = 0):
        if len(self.traces) > tr_nr:
            self.traces[tr_nr].set_order_positions(order_positions)

        else:
            raise ValueError("tr_nr >  len(self.traces), {} > {}".format(tr_nr, len(self.traces)))


    def allCenters(self):
        if len(self.traces) == 0:
            return None
        elif len(self.traces) == 1:
            return self.traces[0].allCenters()
        else:
            Centers = self.traces[0].allCenters()

            for i in range(1, len(self.traces)):
                Centers = np.append(Centers, self.traces[i].allCenters(), axis = 0)


            #sort Centers by median value
            Centers = Centers[np.median(Centers,axis=1).argsort()]

            return Centers

    def allPol_Coefficients(self):
        Pols = []

        for Trace in self.traces:
            Pols += Trace.all_pol_coefficients()

        return Pols

    def __len__(self):
        return self.nr_of_orders

    def __setitem__(self, string, value):
        if string == 'nr_of_orders':
            self.nr_of_orders = value
        elif string == 'filename':
            self.filename = value
        elif string == 'filename_of_spec':
            self.filename_of_spec = value
        elif string == 'ron_flat':
            self.ron_flat = value
        elif string == 'gain_flat':
            self.gain_flat = value
        elif string == 'ron_bias':
            self.ron_bias = value
        elif string == 'gain_bias':
            self.gain_bias = value
        elif string == 'daughter_filenames':
            self.daughter_filenames = value
        elif string == 'order_multiplicity':
            self.order_multiplicity = value
        else:
            raise ValueError('Input None or not part of Trace_data.')

    def save(self, filename=None):
        """
            Save the data to a hdf5 file, as dictionary.
        """

        """
        if filename == None:
            filename = self.filename
        if filename == None:
            raise ValueError('No output filename given!')
        elif splitext(filename)[-1] == ('' or not '.h5'):
            filename += '.h5'

        if len(self.traces) == 1:
            if self.traces[0].filename is None:
                self.traces[0].filename = self.filename.replace('.h5', '_1.h5')

            self.daughter_filenames = [self.traces[0].filename]

        data = self.to_dict()
        comp_io.save_dict_hdf5(data, filename)

        for Trace in self.traces:
            Trace.save()

        """
        iocomp.save_traces(filename, self)


    def load(filename=None):
        """
            Load the data from a hdf5 file, as dictionary.
        """
        return iocomp.load_traces(filename)


    def toOneString(self, stringl, sep = ":"):
        if len(stringl) == 1:
            return stringl[0]


        s = ""

        for i in range(len(stringl)):
            if i == len(stringl) - 1:
                s += stringl[i]
            else:
                s += stringl[i] + sep

        return s

    def __str__(self):
        return '<Trace_data (Filename: {},'.format(self.filename) + \
                '\n\t     Corresponding spectrum: {},'.format(self.filename_of_spec) + \
                '\n\t     Number of orders: {})>'.format(self.nr_of_orders)

    def to_dict(self):
        daughter_filenames = self.toOneString(self.daughter_filenames)


        data = {
                'order_multiplicity' : self.order_multiplicity,
                'nr_of_orders': self.nr_of_orders,
                'filename': self.filename,
                'daughter_filenames': daughter_filenames,
                'filename_of_spec': self.filename_of_spec,
                'ron_flat': self.ron_flat,
                'gain_flat': self.gain_flat,
                'ron_bias': self.ron_bias,
                'gain_bias': self.gain_bias
                }
        return data

    def from_dict(self, data=None):
        if type(data) != dict:
            raise ValueError('Input needs to be a dictionary!')
        else:
            keys = ['order_multiplicity', 'nr_of_orders', 'filename', 'daughter_filenames', \
                    'filename_of_spec', 'ron_flat', 'gain_flat', 'ron_bias', 'gain_bias']
            for key in keys:
                try:
                    try:
                        self[key] = data[key].decode()
                    except:
                        self[key] = data[key]
                except:
                    print('Key {} not in dictionary.'.format(key))

    def nr_of_fibers(self):
        return len(self.traces)

    def copy(self):
        return copy.deepcopy(self)

#TODO: Fit Funktionen in andere Datei auslagern, damit es hier etwas übersichtlicher wird?
class ContinuumFitter:
    """
    # The idea how to fit the spectral continuum comes from the pipeline PyReduce (https://github.com/AWehrhahn/PyReduce). All credit goes to them.
    # The concept of this method can be found in the corresponding publication ("Optimal extraction of echelle spectra: Getting the most out of observations", Piskunov, Wehrhahn & Marquart, 2021)
    #
    # The code was not directly ported, but adapted to my data structure
    """

    def __init__(self, mask_balmer=False, mask_tellurics=False, telluric_atlas=None, tellurics_FWHM=5, balmer_FWHM = 50):
        if telluric_atlas is None:
            self.telluric_lines = None
        elif isinstance(telluric_atlas, np.ndarray) and len(telluric_atlas.shape) >= 2:
            self.telluric_lines = telluric_atlas
        elif type(telluric_atlas) is str:
            try:
                self.telluric_lines = np.genfromtxt(telluric_atlas, comments='#', delimiter='  ', autostrip=True)

                if len(self.telluric_lines) < 2:
                    self.telluric_lines = None
            except Exception as e:
                self.telluric_lines = None
        else:
            self.telluric_lines = None

        self.tellurics_FWHM = tellurics_FWHM
        self.balmer_FWHM    = balmer_FWHM

        self.mask_balmer    = mask_balmer
        self.mask_tellurics = mask_tellurics

        if self.telluric_lines is None:
            self.mask_tellurics = False

        # balmer lines in angstrom, up to H-eta
        balmer_reference = np.array([3771.689, 3798.972, 3835.40, 3889.06, 3970.08, 4101.73, 4340.47, 4861.35, 6562.79])

        # create list of masked areas
        self.balmer_lines = np.vstack([balmer_reference - self.balmer_FWHM, balmer_reference + self.balmer_FWHM]).T


    def _fit(self, order, guess=None, niter = 10, lam1=1e5, lam2=5e6, method='upper', filter_bins=100, barycorrv=0.):
        # This function wants to find the fit function for the continuum
        #
        # Parameters
        # -----------
        # order: SpectralOrder object to fit
        # guess: SpectralOrder object with initial guess for continuum. If not defined use median of spectrum instead.
        # niter: int, Number of iterations (default 10)
        # lam1 : float, Constraint for first derivative
        # lam2 : float, Constraint to second derivative
        #
        # Returns
        # ----------
        # Cont: SpectralOrder object with continuum


        order = order.copy()

        #check if wavelengths are sorted
        if not np.all(order.wave[:-1] < order.wave[1:]):
            sort_inds = np.argsort(order.wave)

            order.wave = order.wave[sort_inds]
            order.flux = order.flux[sort_inds]

            if order.errors is not None:
                order.errors = order.errors[sort_inds]

        median = np.ma.median(order.flux)
        #dev = np.sqrt(np.nansum((order.flux- median) ** 2) / len(order.flux))

        #bad_inds = np.where(np.abs(order.flux- median) > 5 * dev)[0]
        #order.flux[bad_inds] = 0

        if guess is None:
            filter_size = int(len(order.flux) / filter_bins)
            #median_flux = np.ones_like(order.wave) * np.ma.median(order.flux)
            median_flux = median_filter(order.flux, size=filter_size)


            guess = SpectralOrder(order.wave.copy(), median_flux, errors=np.zeros_like(median_flux))

        # maximal differences:
        maxdiff1 = 1e-4
        maxdiff2 = 0.001 * (1 - 1./(np.min([2.,np.sqrt(median)])))

        method = method.lower()
        allowed_methods = ['middle', 'upper']
        if not method in allowed_methods:
            raise ValueError('method must be in {}'.format(allowed_methods))

        if method == 'middle':
            fit_func = self._fit_middle
        elif method == 'upper':
            fit_func = self._fit_upper


        weights = np.ones_like(order.flux)


        if isinstance(order, SpectralOrder):
            wmin = np.nanmin(order.wave)
            wmax = np.nanmax(order.wave)

            #mask balmer lines
            if self.mask_balmer:
                for i in range(self.balmer_lines.shape[0]):
                    start, end = self.balmer_lines[i,0], self.balmer_lines[i,1]

                    if wmin <= start <= wmax or wmin <= end <= wmax:
                        mask_inds = np.where((order.wave >= start) & (order.wave <= end))[0]
                        weights[mask_inds] = 0

                if np.count_nonzero(weights) < 5:
                    weights = np.ones_like(order.flux)

            if self.mask_tellurics:
                for i in range(self.telluric_lines.shape[0]):
                    start, end = self.telluric_lines[i,0] - self.tellurics_FWHM, self.telluric_lines[i,1] + self.tellurics_FWHM

                    #apply barycorr shift
                    start *= (1 + barycorrv/Constants.c)
                    end   *= (1 + barycorrv/Constants.c)

                    if wmin <= start <= wmax or wmin <= end <= wmax:
                        mask_inds = np.where((order.wave >= start) & (order.wave <= end))[0]
                        weights[mask_inds] = 0

                if np.count_nonzero(weights) < 5:
                    weights = np.ones_like(order.flux)

        weights_copy = weights.copy()

        for i in range(niter):
            tmp_normed = order / guess

            weights = weights_copy.copy()

            for _ in range(niter):
                tmp_norm, weights = fit_func(tmp_normed.flux, errors=tmp_normed.errors, weights=weights, niter=niter, lam1=lam1, lam2=lam2, maxdiff=maxdiff1)
                tmp_normed.flux   = np.clip(tmp_norm, tmp_normed.flux, None)

            tmp_normed.flux, _ = fit_func(tmp_normed.flux, errors=tmp_normed.errors, weights=weights, niter=niter, lam1=lam1, lam2=lam2,
                                                 maxdiff=maxdiff2)

            # scale new continuum guess
            tmp_normed = tmp_normed * guess

            if np.sum(np.abs(tmp_normed.flux - guess.flux)) < maxdiff2 * len(guess.flux):
                break


            #interpolate areas with zero weight
            good_inds = np.where(weights > 0.5)[0]
            min_inds = np.arange(np.min(good_inds))
            max_inds = np.arange(np.max(good_inds), len(weights))

            good_inds = np.append(good_inds, min_inds)
            good_inds = np.append(good_inds, max_inds)
            good_inds = np.sort(np.unique(good_inds))

            interpolator    = interpolate.Akima1DInterpolator(tmp_normed.wave[good_inds], tmp_normed.flux[good_inds])
            tmp_normed.flux = interpolator(tmp_normed.wave)

            guess = tmp_normed.copy()
            guess.errors = np.zeros_like(guess.flux)

        Cont = guess.copy()

        return Cont

    def fit(self, SpecList, guess=None, niter = 10, lam1=1e5, lam2=5e6, workers=1, method='middle', adjust_lam=False, filter_bins=100):
        # This function wants to find the fit function for the continuum
        #
        # Parameters
        # -----------
        # SpecList  : SpectraList object to fit
        # guess     : SpectraList object with initial guess for continuum. If not defined use median of spectrum instead.
        # niter     : int, Number of iterations (default 10)
        # lam1      : float, Constraint for first derivative
        # lam2      : float, Constraint to second derivative
        # workers   : int, number of parallel workers, 1 by default
        # method    : string, fit 'middle' or 'upper' edge of spectrum
        # adjust_lam: boolean, whether to adjust the lambdas based on the length of the spectrum
        #
        # Returns
        # ----------
        # Cont: SpectralOrder object with continuum

        Continuum = SpectraList()

        try:
            for fiber_nr in range(SpecList.nr_of_spectra()):
                Fiber_Spectrum = Spectrum()


                try:
                    barycorrv = float(SpecList[fiber_nr].header['HIERARCH BARYCORR km/s'])
                except:
                    barycorrv = 0.

                for order_nr in range(SpecList[fiber_nr].nr_of_orders()):
                    order = SpecList[fiber_nr][order_nr]

                    if guess is not None:
                        try:
                            order_guess = guess[fiber_nr][order_nr]
                        except:
                            order_guess = None
                    else:
                        order_guess = None

                    if adjust_lam:
                        #total empirical values
                        lam1 = len(order.flux)
                        lam2 = 50 * lam1

                    method = method.lower()
                    allowed_methods = ['middle', 'upper']
                    if not method in allowed_methods:
                        raise ValueError('method must be in {}'.format(allowed_methods))

                    if workers == 1:
                        Cont = self._fit(order, guess=order_guess, niter=niter, lam1=lam1, lam2=lam2, method=method, filter_bins=filter_bins, barycorrv=barycorrv)

                    else:
                        #split wavelength range in (workers) different parts
                        min_wav, max_wav = np.min(order.wave), np.max(order.wave)
                        delta_wav = (max_wav - min_wav)
                        wav_steps = delta_wav / workers

                        #increase max_wav a bit to include it
                        max_wav += 1e-3


                        Spec_Parts  = []
                        Guess_Parts = []

                        for i in range(workers):
                            #include 10% of delta_wav to get besser fits at the edges
                            if i == 0:
                                start_wav = min_wav
                                end_wav   = min_wav + 1.1 * wav_steps

                            elif i == workers -1:
                                start_wav = min_wav + (i-0.1) * wav_steps
                                end_wav   = max_wav

                            else:
                                start_wav = min_wav + (i-0.1) * wav_steps
                                end_wav   = min_wav + (i+1.1) * wav_steps


                            Spec_Parts.append(order.return_between((start_wav, end_wav)))
                            if guess is not None:
                                Guess_Parts.append(guess.return_between((start_wav, end_wav)))

                        if Guess_Parts == []:
                            Guess_Parts = [None for _ in range(workers)]


                        niter_Parts  = [niter]      *workers
                        lam1_Parts   = [lam1]       *workers
                        lam2_Parts   = [lam2]       *workers
                        methods      = [method]     *workers
                        filter_binss = [filter_bins]*workers
                        barycorrvs   = [barycorrv]  * workers

                        with Pool(processes=npools, initializer=init_pools, initargs=(datashare.reduction_parameters, datashare.instrument, datashare.camera)) as p:
                            Cont_Parts = p.starmap(self._fit, zip(Spec_Parts, Guess_Parts, niter_Parts, lam1_Parts, lam2_Parts, methods, filter_binss, barycorrvs))

                        Cont = Cont_Parts[0].return_between((min_wav, min_wav + wav_steps))

                        for i in range(1, workers-1):
                            Cont.append(Cont_Parts[i].return_between((min_wav + i * wav_steps, min_wav + (i+1)*wav_steps)))

                        Cont.append(Cont_Parts[-1].return_between((min_wav + (workers-1)*wav_steps, max_wav)))

                    Fiber_Spectrum.addOrder(Cont)

                    if datashare.reduction_parameters.plot_ContNormalization:
                        fig, axs = plt.subplots(2, figsize=(10, 5), sharex=True)

                        axs[0].plot(*order.plot_data(), label='Measurement')
                        axs[0].plot(*Cont.plot_data(), label='Continuum fit')
                        axs[1].plot(*((order / Cont).plot_data()))

                        axs[0].legend()
                        axs[0].set(ylabel='Flux in a.u', title='Spectrum and continuum fit')
                        axs[1].set(xlabel=r'$\lambda$ in $\mathrm{\AA}$', ylabel='Normalized spectrum', title='Normalized spectrum')

                        if datashare.reduction_parameters.save_plots:
                            filename = os.path.join(datashare.reduction_parameters.plot_dir, os.path.splitext(datashare.current_filename)[0] + "_contnorm_fiber{}_order{}.png".format(fiber_nr, order_nr))
                            plt.savefig(filename, dpi=300)

                        if datashare.reduction_parameters.show_plots:
                            plt.show()

                        plt.close()

                Continuum.addSpectrum(Fiber_Spectrum)

        except Exception as e:
            logging.warning('WARNING: error during continuum normalization: {}'.format(str(e)))

            Continuum = None

        return Continuum

    def _optimal_fit(self, f, weights=None, lam1=1e5, lam2=5e6):
        # This function wants to find the fit function
        #
        # Parameters
        # -----------
        # f      : ndarray, function to fit
        # weights: weights for each datapoint, one by default
        # lam1   : Constraint for first derivative
        # lam2   : Constraint to second derivative
        #
        # Returns
        # ----------
        # fit: ndarray with fitted function

        if weights is None:
            weights = np.ones_like(f)
        else:
            weights = weights.copy()

        if f.ndim != 1:
            raise ValueError('f must be onedimensional!')

        s = f.size
        #matrix from https://github.com/AWehrhahn/PyReduce/blob/master/pyreduce/util.py

        if s < 5:
            return np.ones_like(f)

        matrix = np.zeros((5, s))
        # 2nd lower subdiagonal
        matrix[0, 2:] = lam2
        # Lower subdiagonal
        matrix[1, 1] = -lam1 - 2 * lam2
        matrix[1, 2:-1] = -lam1 - 4 * lam2
        matrix[1, -1] = -lam1 - 2 * lam2
        # Main diagonal
        matrix[2, 0] = weights[0] + lam1 + lam2
        matrix[2, 1] = weights[1] + 2 * lam1 + 5 * lam2
        matrix[2, 2:-2] = weights[2:-2] + 2 * lam1 + 6 * lam2
        matrix[2, -2] = weights[-2] + 2 * lam1 + 5 * lam2
        matrix[2, -1] = weights[-1] + lam1 + lam2
        # Upper subdiagonal
        matrix[3, 0] = -lam1 - 2 * lam2
        matrix[3, 1:-2] = -lam1 - 4 * lam2
        matrix[3, -2] = -lam1 - 2 * lam2
        # 2nd upper subdiagonal
        matrix[4, 0:-2] = lam2

        #solve differential equation
        fit = solve_banded((2,2), matrix, weights * f)

        return fit


    def _fit_middle(self, f, errors=None, x=None, weights=None, niter=10, lam1=1e5, lam2=5e6, minf=None, maxf=None, maxdiff=1e-3, error_threshold=10):
        # This function wants to find the fit function for f, fitting the 'middle' of the function
        #
        # Parameters
        # -----------
        # f      : ndarray, function to fit
        # errors : flux errors
        # x      : x grid for f, default use linspace from -1 to 1
        # weights: weights for each datapoint, one by default
        # niter: Number of iterations (default 10)
        # lam1   : Constraint for first derivative
        # lam2   : Constraint to second derivative
        # minf   : minimum value of f
        # maxf   : maximum value of f
        # maxdiff: maximal deviation between fit and function
        # error_threshold: if errors are given, this is the SNR threshold for relevant data points
        #
        # Returns
        # ----------
        # fit: ndarray with fitted function

        f = np.asarray(f)

        if weights is None:
            weights = np.ones_like(f)
        else:
            weights = weights.copy()

        if x is None:
            x = np.linspace(-1,1,f.size)

        if minf is None:
            minf = np.min(f)
        if maxf is None:
            maxf = np.max(f)

        #filter f by minimum/maximum values
        inds = np.where(np.logical_or((f < minf),(f > maxf)))[0]

        #f       = f[inds]
        #weights = weights[inds]
        weights[inds] = 0


        if errors is not None:
            #errors = errors[inds]

            inds = np.where(f <= error_threshold * np.abs(errors))[0]
            weights[inds] = 0.

            if np.all(weights == 0):
                weights = np.ones_like(f)


        minf -= 1
        maxf += 1

        #norm f
        fit_f     = (f - minf) / (maxf - minf)
        fit_f_old = fit_f

        for _ in range(niter):
            # fit
            tmp_fit = median_filter(self._optimal_fit(fit_f, weights=weights, lam1=lam1, lam2=lam2),3)
            tmp_err = self._optimal_fit(np.power(fit_f - tmp_fit, 2.), weights=weights, lam1=lam1, lam2=lam2)

            # get standart deviation, throw away values below 0
            dev = np.sqrt(np.clip(tmp_err, 0, None))

            #norm fit_f by values that are close enough to tmp_fit
            fit_f = np.clip(tmp_fit - 2 * dev, fit_f, tmp_fit + 2 * dev)

            #maximal deviation from new fit to old fit
            dev2 = np.max(weights * np.abs(fit_f - fit_f_old))
            fit_f_old = fit_f

            if dev2 <= maxdiff:
                break


        # renorm fit
        fit = tmp_fit * (maxf - minf) + minf

        return fit, weights

    def _fit_upper(self, f, errors=None, x=None, weights=None, niter=10, lam1=1e5, lam2=5e6, minf=None, maxf=None, maxdiff=1e-3, error_threshold=10):
        # This function wants to find the fit function for f, fitting the 'upper edge' of the function
        #
        # Parameters
        # -----------
        # f      : ndarray, function to fit
        # errors : flux errors
        # x      : x grid for f, default use linspace from -1 to 1
        # weights: weights for each datapoint, one by default
        # niter: Number of iterations (default 40)
        # lam1   : Constraint for first derivative
        # lam2   : Constraint to second derivative
        # minf   : minimum value of f
        # maxf   : maximum value of f
        # maxdiff: maximal deviation between fit and function
        # error_threshold: if errors are given, this is the SNR threshold for relevant data points
        #
        # Returns
        # ----------
        # fit: ndarray with fitted function

        f = np.asarray(f)

        if weights is None:
            weights = np.ones_like(f)
        else:
            weights = weights.copy()

        if x is None:
            x = np.linspace(-1, 1, f.size)

        if minf is None:
            minf = np.min(f)
        if maxf is None:
            maxf = np.max(f)

        #filter f by minimum/maximum values
        inds = np.where(np.logical_or((f < minf),(f > maxf)))[0]

        #f       = f[inds]
        #weights = weights[inds]
        weights[inds] = 0

        if errors is not None:
            #errors = errors[inds]

            inds = np.where(f <= error_threshold * np.abs(errors))[0]
            weights[inds] = 0.

            if np.all(weights == 0):
                weights = np.ones_like(f)



        minf -= 1
        maxf += 1

        # norm f
        fit_f = (f - minf) / (maxf - minf)
        fit_f_old = fit_f

        for i in range(niter):
            # fit
            tmp_fit = median_filter(self._optimal_fit(fit_f, weights=weights, lam1=lam1, lam2=lam2), 3)
            tmp_err = self._optimal_fit(np.power(fit_f - tmp_fit, 2.), weights=weights, lam1=lam1, lam2=lam2)

            # get standart deviation, throw away values below 0
            dev = np.sqrt(np.clip(tmp_err, 0, None))

            # norm fit_f by values that are close enough to tmp_fit
            # smaller limit at lower edge, higher limit on higher edge as we want to fit the 'upper edge' of the function
            fit_f = np.clip(tmp_fit - maxdiff, fit_f, tmp_fit + dev*2)

            # maximal deviation from new fit to old fit
            dev2 = np.max(weights * np.abs(fit_f - fit_f_old))
            fit_f_old = fit_f

            if dev2 <= maxdiff:
                break

        # renorm fit
        fit = tmp_fit * (maxf - minf) + minf

        return fit, weights
