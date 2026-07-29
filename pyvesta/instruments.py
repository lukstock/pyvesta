import numpy as np

from pyvesta import Spectra

#TODO: Add other parameters (as fitting polynomial degrees etc.) to this file!

class Instrument:
    """
    This object contains the information about all parameters of the instrument and the parameters used for extraction.
    """

    def __init__(self):
        #instrument information
        self.resolution = 1                         #Resolution of the instrument

        #geographic position of the instrument, used for barycentric correction
        self.lat = 0                                #Instrument latitude in degree
        self.lon = 0                                #Instrument longitude in degree
        self.alt = 0                                #Instrument altitude in m

        #----------------------------------------------------------------------------------------------------------------------------------------------------------------
        #instrument specific extraction parameters

        #use this for different filename extensions, if more than one image are stored in a fits file
        #e.g. "WaveSolution.hdf5" will become "WaveSolution_B.hdf5" with an extension "_B"
        #The default will lead to no extension
        self.filename_extensions_for_images = ''


        #JUST FOR HARPS / INSTRUMENTS WITH MORE THAN ONE IMAGE:
        #index of hdu to use this extraction on. This code must be run multiple times with different indices.
        #The different images are independent of each other, therefore we need to run the extraction individually for each image
        self.index = 0     #use 0 for "normal" images, 1 for HARPS blue and 2 for HARPS red


        self.Use_Flat_for_Orderdef = True           #whether to use flat frames for orderdef purposes (so if no explicit orderdef frames are available)

        self.transpose_image = False                #whether to transpose the image to make the orders run approximately along the pixel rows (horizontally)
        self.reverse_orders  = False                #whether to rearrange the extracted orders, so that higher order numbers correspond to smaller wavelengths (as lambda ~ 1/m)
        self.reverse_pixels  = False                #whether to rearrange the pixels, so that the wavelength increases along with the pixels
        self.use_darks       = True                 #whether to use dark frames

        self.order_separation = 10                  #approx. separation between orders in pixels
        self.orders_sigma     = 2                   #approx. gaussian standard deviation of orders in cross-dispersion direction in pixels
        self.order_deg        = 4                   #polynomial degree used for order tracing

        self.use_linetilt     = True                #whether to use line tilt. If False, tilt will be fixed at 0°

        self.image_slicer     = False               #whether the instrument has an image slicer
        self.nr_of_fibers     = 1                   #number of fibers of the spectrograph. Note that more than 2 fibers are currently not supported!

        self.background_degx  = 8                   #polynomial degree of background fitting in x (dispersion) direction
        self.background_degy  = 8                   #polynomial degree of background fitting in y (cross-dispersion) direction

        self.m0_default     = 0                     #Default m0 value (physical diffraction order number corresponding to first extracted order)
        self.m0_searchrange = 3                     #Search real m0 around m0_default in range (m0_default - m0_searchrange, m0_default + m0_searchrange +1)

        self.MaxWavesolution_RMS = 0.2              #Maximal acceptable RMS of final wavelength solution in angstrom
        self.wav_tol             = 0.5              #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

        self.reference_filename = '../reference/ThAr-Audela.txt'        #Relative (to this file) filename of the reference list with all reference lines (without pixel/order information). This file is mandatory
        self.peaklinelist       = ''                                    #Relative (to this file) filename of the reference list with pixel/order informations. This file is not mandatory. If this file is provided, the wavelength calibration is used the predefined peakslists in this file. If not, the overlap wavelength solution algorithm is used.

        self.max_pixshift = 100                     #Maximal pixel shift to reference, only used if peaklinelist is given
        self.reference_list_testorder = 0           #Physical diffraction order number which is used to determine m0 from the peaklist, only if peaklinelist is given



        #----------------------------------------------------------------------------------------------------------------------------------------------------------------
        #general extraction parameters

        self.npix_init_overlaps = 1                 #Polynomial degree for fitting overlaps between orders

        self.nord_overlaps = 1                      #Polynomial degree in cross-dispersion direction for wavelength fit from overlaps
        self.npix_overlaps = 1                      #Polynomial degree in pixel direction for wavelength fit from overlaps
        self.final_wavelength_fit_number_orders_to_start = 1        #how many orders to use when starting final wavelength fit (default 1, just middle one). If more than 1 will use neighboring orders of middle order

        self.nord_init_final = 1                    #Polynomial degree to start at when performing final wavelength fit in cross-dispersion direction
        self.npix_init_final = 1                    #Polynomial degree to start at when performing final wavelength fit in pixel direction
        self.nord_final      = 1                    #Final polynomial degree when performing final wavelength fit in cross-dispersion direction
        self.npix_final      = 1                    #Final polynomial degree when performing final wavelength fit in pixel direction

        self.pixfraction_blue = 0.75                #fraction of pixel range used in blue order for overlap detection
        self.pixfraction_red  = 0.5                 #fraction of pixel range used in red order for overlap detection

        self.tiltfit_orderdeg = -1                  #polynomial degree for tilt interpolation between orders. If < 0, each order will be fitted individually instead of an 2D fit over all orders
        self.tiltfit_pixeldeg = 1                   #polynomial degree for tilt interpolation within one order (along pixels)

        self.ordershape_dispdeg      = 3            #polynomial chebyshev degree for ordershape fit of chebyshev parameters in dispersion direction. Keep this small, as the parameters should vary slowly
        self.ordershape_crossdispdeg = 10           #polynomial chebyshev degree for ordershape fit in cross-dispersion direction. Here higher degrees can be used to ensure a nice fit.

        #------–----------------------------------------------------------------------------------------------------------------------------------------------------------
        #continuum fitter parameters

        self.cont_lam1              = 1e4                           #lambda_1 value for continuum calculation. Needs to be evalated manually for each instrument. The larger this value the less variations (first derivative) the continuum will have
        self.cont_lam2              = 1e5                           #lambda_2 value for continuum calculation. Needs to be evalated manually for each instrument. The larger this value the less variations (second derivative) the continuum will have
        self.cont_niter             = 2                             #number of iterations for continuum calculation
        self.cont_mask_balmer       = True                          #whether to mask balmer lines for continuum calculation
        self.cont_mask_tellurics    = True                          #whether to mask telluric lines for continuum calculation
        self.tellurics_filename     = './Referenz/ath_abs.txt'      #filename of tellurics reference file


    #these functions must be defined to read instrument headers correctly
    def getMJD(self, header):
        #return observation MJD (mid of observation) as float
        return np.nan

    def setMJD(self, header, mjd):
        #set mjd in header
        return header

    def getObjectName(self, header):
        #return name of target as starting
        return ''

    def setObjectName(self, header, name):
        #set object name in header
        return header

    def getExptime(self, header):
        #return exptime in s
        return np.nan

    def setExpTime(self, header, exptime):
        #set exptime in header
        return header

    def getStarCoordinates(self, header):
        #return RA (hourangle) and DEC (degree) of target star
        return np.nan, np.nan

    def setStarCoordinates(self, header, RA, DEC):
        #set RA und DEC in header
        return header

    def getTemperature(self, header):
        #return temperature
        return np.nan

    def setTemperature(self, header, temp):
        #set temperature in header
        return header

    def classifyFrame(self, header):
        #return ObservationMode
        return Spectra.ObservationMode.LIGHT


class Camera:
    """
    This object contains the information about the camera used to acquire the spectral images
    """
    def __init__(self):
        self.name     = ''        #name of camera


    def get_gain(self, header):
        #returns camera gain
        return 1

    def get_RON(self, header):
        #returns camera read out noise (in electrons)
        return 0

    def get_maxcount(self, header):
        #return maximal ADU count of camera.
        #Should be the maximal count where camera response is still linear, not necessarily the actual maximal value
        return np.inf

    def get_badrows(self, header):
        #returns list of bad rows of this camera. These rows will be interpolated by using the neighboring rows.
        return None

class FEROS(Instrument):
    def __init__(self):
        super().__init__()

        self.Use_Flat_for_Orderdef = True
        self.transpose_image = True
        self.reverse_orders  = False
        self.reverse_pixels  = False
        self.use_darks       = False                       #Whether to use dark frames

        self.order_separation = 14
        self.order_deg        = 4
        self.orders_sigma     = 2
        self.image_slicer     = True
        self.nr_of_fibers     = 2

        self.use_linetilt     = False

        self.nord_overlaps      = 3
        self.npix_overlaps      = 3
        self.nord_init_final    = 3
        self.npix_init_final    = 3
        self.nord_final         = 5
        self.npix_final         = 7
        self.nord_init_final_fromoverlaps = 3
        self.npix_init_final_fromoverlaps = 3
        self.nord_final_fromoverlaps      = 5
        self.npix_final_fromoverlaps      = 7
        self.npix_init_overlaps = 3

        self.final_wavelength_fit_number_orders_to_start = 1        #how many orders to use when starting final wavelength fit (default 1, just middle one). If more than 1 will use neighboring orders of middle order

        self.pixfraction_blue = 0.75    #fraction of pixel range used in blue / red ordre for overlap detection
        self.pixfraction_red  = 0.75

        self.m0_default     = 26
        self.m0_searchrange = 3


        self.ordershape_crossdispdeg = 20

        self.MaxWavesolution_RMS = 0.02     #maximum RMS of wavelength solution in angstrom
        self.wav_tol             = 0.5     #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

        self.reference_filename = './reference/FEROS_ThAr_List.txt'

        self.resolution = 48000

        #geographic position of the instrument
        self.lat = -29.254286944    #latitude in degree
        self.lon = -70.734595278    #longitude in degree
        self.alt = 2335    #altitude in m

    def getMJD(self, header):
        #return observation MJD (mid of observation) as float
        try:
            return float(header['MJD-OBS'])
        except:
            return np.nan

    def setMJD(self, header, mjd):
        #set mjd in header
        try:
            header['MJD-OBS'] = mjd
        except:
            pass

        return header

    def getObjectName(self, header):
        #return name of target as starting
        try:
            return str(header['OBJECT'])
        except:
            return ''

    def setObjectName(self, header, name):
        #set object name in header
        try:
            header['OBJECT'] = name
        except:
            pass

        return header

    def getExptime(self, header):
        #return exptime in s
        try:
            return float(header['EXPTIME'])
        except:
            return np.nan

    def setExpTime(self, header, exptime):
        #set exptime in header
        try:
            header['EXPTIME'] = exptime
        except:
            pass

        return header

    def getStarCoordinates(self, header):
        #return RA (hourangle) and DEC (degree) of target star
        try:
            RA  = float(header['RA'])
            DEC = float(header['DEC'])

            return RA, DEC
        except:
            return np.nan, np.nan

    def setStarCoordinates(self, header, RA, DEC):
        #set RA und DEC in header
        try:
            header['RA']  = RA
            header['DEC'] = DEC
        except:
            pass

        return header

    def getTemperature(self, header):
        #return temperature
        try:
            temp1 = float(header['HIERARCH ESO INS TEMP1 VAL'])
            temp2 = float(header['HIERARCH ESO INS TEMP2 VAL'])
            temp3 = float(header['HIERARCH ESO INS TEMP3 VAL'])
            temp4 = float(header['HIERARCH ESO INS TEMP4 VAL'])
            temp5 = float(header['HIERARCH ESO INS TEMP5 VAL'])

            return np.mean([temp1, temp2, temp3, temp4, temp5])

        except:
            return 0

    def setTemperature(self, header, temp):
        #set temperature in header

        #not supported, skip
        return header

    def getExpMode(self, header):
        #return ObservationMode
        if header is None or 'OBJECT' not in header:
            #return LIGHT if not known
            return Spectra.ObservationMode.LIGHT

        try:
            identifier = str(header['OBJECT']).lower()

            if 'bias' in identifier:
                return Spectra.ObservationMode.BIAS
            elif 'flat' in identifier:
                return Spectra.ObservationMode.FLAT
            elif 'wave' in identifier:
                return Spectra.ObservationMode.THAR

            #return LIGHT, as identifier is most likely the targets name
            else:
                return Spectra.ObservationMode.LIGHT
        except:
            return None

#Class for the blue part of HARPS
class HARPS_blue(Instrument):
    def __init__(self):
        super().__init__()

        self.filename_extensions_for_images = '_B'

        self.index = 1

        self.Use_Flat_for_Orderdef = True
        self.transpose_image = True
        self.reverse_orders  = True
        self.reverse_pixels  = True
        self.use_darks       = False

        self.order_separation = 10
        self.order_deg        = 4
        self.orders_sigma     = 2
        self.image_slicer     = False
        self.nr_of_fibers     = 2

        self.use_linetilt     = False


        self.nord_overlaps      = 3
        self.npix_overlaps      = 3
        self.nord_init_final    = 3
        self.npix_init_final    = 3
        self.nord_final         = 5
        self.npix_final         = 7
        self.nord_init_final_fromoverlaps = 2
        self.npix_init_final_fromoverlaps = 2
        self.nord_final_fromoverlaps      = 3
        self.npix_final_fromoverlaps      = 5
        self.npix_init_overlaps = 2

        self.final_wavelength_fit_number_orders_to_start = 3        #how many orders to use when starting final wavelength fit (default 1, just middle one). If more than 1 will use neighboring orders of middle order

        self.m0_default     = 116
        self.m0_searchrange = 3

        self.MaxWavesolution_RMS = 0.01     #maximum RMS of wavelength solution in angstrom
        self.wav_tol             = 0.1     #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

        self.reference_filename = './reference/HARPS_reference_B.txt'

        self.camera = HARPS_camera_blue()

        self.resolution = 120000

        #geographic position of the instrument
        self.lat = -29.2584    #latitude in degree
        self.lon = -70.7345    #longitude in degree
        self.alt = 2375    #altitude in m

    def getMJD(self, header):
        #return observation MJD (mid of observation) as float
        try:
            return float(header['MJD-OBS'])
        except:
            return np.nan

    def setMJD(self, header, mjd):
        #set mjd in header
        try:
            header['MJD-OBS'] = mjd
        except:
            pass

        return header

    def getObjectName(self, header):
        #return name of target as starting
        try:
            return str(header['OBJECT'])
        except:
            return ''

    def setObjectName(self, header, name):
        #set object name in header
        try:
            header['OBJECT'] = name
        except:
            pass

        return header

    def getExptime(self, header):
        #return exptime in s
        try:
            return float(header['EXPTIME'])
        except:
            return np.nan

    def setExpTime(self, header, exptime):
        #set exptime in header
        try:
            header['EXPTIME'] = exptime
        except:
            pass

        return header

    def getStarCoordinates(self, header):
        #return RA (hourangle) and DEC (degree) of target star
        try:
            RA  = float(header['RA'])
            DEC = float(header['DEC'])

            return RA, DEC
        except:
            return np.nan, np.nan

    def setStarCoordinates(self, header, RA, DEC):
        #set RA und DEC in header
        try:
            header['RA']  = RA
            header['DEC'] = DEC
        except:
            pass

        return header


    def getTemperature(self, header):
        #return temperature
        try:
            return float(header['HIERARCH ESO INS TEMP21 VAL'])     #Fiber exit temperature, more or less arbirtarily chosen
        except:
            return 0

    def setTemperature(self, header, temp):
        #set temperature in header

        #not supported, skip
        return header


    def getExpMode(self, header):
        #return ObservationMode
        if header is None or 'OBJECT' not in header:
            #return LIGHT if not known
            return Spectra.ObservationMode.LIGHT

        try:
            identifier = str(header['OBJECT']).lower()

            if 'bias' in identifier:
                return Spectra.ObservationMode.BIAS
            elif 'lamp' in identifier:
                return Spectra.ObservationMode.FLAT
            elif 'wave' in identifier:
                return Spectra.ObservationMode.THAR

            #return LIGHT, as identifier is most likely the targets name
            else:
                return Spectra.ObservationMode.LIGHT
        except:
            return None


#Class for the red part of HARPS
class HARPS_red(Instrument):
    def __init__(self):
        super().__init__()

        self.filename_extensions_for_images = '_R'

        self.index = 2

        self.Use_Flat_for_Orderdef = True
        self.transpose_image = True
        self.reverse_orders  = True
        self.reverse_pixels  = True
        self.use_darks       = False

        self.order_separation = 10
        self.order_deg        = 4
        self.orders_sigma     = 2
        self.image_slicer     = False
        self.nr_of_fibers     = 2

        self.use_linetilt     = False


        self.nord_overlaps      = 3
        self.npix_overlaps      = 3
        self.nord_init_final    = 3
        self.npix_init_final    = 3
        self.nord_final         = 5
        self.npix_final         = 7
        self.nord_init_final_fromoverlaps = 2
        self.npix_init_final_fromoverlaps = 2
        self.nord_final_fromoverlaps      = 5
        self.npix_final_fromoverlaps      = 7
        self.npix_init_overlaps = 3

        self.final_wavelength_fit_number_orders_to_start = 3        #how many orders to use when starting final wavelength fit (default 1, just middle one). If more than 1 will use neighboring orders of middle order

        self.pixfraction_blue = 0.2    #fraction of pixel range used in blue / red ordre for overlap detection
        self.pixfraction_red  = 0.2

        self.m0_default     = 89
        self.m0_searchrange = 1

        self.MaxWavesolution_RMS = 0.01     #maximum RMS of wavelength solution in angstrom
        self.wav_tol             = 0.1     #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

        self.reference_filename = './reference/HARPS_reference_R.txt'

        self.camera = HARPS_camera_red()

        self.resolution = 120000

        #geographic position of the instrument
        self.lat = -29.2584    #latitude in degree
        self.lon = -70.7345    #longitude in degree
        self.alt = 2375    #altitude in m

    def getMJD(self, header):
        #return observation MJD (mid of observation) as float
        try:
            return float(header['MJD-OBS'])
        except:
            return np.nan

    def setMJD(self, header, mjd):
        #set mjd in header
        try:
            header['MJD-OBS'] = mjd
        except:
            pass

        return header

    def getObjectName(self, header):
        #return name of target as starting
        try:
            return str(header['OBJECT'])
        except:
            return ''

    def setObjectName(self, header, name):
        #set object name in header
        try:
            header['OBJECT'] = name
        except:
            pass

        return header

    def getExptime(self, header):
        #return exptime in s
        try:
            return float(header['EXPTIME'])
        except:
            return np.nan

    def setExpTime(self, header, exptime):
        #set exptime in header
        try:
            header['EXPTIME'] = exptime
        except:
            pass

        return header

    def getStarCoordinates(self, header):
        #return RA (hourangle) and DEC (degree) of target star
        try:
            RA  = float(header['RA'])
            DEC = float(header['DEC'])

            return RA, DEC
        except:
            return np.nan, np.nan

    def setStarCoordinates(self, header, RA, DEC):
        #set RA und DEC in header
        try:
            header['RA']  = RA
            header['DEC'] = DEC
        except:
            pass

        return header

    def getTemperature(self, header):
        #return temperature
        try:
            return float(header['HIERARCH ESO INS TEMP21 VAL'])     #Fiber exit temperature, more or less arbirtarily chosen
        except:
            return 0

    def setTemperature(self, header, temp):
        #set temperature in header

        #not supported, skip
        return header

    def getExpMode(self, header):
        #return ObservationMode
        if header is None or 'OBJECT' not in header:
            #return LIGHT if not known
            return Spectra.ObservationMode.LIGHT

        try:
            identifier = str(header['OBJECT']).lower()

            if 'bias' in identifier:
                return Spectra.ObservationMode.BIAS
            elif 'lamp' in identifier:
                return Spectra.ObservationMode.FLAT
            elif 'wave' in identifier:
                return Spectra.ObservationMode.THAR

            #return LIGHT, as identifier is most likely the targets name
            else:
                return Spectra.ObservationMode.LIGHT
        except:
            return None


class eShel(Instrument):
    def __init__(self):
        super().__init__()

        self.Use_Flat_for_Orderdef = False
        self.transpose_image = False
        self.reverse_orders  = True
        self.reverse_pixels  = False
        self.use_darks       = True

        self.background_degx  = 4                   #polynomial degree of background fitting in x (dispersion) direction
        self.background_degy  = 4


        self.order_separation = 20
        self.order_deg        = 4
        self.orders_sigma     = 2
        self.image_slicer     = False
        self.nr_of_fibers     = 1

        self.nord_overlaps      = 3
        self.npix_overlaps      = 3
        self.nord_init_final    = 2
        self.npix_init_final    = 2
        self.nord_final         = 5         #was 5
        self.npix_final         = 6         #was 4
        self.nord_init_final_fromoverlaps = 3
        self.npix_init_final_fromoverlaps = 3
        self.nord_final_fromoverlaps      = 3
        self.npix_final_fromoverlaps      = 7
        self.npix_init_overlaps = 3

        self.final_wavelength_fit_number_orders_to_start = 1        #how many orders to use when starting final wavelength fit (default 1, just middle one). If more than 1 will use neighboring orders of middle order

        self.m0_default     = 27        #was 27
        self.m0_searchrange = 3         #was 3

        self.MaxWavesolution_RMS = 0.2
        self.wav_tol             = 0.5     #wavelength tolerance in angstrom. Map detected peak and reference peak if difference is smaller than wav_tol

        self.reference_filename = './reference/ThAr-Audela.txt'
        #self.peaklinelist       = './reference/Reference_eShel.csv'


        self.camera = Atik383L()

        self.resolution = 10000

        #geographic position of the instrument
        self.lat = -29.2584    #latitude in degree
        self.lon = -70.7345    #longitude in degree
        self.alt = 2375    #altitude in m


        self.max_pixshift = 100
        self.reference_list_testorder    = 40


    def getMJD(self, header):
        #return observation MJD (mid of observation) as float
        try:
            return float(header['MJD'])
        except:
            return np.nan

    def setMJD(self, header, mjd):
        #set mjd in header
        try:
            header['MJD'] = mjd
        except:
            pass

        return header

    def getObjectName(self, header):
        #return name of target as starting
        try:
            return str(header['object'])
        except:
            return ''

    def setObjectName(self, header, name):
        #set object name in header
        try:
            header['object'] = name
        except:
            pass

        return header

    def getExptime(self, header):
        #return exptime in s
        try:
            return float(header['EXPTIME'])
        except:
            return np.nan

    def setExpTime(self, header, exptime):
        #set exptime in header
        try:
            header['EXPTIME'] = exptime
        except:
            pass

        return header

    def getStarCoordinates(self, header):
        #return RA (hourangle) and DEC (degree) of target star
        try:
            RA  = float(header['TELESCOPE RA'])
            DEC = float(header['TELESCOPE DEC'])

            return RA, DEC
        except:
            return np.nan, np.nan

    def setStarCoordinates(self, header, RA, DEC):
        #set RA und DEC in header
        try:
            header['TELESCOPE RA']  = RA
            header['TELESCOPE DEC'] = DEC
        except:
            pass

        return header

    def getTemperature(self, header):
        #return temperature
        try:
            return 0.5 * (float(header['HIERARCH ENVIRONMENT TEMP1'] + header['HIERARCH ENVIRONMENT TEMP5']))     #Fiber exit temperature, more or less arbirtarily chosen
        except:
            return 0

    def setTemperature(self, header, temp):
        #set temperature in header

        #not supported, skip
        return header

    def getExpMode(self, header):
        #return ObservationMode
        if header is None or 'HIERARCH OBSERVATION MODE' not in header:
            #return LIGHT if not known
            return Spectra.ObservationMode.LIGHT

        try:
            identifier = str(header['HIERARCH OBSERVATION MODE']).lower()

            if 'bias' in identifier:
                return Spectra.ObservationMode.BIAS
            elif 'dark' in identifier:
                return Spectra.ObservationMode.DARK
            elif 'flat' in identifier:
                return Spectra.ObservationMode.FLAT
            elif 'orderdef' in identifier:
                return Spectra.ObservationMode.ORDERDEF
            elif 'thar' in identifier:
                return Spectra.ObservationMode.THAR

            #return LIGHT, as identifier is most likely the targets name
            else:
                return Spectra.ObservationMode.LIGHT
        except:
            return None



class Atik383L(Camera):
    #this camera is used at the eShel instrument in Stumpertenrod
    #RON from https://www.atik-cameras.com/wp-content/uploads/2014/08/Atik383L-Manual.pdf, page 6
    def __init__(self):
        super().__init__()
        self.name = 'Atik 383L+'


    def get_gain(self, header):
        #returns camera gain
        return 0.2     #Measured from bias
        #return 0.41   #Official value from datasheet


    def get_RON(self, header):
        #returns camera read out noise (in electrons)
        return 7.

    def get_maxcount(self, header):
        #return maximal ADU count of camera.
        #Should be the maximal count where camera response is still linear, not necessarily the actual maximal value
        return 65535   #16 bit camera

    def get_badrows(self, header):
        #returns list of bad rows of this camera. These rows will be interpolated by using the neighboring rows.
        return None

class HARPS_camera_blue(Camera):
    # properties of the HARPS camera blue (Linda). Information from https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/manual/userman1_0.pdf, page 16
    def __init__(self):
        super().__init__()

        self.name = 'HARPS camera blue (Linda)'

    def get_gain(self, header):
        #returns camera gain
        try:
            if '416' in header['HIERARCH ESO DET READ SPEED']:
                return 1.4
            else:
                return 0.62
        except:
            return 0.62         #conservative estimate

    def get_RON(self, header):
        #returns camera read out noise (in electrons)
        try:
            if '416' in header['HIERARCH ESO DET READ SPEED']:
                return 5.5
            else:
                return 2.76
        except:
            return 5.5          #conservative estimate

    def get_maxcount(self, header):
        #return maximal ADU count of camera.
        #Should be the maximal count where camera response is still linear, not necessarily the actual maximal value
        return 65535   #16 bit camera

    def get_badrows(self, header):
        #returns list of bad rows of this camera. These rows will be interpolated by using the neighboring rows.
        return [107]


class HARPS_camera_red(Camera):
    # properties of the HARPS camera red (Jasmin). Information from https://www.eso.org/sci/facilities/lasilla/instruments/harps/doc/manual/userman1_0.pdf, page 16
    def __init__(self):
        super().__init__()

        self.name = 'HARPS camera red (Jasmin)'

    def get_gain(self, header):
        #returns camera gain
        try:
            if '416' in header['HIERARCH ESO DET READ SPEED']:
                return 1.42
            else:
                return 0.63
        except:
            return 0.63          #conservative estimate

    def get_RON(self, header):
        #returns camera read out noise (in electrons)
        try:
            if '416' in header['HIERARCH ESO DET READ SPEED']:
                return 7.05
            else:
                return 2.87
        except:
            return 7.05          #conservative estimate

    def get_maxcount(self, header):
        #return maximal ADU count of camera.
        #Should be the maximal count where camera response is still linear, not necessarily the actual maximal value
        return 65535   #16 bit camera

    def get_badrows(self, header):
        #returns list of bad rows of this camera. These rows will be interpolated by using the neighboring rows.
        return None



class FEROS_camera(Camera):
    #conservative estimate of the properties of the FEROS camera. Information from https://www.eso.org/sci/facilities/lasilla/instruments/feros/doc/manual/FEROSII-UserManual-1.4.pdf, page 20
    def __init__(self):
        super().__init__()

        self.name = 'FEROS camera'

    def get_gain(self, header):
        #returns camera gain
        try:
            if 'fast' in header['HIERARCH ESO DET READ SPEED'].lower():
                return 1./3.2
            else:
                return 1
        except:
            return 1./3.2          #conservative estimate

    def get_RON(self, header):
        #returns camera read out noise (in electrons)
        try:
            if 'fast' in header['HIERARCH ESO DET READ SPEED'].lower():
                return 5.1
            else:
                return 3.
        except:
            return 5.1          #conservative estimate

    def get_maxcount(self, header):
        #return maximal ADU count of camera.
        #Should be the maximal count where camera response is still linear, not necessarily the actual maximal value

        try:
            if 'fast' in header['HIERARCH ESO DET READ SPEED'].lower():
                return 40000        #faster readout leads to smaller saturation limit
            else:
                return 65000        #almost maximum of 16bit camera
        except:
            return 40000          #conservative estimate

    def get_badrows(self, header):
        #returns list of bad rows of this camera. These rows will be interpolated by using the neighboring rows.
        return [369, 375]




