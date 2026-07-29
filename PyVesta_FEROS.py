#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import sys
import argparse

import matplotlib
matplotlib.use('Agg')  # must be before importing pyplot

import warnings
warnings.filterwarnings('ignore')   #filter warnings

import logging
logging.getLogger().setLevel(logging.INFO)  #set logging status

from multiprocessing import freeze_support, set_start_method


from pyvesta import extraction
from pyvesta import CCD_corrections
from pyvesta import order_traces
from pyvesta import ThAr_Peak_detection
from pyvesta import wavelength_calibration
from pyvesta import instruments
from pyvesta import spectrum_operations
from pyvesta import Spectra
from pyvesta import config
from pyvesta import datashare

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


reduction_parameters = config.P14s_parameters()

reduction_parameters.base_dir  = ''     #directory with all files

#remove / at end
if reduction_parameters.base_dir[-1] == '/':
    reduction_parameters.base_dir = reduction_parameters.base_dir[:-1]

#name of output directory, maybe make this more adjustable
reduction_parameters.extracted_dir = reduction_parameters.base_dir + '_extracted'

#check if base directory exists
if not os.path.exists(reduction_parameters.base_dir):
    raise ValueError('Base directory does not exist!')

#name of output directory if not specified before
if reduction_parameters.extracted_dir == '':
    reduction_parameters.extracted_dir = reduction_parameters.base_dir + '_ext'

#initialize instrument
instrument = instruments.FEROS()

#initialize camera
camera = instruments.FEROS_camera()

#initialize data share with other files
#check if already initialized
if datashare.reduction_parameters is None:
    datashare.instrument           = instrument
    datashare.camera               = camera
    datashare.reduction_parameters = reduction_parameters


if __name__ == '__main__':
    freeze_support()
    set_start_method('spawn')

    import faulthandler
    faulthandler.enable()


    #initialize reduction parameters from config.py and command line parameters

    parser = argparse.ArgumentParser(description='PyVesta, a data reduction software for fiber-fed echelle spectrographs. Default values always as in config.py.')

    parser.add_argument('-base_dir'                  , type=str , default=datashare.reduction_parameters.base_dir                  , help='Directory with raw files')
    parser.add_argument('-extracted_dir'             , type=str , default=datashare.reduction_parameters.extracted_dir             , help='Directory to store extracted files')
    parser.add_argument('-npools'                    , type=int , default=datashare.reduction_parameters.npools                    , help='Number of threads')
    parser.add_argument('-npools_extract'            , type=int , default=datashare.reduction_parameters.npools_extract            , help='Number of threads for extraction')
    parser.add_argument('-show_plots'                , type=bool, default=datashare.reduction_parameters.show_plots                , help='Show plots interactively')
    parser.add_argument('-save_plots'                , type=bool, default=datashare.reduction_parameters.save_plots                , help='Save plots')
    parser.add_argument('-plot_Ordertrace'           , type=bool, default=datashare.reduction_parameters.plot_Ordertrace           , help='Plot Ordertraces')
    parser.add_argument('-plot_Fibersplit'           , type=bool, default=datashare.reduction_parameters.plot_Fibersplit           , help='Plot result of split to single fibers')
    parser.add_argument('-plot_ThArPeaks'            , type=bool, default=datashare.reduction_parameters.plot_ThArPeaks            , help='Plot ThAr peak detection')
    parser.add_argument('-plot_ThArSigma'            , type=bool, default=datashare.reduction_parameters.plot_ThArSigma            , help='Plot ThAr peak width calculation')
    parser.add_argument('-plot_ThArGoodPeaks'        , type=bool, default=datashare.reduction_parameters.plot_ThArGoodPeaks        , help='Plot good ThAr peaks')
    parser.add_argument('-plot_ThArTiltFit'          , type=bool, default=datashare.reduction_parameters.plot_ThArTiltFit          , help='Plot emission line tilt calculation')
    parser.add_argument('-plot_ThArTiltInterpolation', type=bool, default=datashare.reduction_parameters.plot_ThArTiltInterpolation, help='Plot emission line tilt fitting')
    parser.add_argument('-plot_Backgroundfit'        , type=bool, default=datashare.reduction_parameters.plot_Backgroundfit        , help='Plot background fit')
    parser.add_argument('-plot_Weights'              , type=bool, default=datashare.reduction_parameters.plot_Weights              , help='Plot weights calculation')
    parser.add_argument('-plot_Ordershapes'          , type=bool, default=datashare.reduction_parameters.plot_Ordershapes          , help='Plot ordershape calculation')
    parser.add_argument('-plot_FlatImage'            , type=bool, default=datashare.reduction_parameters.plot_FlatImage            , help='Plot flat image calculation')
    parser.add_argument('-plot_Extraction'           , type=bool, default=datashare.reduction_parameters.plot_Extraction           , help='Plot extraction results')
    parser.add_argument('-plot_Veryfastextraction'   , type=bool, default=datashare.reduction_parameters.plot_Veryfastextraction   , help='Plot veryfast extraction results')
    parser.add_argument('-plot_speccosmicfilter'     , type=bool, default=datashare.reduction_parameters.plot_speccosmicfilter     , help='Plot cosmics filtering in extracted spectra')
    parser.add_argument('-plot_WavelengthFit'        , type=bool, default=datashare.reduction_parameters.plot_WavelengthFit        , help='Plot wavelength fit')
    parser.add_argument('-plot_WaveGlobalScale'      , type=bool, default=datashare.reduction_parameters.plot_WaveGlobalScale      , help='Plot wavelength global scale calculation')
    parser.add_argument('-plot_WaveOverlaps'         , type=bool, default=datashare.reduction_parameters.plot_WaveOverlaps         , help='Plot wavelength overlaps calculation')
    parser.add_argument('-plot_InitialSolution'      , type=bool, default=datashare.reduction_parameters.plot_InitialSolution      , help='Plot initial wavelength solution')
    parser.add_argument('-plot_WaveReference'        , type=bool, default=datashare.reduction_parameters.plot_WaveReference        , help='Plot reference wavelength solution')
    parser.add_argument('-plot_ThArGroupMedian'      , type=bool, default=datashare.reduction_parameters.plot_ThArGroupMedian      , help='Plot grouped ThAr shifts')
    parser.add_argument('-plot_ContNormalization'    , type=bool, default=datashare.reduction_parameters.plot_ContNormalization    , help='Plot continuum normalization')

    args = parser.parse_args()

    datashare.reduction_parameters.base_dir                     = args.base_dir
    datashare.reduction_parameters.extracted_dir                = args.extracted_dir
    datashare.reduction_parameters.npools                       = args.npools
    datashare.reduction_parameters.npools_extract               = args.npools_extract
    datashare.reduction_parameters.show_plots                   = args.show_plots
    datashare.reduction_parameters.save_plots                   = args.save_plots
    datashare.reduction_parameters.plot_Ordertrace              = args.plot_Ordertrace
    datashare.reduction_parameters.plot_Fibersplit              = args.plot_Fibersplit
    datashare.reduction_parameters.plot_ThArPeaks               = args.plot_ThArPeaks
    datashare.reduction_parameters.plot_ThArSigma               = args.plot_ThArSigma
    datashare.reduction_parameters.plot_ThArGoodPeaks           = args.plot_ThArGoodPeaks
    datashare.reduction_parameters.plot_ThArTiltFit             = args.plot_ThArTiltFit
    datashare.reduction_parameters.plot_ThArTiltInterpolation   = args.plot_ThArTiltInterpolation
    datashare.reduction_parameters.plot_Backgroundfit           = args.plot_Backgroundfit
    datashare.reduction_parameters.plot_Weights                 = args.plot_Weights
    datashare.reduction_parameters.plot_Ordershapes             = args.plot_Ordershapes
    datashare.reduction_parameters.plot_FlatImage               = args.plot_FlatImage
    datashare.reduction_parameters.plot_Extraction              = args.plot_Extraction
    datashare.reduction_parameters.plot_Veryfastextraction      = args.plot_Veryfastextraction
    datashare.reduction_parameters.plot_speccosmicfilter        = args.plot_speccosmicfilter
    datashare.reduction_parameters.plot_WavelengthFit           = args.plot_WavelengthFit
    datashare.reduction_parameters.plot_WaveGlobalScale         = args.plot_WaveGlobalScale
    datashare.reduction_parameters.plot_WaveOverlaps            = args.plot_WaveOverlaps
    datashare.reduction_parameters.plot_InitialSolution         = args.plot_InitialSolution
    datashare.reduction_parameters.plot_WaveReference           = args.plot_WaveReference
    datashare.reduction_parameters.plot_ThArGroupMedian         = args.plot_ThArGroupMedian
    datashare.reduction_parameters.plot_ContNormalization       = args.plot_ContNormalization

    datashare.reduction_parameters.plot_dir   = os.path.join(datashare.reduction_parameters.extracted_dir, 'Plots')


    logging.info('\t ***** Welcome to PyVesta! *****')
    logging.info('\n\t We start a new extraction.')
    logging.info('\n\t Initialize file handlers.')



    #filenames of initial and final reference wavelength solution
    InitialSolution_filename   = os.path.join(datashare.reduction_parameters.extracted_dir, 'InitialSolution{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    ReferenceSolution_filename = os.path.join(reduction_parameters.extracted_dir, 'ReferenceSolution{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))

    #filames of CCD correction files
    masterbias_filename          = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterbias{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    masterflat_filename          = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterflat{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    masterflat_bkg_filename      = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterflat_bkg{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    masterflat_corr_filename     = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterflat_corr{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    flatimage_filename           = os.path.join(datashare.reduction_parameters.extracted_dir, 'Flatimage{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    masterorderdef_filename      = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterorderdef{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    masterorderdef_intp_filename = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterorderdef_badrowsinterpolated{}.fits'.format(datashare.instrument.filename_extensions_for_images))
    masterdark_filenames         = os.path.join(datashare.reduction_parameters.extracted_dir, 'Masterdark_{}s' + datashare.instrument.filename_extensions_for_images + '.fits') #cannot use format here, we want to keep the {} for later

    #filenames of traces and weights
    trace_filename             = os.path.join(datashare.reduction_parameters.extracted_dir, 'Traces{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    weights_filename           = os.path.join(datashare.reduction_parameters.extracted_dir, 'Weights{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))


    #Initialize file handlers, these take care of all files
    RawHandler            = Spectra.RawHandler(index=datashare.instrument.index)
    CCDCorrectedHandler   = Spectra.ExtractedHandler(suffix='_CCDcorr'        ,default_extension='{}.fits'.format(datashare.instrument.filename_extensions_for_images))  #save images
    BackgroundHandler     = Spectra.ExtractedHandler(suffix='_background'     ,default_extension='{}.fits'.format(datashare.instrument.filename_extensions_for_images)) #save images
    ExtractedHandler      = Spectra.ExtractedHandler(suffix='_ext'            ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))  #from here on save extracted spectra (hdf5 file extension)
    FlattedHandler        = Spectra.ExtractedHandler(suffix='_flat'           ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    CalibratedHandler     = Spectra.ExtractedHandler(suffix='_cal'            ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    FilteredHandler       = Spectra.ExtractedHandler(suffix='_filtered'       ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    MergedHandler         = Spectra.ExtractedHandler(suffix='_merged'         ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    ContnormHandler       = Spectra.ExtractedHandler(suffix='_contnorm'       ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    ContnormOrdersHandler = Spectra.ExtractedHandler(suffix='_contnorm_orders',default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    ContinuumHandler      = Spectra.ExtractedHandler(suffix='_continuum'      ,default_extension='{}.hdf5'.format(datashare.instrument.filename_extensions_for_images))
    #LinLogHandler        = Spectra.ExtractedHandler(suffix='_linlog')
    SolutionHandler       = Spectra.SolutionHandler(datashare.reduction_parameters.extracted_dir, suffix=datashare.instrument.filename_extensions_for_images)


    #get list of all files
    RawHandler.autoLoad(datashare.reduction_parameters.base_dir)

    logging.info('\n\t Create directories.')


    #create output directory
    ExtractedHandler.makeBaseDirectories(datashare.reduction_parameters.extracted_dir)
    base_directories = ExtractedHandler.getBaseDirectories()

    #create subdirectories
    ExtractedHandler.makeOwnDirectories()
    CCDCorrectedHandler.makeOwnDirectories(base_directories)
    BackgroundHandler.makeOwnDirectories(base_directories)
    FlattedHandler.makeOwnDirectories(base_directories)
    CalibratedHandler.makeOwnDirectories(base_directories)
    FilteredHandler.makeOwnDirectories(base_directories)
    MergedHandler.makeOwnDirectories(base_directories)
    ContnormHandler.makeOwnDirectories(base_directories)
    ContnormOrdersHandler.makeOwnDirectories(base_directories)
    ContinuumHandler.makeOwnDirectories(base_directories)
    #LinLogHandler.makeOwnDirectories(base_directories)

    SolutionHandler.makeDirectories()


    if datashare.reduction_parameters.save_plots:
        datashare.reduction_parameters.plot_dir = os.path.join(datashare.reduction_parameters.extracted_dir, 'Plots')

        if not os.path.exists(datashare.reduction_parameters.plot_dir):
            os.mkdir(datashare.reduction_parameters.plot_dir)

    #print all ThAr filenames to get an overview, just for debugging
    #print(RawHandler.filenames[Spectra.ObservationMode.THAR])

    #create masterbias and save or load, if it already exists

    logging.info('\n\t Creating Masterbias.')

    #set correct filename for plots etc.
    datashare.current_filename = 'Masterbias.fits'

    if not os.path.exists(masterbias_filename):
            master_bias = RawHandler.MedianCombine(Spectra.ObservationMode.BIAS)
            master_bias.save(masterbias_filename)
    else:
            master_bias = Spectra.Image.from_file(masterbias_filename, raw=False)

    #remove empty light files
    RawHandler.removeEmptyFiles(master_bias)

    #sort files by MJD
    RawHandler.sortbyMJD()

    #print ThAr filenames again, just for debugging
    #print(RawHandler.filenames[Spectra.ObservationMode.BIAS])
    #print(RawHandler.filenames[Spectra.ObservationMode.DARK])
    #print(RawHandler.filenames[Spectra.ObservationMode.FLAT])
    #print(RawHandler.filenames[Spectra.ObservationMode.ORDERDEF])
    #print(RawHandler.filenames[Spectra.ObservationMode.LIGHT])

    logging.info('\n\t Creating Masterflat.')

    #set correct filename for plots etc.
    datashare.current_filename = 'Masterflat.fits'

    #create masterflat and save or load, if it already exists
    if not os.path.exists(masterflat_filename):
            master_flat = RawHandler.MedianCombine(Spectra.ObservationMode.FLAT)
            master_flat -= master_bias             #substract masterbias

            #clip negative values
            #master_flat.clip_negatives()

            master_flat.save(masterflat_filename)
    else:
            master_flat = Spectra.Image.from_file(masterflat_filename, raw=False)


    #set correct filename for plots etc.
    datashare.current_filename = 'Masterorderdef.fits'

    #use masterflat if spectrograph does not have specific orderdef frames
    if datashare.instrument.Use_Flat_for_Orderdef:
        master_orderdef = master_flat
    else:
        logging.info('\n\t Creating Masterorderdef.')

        #create masterorderdef and save or load, if it already exists
        if not os.path.exists(masterorderdef_filename):
                master_orderdef = RawHandler.MedianCombine(Spectra.ObservationMode.ORDERDEF)
                master_orderdef -= master_bias                 #substract masterbias

                #clip negative values
                #master_orderdef.clip_negatives()

                master_orderdef.save(masterorderdef_filename)
        else:
                master_orderdef = Spectra.Image.from_file(masterorderdef_filename, raw=False)

    #calculate darks, if requested and darks are provided
    if datashare.instrument.use_darks and RawHandler.number_of(Spectra.ObservationMode.DARK) > 0:  #recreate them every time
        logging.info('\n\t Creating Masterdarks.')


        master_darks    = RawHandler.MedianCombine(Spectra.ObservationMode.DARK)     #remember: this is a dict!

        #set correct filename for plots etc.
        datashare.current_filename = 'Masterdark.fits'

        for exptime in master_darks.keys():
                master_darks[exptime] -= master_bias           #substract masterbias

                #clip negative values
                #master_darks[exptime].clip_negatives()


        master_darks = RawHandler.generateDarks(master_darks)

        for exptime in master_darks.keys():
                master_darks[exptime].save(masterdark_filenames.format(np.round(exptime, 2)))


    #set correct filename for plots etc.
    datashare.current_filename = 'Masterorderdef.fits'

    logging.info('\n\t Start ordertracing.')

    #create order traces and save or load, if they already exist
    if os.path.exists(trace_filename):
        Trace_data = Spectra.Trace_data.load(trace_filename)
    else:
        #trace orders

        #make copy of orderdef. We will interpolate dead rows, but we do not want to affect the original master orderdef, as it is used for weight calculation
        #the dead row will result in low weights there, therefore it will mask itself. Because of this we must not interpolate the dead rows for the weight calculation
        master_orderdef_tracing = master_orderdef.copy()

        master_orderdef_tracing = CCD_corrections.InterpolateBadRows(master_orderdef_tracing)

        master_orderdef_tracing.save(masterorderdef_intp_filename)

        Trace_data = order_traces.trace_orders(master_orderdef_tracing, nsigmas=20)


        #calculate line tilt, if necessary
        if datashare.instrument.use_linetilt:
            #use first ThAr frame to get tilt of extracction lines
            first_thar  = RawHandler.getFile(Spectra.ObservationMode.THAR, 0)
            first_thar -= master_bias  #substract masterbias

            #clip negative values
            #first_thar.clip_negatives()

            logging.info('\tstart veryfast extraction of ThAr frame for line tilt calculation')

            #use veryfast extraction. This spectrum will only be used to identify ThAr peaks and not for wavelength calibration
            thar_veryfast = extraction.veryfast_extract(first_thar, Trace_data)

            for fiber_nr in range(Trace_data.nr_of_fibers()):
                #get good ThAr peaks
                logging.info('\n\tstart ThAr peak fitting')
                peak_list = ThAr_Peak_detection.get_good_peaks(thar_veryfast[fiber_nr], approx_width=4)

                logging.info('\tstart ThAr tilt fitting')
                #find ThAr line tilts and save them in the trace object
                Trace_data.traces[fiber_nr] = ThAr_Peak_detection.fit_ThAr_lines(first_thar, Trace_data.traces[fiber_nr], peak_list, npools=1, degx=datashare.instrument.tiltfit_pixeldeg, degy=datashare.instrument.tiltfit_orderdeg)

            #only possible for deg = 0 (median). TODO: Autodetect this in the future!
            #Trace_data.traces[0] = ThAr_Peak_detection.interpolate_tilts(Trace_data.traces[0], nord=2, nsigmas=10)

        #save
        Trace_data.save(trace_filename)


    #calculate and substract background for masterflat. Save background in extra file.
    #If background corrected masterflat already exists just load it

    #set correct filename for plots etc.
    datashare.current_filename = 'Masterflat.fits'

    logging.info('\n\t start flat image calculation')

    if not os.path.exists(masterflat_corr_filename):
        if datashare.reduction_parameters.do_background:
            masterflat_background = CCD_corrections.CreateBackground(master_flat, Trace_data, degx=datashare.instrument.background_degx, degy=datashare.instrument.background_degy)
            masterflat_background.save(masterflat_bkg_filename)
            master_flat = master_flat - masterflat_background

        if datashare.reduction_parameters.do_flatimage:
            if not os.path.exists(flatimage_filename):
                flatimage = CCD_corrections.CreateFlatImage(master_flat, Trace_data)

                flatimage.save(flatimage_filename)
            else:
                flatimage = Spectra.Image.from_file(flatimage_filename, raw=False)

            master_flat = master_flat / flatimage

        #clip negative values
        #master_flat.clip_negatives()

        master_flat.save(masterflat_corr_filename)
    else:
        master_flat = Spectra.Image.from_file(masterflat_corr_filename, raw=False)

        if datashare.reduction_parameters.do_flatimage:
            if not os.path.exists(flatimage_filename):
                flatimage = CCD_corrections.CreateFlatImage(master_flat, Trace_data)

                flatimage.save(flatimage_filename)
            else:
                flatimage = Spectra.Image.from_file(flatimage_filename, raw=False)


    #calculate extraction weights or load them if they already exist
    if os.path.exists(weights_filename):
        logging.info('\n\tstart loading weights')
        start_time = time.time()
        #load weights
        Weights = Spectra.ExtractionWeights.load(weights_filename)
        logging.info('\tfinished loading after {}s'.format(np.around(time.time() - start_time, 2)))
    else:
        logging.info('\n\tstart calculating weights')
        start_time = time.time()
        #calculate weights

        if datashare.reduction_parameters.do_flatimage:
            master_orderdef /= flatimage

        Weights = extraction.create_weights_noSNR(master_orderdef, Trace_data)
        logging.info('\tfinished calculating weights after {}s'.format(np.around(time.time() - start_time, 2)))

        logging.info('\n\tstart saving weights')
        start_time = time.time()
        #save weights
        Weights.save(weights_filename)
        logging.info('\tfinished saving after {}s'.format(np.around(time.time() - start_time, 2)))

    #extract masterflat spectrum or load it, if it already exists

    extracted_masterflat_name = os.path.basename(masterflat_filename).replace('.hdf5','').replace('.fits','')

    if ExtractedHandler.exists(extracted_masterflat_name, Spectra.ObservationMode.FLAT):
        ExtractedHandler.logExisting(extracted_masterflat_name, Spectra.ObservationMode.FLAT)

        #load masterflat spectrum
        masterflat_spectrum = ExtractedHandler.loadExtracted(extracted_masterflat_name, Spectra.ObservationMode.FLAT)
    else:
        logging.info('\n\tstart extracting Masterflat')
        start_time = time.time()

        flat_file = master_flat

        #weighted extract masterflat
        #IMPORTANT: This is mostly limited by RAM speed, not CPU speed. Therefore limit npools to two, as more cores are actually slower due to more overhead
        masterflat_spectrum = extraction.weighted_extract_optimal(flat_file , Trace_data, Weights, npools=4, filter_cosmics=True)
        logging.info('\t -> finished extracting after {}s'.format(np.around(time.time() - start_time, 2)))

        #reverse order of diffraction orders if necessary, so that orders with smaller ordernumber have smaller wavelengths (depends on the orientation of the camera)
        if datashare.instrument.reverse_orders:
            masterflat_spectrum.reverse_orders()

        if datashare.instrument.reverse_pixels:
            masterflat_spectrum.reverse_pixels()

        logging.info('\tstart saving file {}'.format(extracted_masterflat_name))
        start_time = time.time()
        #save spectrum
        new_filename = ExtractedHandler.saveSpectrum(masterflat_spectrum, extracted_masterflat_name, frame_type = Spectra.ObservationMode.FLAT)
        logging.info('\t -> file saved to {}'.format(new_filename, np.around(time.time() - start_time, 2)))



    logging.info('\n\t start extracting all ThAr frames')


    #extract all ThAr files

    #total number of ThAr files
    total_number = RawHandler.number_of(Spectra.ObservationMode.THAR)

    #go thorugh each ThAr file
    for thar_nr in range(RawHandler.number_of(Spectra.ObservationMode.THAR)):
        #basename of file, will add appendix to this filename
        basename  = RawHandler.getBasename(Spectra.ObservationMode.THAR, thar_nr)

        #set correct filename for plots etc.
        datashare.current_filename = basename + '.fits'

        file_start_time = time.time()

        #extract ThAr frame or log that it already exists (we do not need to load it here)
        if ExtractedHandler.exists(basename, Spectra.ObservationMode.THAR):
            ExtractedHandler.logExisting(basename, Spectra.ObservationMode.THAR)
        else:
            #load image
            logging.info('\n\t start extracting ThAr file {}'.format(basename))

            thar_file  = RawHandler.getFile(Spectra.ObservationMode.THAR, thar_nr)

            #substract masterbias
            thar_file -= master_bias

            #clip negative values
            #thar_file.clip_negatives()

            if datashare.reduction_parameters.do_flatimage:
                #flat correction
                thar_file = thar_file / flatimage

            if datashare.reduction_parameters.do_background:
                #subtract background
                logging.info('\tstart calculating background for file {}'.format(basename))
                start_time      = time.time()
                basename        = RawHandler.getBasename(Spectra.ObservationMode.THAR, thar_nr)
                thar_background = CCD_corrections.CreateBackground(thar_file, Trace_data, degx=datashare.instrument.background_degx, degy=datashare.instrument.background_degy)
                BackgroundHandler.saveSpectrum(thar_background, basename, frame_type = Spectra.ObservationMode.THAR)
                logging.info('\t -> finished calculating background after {}s\n'.format(np.around(time.time() - start_time, 2)))

                #substract background
                thar_file -= thar_background

            #clip negative values
            #thar_file.clip_negatives()

            #weighted extract
            logging.info('\tstart extracting file {}'.format(basename))
            start_time = time.time()


            #IMPORTANT: This is mostly limited by RAM speed, not CPU speed. Therefore limit npools to two, as more cores are actually slower due to more overhead
            thar_spectra = extraction.weighted_extract_optimal(thar_file , Trace_data, Weights, filter_cosmics=False)
            logging.info('\t -> finished extracting after {}s'.format(np.around(time.time() - start_time, 2)))

            #reverse order of diffraction orders if necessary
            if datashare.instrument.reverse_orders:
                thar_spectra.reverse_orders()

            if datashare.instrument.reverse_pixels:
                thar_spectra.reverse_pixels()

            #save
            logging.info('\tstart saving file {}'.format(basename))
            start_time = time.time()
            new_filename = ExtractedHandler.saveSpectrum(thar_spectra, basename, frame_type = Spectra.ObservationMode.THAR)
            logging.info('\t -> file saved to {}'.format(new_filename))


        #correct for flat or just log, that flattened spectrum exists (we do not need to load it here)
        if FlattedHandler.exists(basename, Spectra.ObservationMode.THAR):
            FlattedHandler.logExisting(basename, Spectra.ObservationMode.THAR)
        else:
            #load spectrum
            if ExtractedHandler.exists(basename, Spectra.ObservationMode.THAR):
                thar_spectra = ExtractedHandler.loadExtracted(basename, Spectra.ObservationMode.THAR)

            #correct for masterflat
            flat_thar = thar_spectra / masterflat_spectrum

            #save
            new_filename = FlattedHandler.saveSpectrum(flat_thar, basename, frame_type = Spectra.ObservationMode.THAR)
            logging.info('\t -> Flattened file saved to {}'.format(new_filename))

        duration = time.time() - file_start_time
        logging.info('\n\t\t -> Finished file {} of {}'.format(thar_nr+1, total_number))
        logging.info('\t\t -> Extraction took {} second(s), will finish in approx. {} minute(s)\n\n'.format(np.around(duration, 0).astype(int), np.around(duration * (total_number - (thar_nr +1)) / 60.).astype(int)))


    logging.info('\n\t start wavelength calculation')

    #calculate reference wavelength solution or load it, if it already exists
    if SolutionHandler.SolutionExists(ReferenceSolution_filename, modify_filename=False):
        Reference_Solution = SolutionHandler.loadSolution(ReferenceSolution_filename, modify_filename=False, log=False)

        #m0 is the physical diffraction number of the first software order
        m0 = Reference_Solution.m0
    else:
        if datashare.instrument.peaklinelist is None or datashare.instrument.peaklinelist == '':
            #no instrument specific line list, use overlap wavelength solution

            logging.info('\t Start calculation of initial wavelength solution')

            #go through ThAr spectra till we find a good splution
            for thar_nr in range(RawHandler.number_of(Spectra.ObservationMode.THAR)):
                #use first ThAr image to get initial solution
                #if solution is bad, use next ThAr
                basename   = RawHandler.getBasename(Spectra.ObservationMode.THAR, thar_nr)

                #set correct filename for plots etc.
                datashare.current_filename = basename + '.fits'

                #get ThAr spectrum
                first_thar = FlattedHandler.loadExtracted(basename, Spectra.ObservationMode.THAR)

                #get m0 and initial wavelength solution
                m0, InitialSolution = wavelength_calibration.WaveSolutionFromUnknownm0(first_thar[0], datashare.instrument.reference_filename)

                #check whether wavelength solution is good enough
                if InitialSolution.rms < datashare.instrument.MaxWavesolution_RMS:
                    break


            thar_spectra = []
            basenames    = []

            #Now that we have an initial wavelength solution we can use that to calculate the wavelength solutions for all other ThAr frames
            for thar_nr in range(RawHandler.number_of(Spectra.ObservationMode.THAR)):
                #basename of ThAr file will also be the name of the wavelength solution
                basename  = RawHandler.getBasename(Spectra.ObservationMode.THAR, thar_nr)

                #set correct filename for plots etc.
                datashare.current_filename = basename + '.fits'

                #load ThAr spectrum and add it to list
                flat_thar = FlattedHandler.loadExtracted(basename, Spectra.ObservationMode.THAR)
                thar_spectra.append(flat_thar[0])
                basenames.append(basename)

            logging.info('\t Create wavelength solutions for all ThAr frames based on initial solution')
            #calculate wavelength solutions for all ThAr frames
            Solutions = wavelength_calibration.CalibrateFromInitialSolution(thar_spectra, InitialSolution, datashare.instrument.reference_filename, fixm0=True)

            #save solutions one by one
            for i in range(len(Solutions)):
                ThAr_Solution = Solutions[i]
                basename      = basenames[i]
                SolutionHandler.saveSolution(ThAr_Solution, basename)


        else:
            #use instrument specific peak list
            thar_spectra = []
            basenames    = []

            #Now that we have an initial wavelength solution we can use that to calculate the wavelength solutions for all other ThAr frames
            for thar_nr in range(RawHandler.number_of(Spectra.ObservationMode.THAR)):
                #basename of ThAr file will also be the name of the wavelength solution
                basename  = RawHandler.getBasename(Spectra.ObservationMode.THAR, thar_nr)

                #load ThAr spectrum and add it to list
                flat_thar = FlattedHandler.loadExtracted(basename, Spectra.ObservationMode.THAR)
                thar_spectra.append(flat_thar[0])
                basenames.append(basename)

            logging.info('\n\t Start calculation of wavelength solutions for all ThAr frames based on reference linelists')

            #calculate wavelength solutions for all ThAr frames
            Solutions = wavelength_calibration.CalibrateFromReferenceList(thar_spectra, datashare.instrument.peaklinelist, datashare.instrument.reference_filename)

            #save solutions one by one
            for i in range(len(Solutions)):
                ThAr_Solution = Solutions[i]
                basename      = basenames[i]
                SolutionHandler.saveSolution(ThAr_Solution, basename)

        #print m0, just for debugging
        logging.info('Calculated m0 is {}'.format(m0))

        #you can uncomment this to have an interactive plot
        #%matplotlib qt

        #plot radial velocity shift of wavelength solutions  over the night
        #wavelength_calibration.PlotThArDrift(Solutions)

        logging.info('\n\t Create reference wavelength solution')

        #Create a reference wavelength solution. This wavelength solution has an interpolator, so that it can interpolate the shift over the night
        Reference_Solution = SolutionHandler.makeReference(reference_filename=ReferenceSolution_filename, shift_method='cubic')

        #save the wavelength solution
        if not SolutionHandler.SolutionExists(ReferenceSolution_filename, modify_filename=False):
            SolutionHandler.saveSolution(Reference_Solution, ReferenceSolution_filename, modify_filename=False, log=False)

    #extract, flat, calibrate, merge and contnorm all light files
    #TODO: At the moment we do this only for the first fiber, also extract all other fibers!

    #create object which will take care of continuum normalization
    ContinuumFitter = Spectra.ContinuumFitter(mask_balmer=datashare.instrument.cont_mask_balmer, mask_tellurics=datashare.instrument.cont_mask_tellurics, telluric_atlas=datashare.instrument.tellurics_filename)

    #total number of light files
    total_number = RawHandler.number_of(Spectra.ObservationMode.LIGHT)

    calculated_previous = False

    logging.info('\n\t Start extracting all Science frames!')

    #go through all light files
    for light_nr in range(total_number):
        file_start_time = time.time()

        #all variants of this light file will have the same basename
        basename  = RawHandler.getBasename(Spectra.ObservationMode.LIGHT, light_nr)

        logging.info('\n\t Loading Science file {}'.format(basename))


        #set correct filename for plots etc.
        datashare.current_filename = basename + '.fits'

        #extract spectrum or log, that is already exists
        if ExtractedHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            ExtractedHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #get file
            light_file = RawHandler.getFile(Spectra.ObservationMode.LIGHT, light_nr)

            #filter cosmics
            light_file = CCD_corrections.filter_cosmics(light_file)

            #subtract masterbias
            light_file -= master_bias

            #try to subtract masterdark
            try:
                exptime = datashare.instrument.getExptime(light_file[0].header)

                if datashare.instrument.use_darks and exptime > 0 and not np.isnan(exptime) and exptime in master_darks.keys():
                    light_file -= master_darks[exptime]

            except:
                pass

            #clip negative values
            #light_file.clip_negatives()

            if datashare.reduction_parameters.do_flatimage:
                #flat correction
                light_file = light_file / flatimage

            if datashare.reduction_parameters.do_background:
                logging.info('\tstart calculating background for file {}'.format(basename))
                start_time = time.time()

                #calculate and save background
                light_background = CCD_corrections.CreateBackground(light_file, Trace_data, degx=datashare.instrument.background_degx, degy=datashare.instrument.background_degy)
                BackgroundHandler.saveSpectrum(light_background, basename, frame_type = Spectra.ObservationMode.LIGHT)
                logging.info('\t -> finished calculating background after {}s'.format(np.around(time.time() - start_time, 2)))

                #substract background
                light_file -= light_background

            #clip negative values
            #light_file.clip_negatives()

            #save CCD corrected file
            logging.info('\tstart saving file {}'.format(basename))
            start_time = time.time()
            new_filename = CCDCorrectedHandler.saveSpectrum(light_file, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> file saved to {}'.format(new_filename))

            #try to get MJD
            try:
                mjd = datashare.instrument.getMJD(light_file[0].header)

                if np.isnan(mjd):
                    mjd = 0
            except:
                mjd = 0

            #extract file
            logging.info('\tstart extracting file {}'.format(basename))
            start_time = time.time()
            #IMPORTANT: This is mostly limited by RAM speed, not CPU speed. Therefore limit npools to two, as more cores are actually slower due to more overhead
            light_spectra = extraction.weighted_extract_optimal(light_file , Trace_data, Weights, filter_cosmics=False)
            logging.info('\t -> finished extracting after {}s'.format(np.around(time.time() - start_time, 2)))

            #reverse order of diffraction orders if necessary
            if datashare.instrument.reverse_orders:
                light_spectra.reverse_orders()

            if datashare.instrument.reverse_pixels:
                light_spectra.reverse_pixels()

            #save
            logging.info('\tstart saving file {}'.format(basename))
            start_time = time.time()
            new_filename = ExtractedHandler.saveSpectrum(light_spectra, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> file saved to {}'.format(new_filename))

            calculated_previous = True


        #correct for flat or log, that flat corrected spectrum already exists
        if FlattedHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            FlattedHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #load spectrum
            if not calculated_previous:
                light_spectra = ExtractedHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #correct for masterflat
            flat_light = light_spectra / masterflat_spectrum

            #save
            new_filename = FlattedHandler.saveSpectrum(flat_light, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Flattened file saved to {}'.format(new_filename))

            calculated_previous = True

        #create wavelength calibrated spectrum or log, that it already exists
        if CalibratedHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            CalibratedHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #load flat corrected spectrum
            if not calculated_previous:
                flat_light = FlattedHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #apply wavelength solution
            calibrated_light = flat_light.applyWaveSolution(Reference_Solution)

            #convert to vacuum wavelengths
            #already done for HARPS, FEROS and eShel.
            #TODO: Put this in instruments file or provide just vacuum wavelengths?
            #calibrated_light.ToVacuum()


            #do barycentric correction
            BJDTDB, RV_corr = spectrum_operations.getBaryCorr(calibrated_light[0].header)

            #add values to header
            if not (np.isnan(BJDTDB) or np.isnan(RV_corr)):
                calibrated_light[0].header['HIERARCH BJDTDB'] = BJDTDB
                calibrated_light[0].header['HIERARCH BARYCORR km/s'] = RV_corr

                #apply barrycentric correction to wavelengths
                calibrated_light[0].applyShift(RV_corr)
            elif calibrated_light[0].header is not None:
                calibrated_light[0].header['HIERARCH BJDTDB'] = 0.
                calibrated_light[0].header['HIERARCH BARYCORR km/s'] = 0.

            #calculate SNR at 5130 angstrom (arbitraty value in the middle of the spectrum to get a rough value for the SNR of the spectrum)
            if calibrated_light[0].header is not None:
                SNR5130 = 0

                for order in range(calibrated_light[0].nr_of_orders()):
                    wave = calibrated_light[0][order].wave
                    inds = np.where((wave > 5128) & (wave < 5132))[0]

                    if len(inds) > 0:
                        SNR = np.nanmedian(calibrated_light[0][order].flux[inds] / calibrated_light[0][order].errors[inds])

                        if SNR > SNR5130:
                            SNR5130 = SNR

                if np.isnan(SNR5130) or SNR5130 < 0:
                        SNR5130 = 0

                #add value to header
                calibrated_light[0].header['SNR5130'] = SNR5130

            #save
            new_filename = CalibratedHandler.saveSpectrum(calibrated_light, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Calibrated file saved to {}'.format(new_filename))

            calculated_previous = True

        #create filtered spectrum or log, that it already exists
        if FilteredHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            FilteredHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #load wavelength calibrated spectrum
            if not calculated_previous:
                calibrated_light = CalibratedHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #merge orders
            filtered_light = calibrated_light.filter_outliners()

            #save
            new_filename = FilteredHandler.saveSpectrum(filtered_light, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Filtered file saved to {}'.format(new_filename))

            calculated_previous = True


        #create order merged spectrum or log, that it already exists
        if MergedHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            MergedHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #load filtered spectrum
            if not calculated_previous:
                filtered_light = FilteredHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #merge orders
            merged_light = filtered_light.mergeOrders()

            #save
            new_filename = MergedHandler.saveSpectrum(merged_light, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Merged file saved to {}'.format(new_filename))

            calculated_previous = True

        #create continuum normalized spectrum or log, that it already exists
        if ContnormHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            ContnormHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #load order merged spectrum
            if not calculated_previous:
                merged_light = MergedHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #fit continuum
            continuum = ContinuumFitter.fit(merged_light, lam1=1e4, lam2=1e5, niter=2)

            #devide by continuum if it is not None
            if continuum is not None:
                contnorm_light = merged_light / continuum
            else:
                contnorm_light = merged_light

            #save continuum normalized spectrum
            new_filename = ContnormHandler.saveSpectrum(contnorm_light, basename, frame_type = Spectra.ObservationMode.LIGHT)

            #save continuum if it is not None
            if continuum is not None:
                new_filename_continuum = ContinuumHandler.saveSpectrum(continuum, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Contnorm file saved to {}'.format(new_filename))

            calculated_previous = True

        #create continuum normalized spectrum on single orders or log, that it already exists
        if ContnormOrdersHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            ContnormOrdersHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            #load continuum spectrum
            if not calculated_previous:
                continuum = ContinuumHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #always load filtered spectrum
            filtered_light = FilteredHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            #continuum normalize single orders
            Orderscontnorm_light = filtered_light.contnorm_orders(continuum)

            #save orderwise continuum normalized spectrum
            new_filename = ContnormOrdersHandler.saveSpectrum(Orderscontnorm_light, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Orderscontnorm file saved to {}'.format(new_filename))

            calculated_previous = True

        duration = time.time() - file_start_time
        logging.info('\n\t\t -> Finished file {} of {}'.format(light_nr+1, total_number))
        logging.info('\t\t -> Extraction took {} second(s), will finish in approx. {} minute(s)\n\n'.format(np.around(duration, 0).astype(int), np.around(duration * (total_number - (light_nr +1)) / 60.).astype(int)))

        """
        if LinLogHandler.exists(basename, Spectra.ObservationMode.LIGHT):
            LinLogHandler.logExisting(basename, Spectra.ObservationMode.LIGHT)

            calculated_previous = False
        else:
            if ContnormHandler.exists(basename, Spectra.ObservationMode.LIGHT):
                contnorm_light = ContnormHandler.loadExtracted(basename, Spectra.ObservationMode.LIGHT)

            linlog_light = contnorm_light.loglinear_wave()

            new_filename = LinLogHandler.saveSpectrum(linlog_light, basename, frame_type = Spectra.ObservationMode.LIGHT)
            logging.info('\t -> Linlog file saved to {}'.format(new_filename))

            calculated_previous = True
        """

