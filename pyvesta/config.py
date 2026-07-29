"""
# This file contains the reduction_parameters object. This object contains all parameters needed for extraction that are not instrument-dependend (e.g. number of threads, where to store pictures etc.)
# For instrument specific parameters (such as the width of diffraction orders etc.) please see the instruments.py file
#
#
# Created by Lukas Stock, 06/26/26
#
"""

import os

class default_parameters:
    def __init__(self):
        # data parameters
        self.base_dir      = ''
        self.extracted_dir = ''

        ########################################################################################
        #processing parameters
        self.npools         = 4     # number of threads
        self.npools_extract = 2     # number of threads for extraction. Extraction is limited by memory performance, not CPU performance. Too many threads might even slow down extraction


        ########################################################################################
        #optional processing steps
        self.do_flatimage  = True   # do flat image (pixel wise) correction
        self.do_background = True   # do baclground subtraction

        ########################################################################################
        #plot parameters
        self.plot_dir   = os.path.join(self.extracted_dir, 'Plots')

        self.show_plots = False     # show plots interactively
        self.save_plots = True      # save plots

        self.plot_Ordertrace                    = True      #Plot Ordertraces
        self.plot_Fibersplit                    = False     #Plot result of split to single fibers

        self.plot_ThArPeaks                     = False     #Plot ThAr peak detection
        self.plot_ThArSigma                     = False     #Plot ThAr peak width calculation
        self.plot_ThArGoodPeaks                 = False     #Plot good ThAr peaks
        self.plot_ThArTiltFit                   = False     #Plot emission line tilt calculation
        self.plot_ThArTiltInterpolation         = False     #Plot emission line tilt fitting

        self.plot_Backgroundfit                 = False     #Plot background fit'
        self.plot_Weights                       = False     #Plot weights calculation
        self.plot_Ordershapes                   = False     #Plot ordershape calculation
        self.plot_FlatImage                     = False     #Plot flat image calculation

        self.plot_Extraction                    = False     #Plot extraction results
        self.plot_Veryfastextraction            = False     #Plot veryfast extraction results
        self.plot_speccosmicfilter              = False     #Plot cosmics filtering in extracted spectra

        self.plot_WavelengthFit                 = True      #Plot wavelength fit
        self.plot_WaveGlobalScale               = False     #Plot wavelength global scale calculation
        self.plot_WaveOverlaps                  = False     #Plot wavelength overlaps calculation
        self.plot_InitialSolution               = True      #Plot initial wavelength solution
        self.plot_WaveReference                 = True      #Plot reference wavelength solution
        self.plot_ThArGroupMedian               = True      #Plot grouped ThAr shifts

        self.plot_ContNormalization             = False     #Plot continuum normalization


class P14s_parameters(default_parameters):
    def __init__(self):
        super().__init__()

        self.npools = 12

        self.plot_Extraction                    = False
        self.plot_WaveGlobalScale               = True

        #disable flat image and background subtraction at the moment, as these cause errors at the extraction.
        #They lead to oscillations in the flat spectrum and the background gets overestimated at the edges of the image, disturbing the order merging
        self.do_flatimage  = False
        self.do_background = False

        #self.plot_WaveOverlaps                  = True

        """
        self.plot_Ordertrace                    = False
        self.plot_Ordertrace                    = False
        self.plot_Fibersplit                    = False

        self.plot_ThArPeaks                     = False
        self.plot_ThArSigma                     = False
        self.plot_ThArGoodPeaks                 = False
        self.plot_ThArTiltFit                   = False
        self.plot_ThArTiltInterpolation         = False

        self.plot_Backgroundfit                 = False
        self.plot_Weights                       = False
        self.plot_Ordershapes                   = False
        self.plot_FlatImage                     = False

        self.plot_Extraction                    = False
        self.plot_Veryfastextraction            = False
        self.plot_speccosmicfilter              = False

        self.plot_WavelengthFit                 = False
        self.plot_WaveGlobalScale               = False
        self.plot_WaveOverlaps                  = False
        self.plot_InitialSolution               = False
        self.plot_WaveReference                 = False
        self.plot_ThArGroupMedian               = False

        self.plot_ContNormalization             = False
        """
