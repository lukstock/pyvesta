import numpy as np
from scipy import interpolate

import sys
sys.path.append("..")

from pyvesta import Spectra
from pyvesta import wavelength_calibration
from pyvesta import ThAr_Peak_detection
from pyvesta import instruments

#FEROS
#ThAr_filename         = '../Copy/53068_extracted/ThAr/ext/FEROS.2004-03-03T03:07:07.216_ext.hdf5'      #filename of extracted (not yet wavelength calibrated) ThAr spectrum
#Wavesolution_filename = '../Copy/53068_extracted/ReferenceSolution.hdf5'
#Reference_filename    = '../reference/ESO_ThAr_List.txt'
#Out_filename          = 'Reference_FEROS.csv'
#instrument            = instruments.FEROS()

#eShel
ThAr_filename         = '../2025-03-05_extracted/ThAr/ext/ThAr_30.0s_2025-03-05T21-47-01_ext.hdf5'      #filename of extracted (not yet wavelength calibrated) ThAr spectrum
Wavesolution_filename = '../2025-03-05_extracted/ReferenceSolution.hdf5'
Reference_filename    = '../reference/ThAr-Audela.txt'
Out_filename          = 'Reference_eShel.csv'
instrument            = instruments.eShel()

#HARPS blue
#ThAr_filename         = '../HARPS_TestData/HIP60710/56338_extracted/ThAr/ext/HARPS.2013-02-14T10:28:43.408_ext_B.hdf5'      #filename of extracted (not yet wavelength calibrated) ThAr spectrum
#Wavesolution_filename = '../HARPS_TestData/HIP60710/56338_extracted/ReferenceSolution_B.hdf5'
#Reference_filename    = '../reference/ESO_ThAr_List.txt'
#Out_filename          = 'Reference_HARPS_blue.csv'
#instrument            = instruments.HARPS_blue()

#HARPS red
#ThAr_filename         = '../HARPS_TestData/HIP60710/56338_extracted/ThAr/ext/HARPS.2013-02-14T10:28:43.408_ext_R.hdf5'      #filename of extracted (not yet wavelength calibrated) ThAr spectrum
#Wavesolution_filename = '../HARPS_TestData/HIP60710/56338_extracted/ReferenceSolution_R.hdf5'
#Reference_filename    = '../reference/ESO_ThAr_List.txt'
#Out_filename          = 'Reference_HARPS_red.csv'
#instrument            = instruments.HARPS_red()



WaveSolution = Spectra.FinalWavelengthSolution.from_file(Wavesolution_filename)
thar_spectra = Spectra.SpectraList.load(ThAr_filename)

reference              = np.genfromtxt(Reference_filename)
#get nearest reference wavelength
reference_interpolator = interpolate.interp1d(reference, reference, kind='nearest', bounds_error=False, \
                         fill_value=(np.min(reference), np.max(reference)))


reference_orders = []
reference_pixels = []
reference_waves  = []


#just use first fiber
thar_spectrum = thar_spectra[0]

#peaks, peak_errs = ThAr_Peak_detection.find_ThAr_peaks(order, maxheight=instrument.camera.maxcount)
all_widths     = wavelength_calibration._getwidths(thar_spectrum, maxheight=0.9 * instrument.camera.maxcount)
_,  ThAr_Peaks = wavelength_calibration._getPeaks(thar_spectrum, all_widths, npools = 8, overlap_threshold=-1, all_threshold=1, maxheight=0.9 * instrument.camera.maxcount, plot=False)

all_peaks, all_orders = ThAr_Peaks.allPeaks()

#go through orders
for ordernr in range(thar_spectrum.nr_of_orders()):
    order = thar_spectrum[ordernr]

    phys_ordernr = ordernr + WaveSolution.m0

    peaks = all_peaks[all_orders == ordernr]

    peak_wavs = WaveSolution.eval_wavelengths(ordernr, peaks)

    reference_residuals = np.abs(peak_wavs - reference_interpolator(peak_wavs))

    good_peaks = peaks[np.where(reference_residuals <= instrument.wav_tol)]

    for peak in good_peaks:
        reference_wave = reference_interpolator(WaveSolution.eval_wavelengths(ordernr, peak))

        #check if there is already a reference with this reference wavelength in this order
        dublicate_inds = np.where((reference_orders == phys_ordernr) & (reference_waves == reference_wave))[0]

        if dublicate_inds.size == 0:
            #line not in list, append

            reference_orders.append(phys_ordernr)
            reference_pixels.append(np.round(peak).astype(int))
            reference_waves.append(reference_interpolator(WaveSolution.eval_wavelengths(ordernr, peak)))

        else:
            #compare residuals
            old_peak = reference_pixels[dublicate_inds[0]]       #there should be no dublicates in list, so just one other value should occur

            new_wave = WaveSolution.eval_wavelengths(ordernr, peak)
            old_wave = WaveSolution.eval_wavelengths(ordernr, old_peak)

            if np.abs(new_wave - reference_wave) < np.abs(old_wave - reference_wave):
                #new value is better. remove old value and add new one
                reference_orders.pop(dublicate_inds[0])
                reference_pixels.pop(dublicate_inds[0])
                reference_waves.pop(dublicate_inds[0])

                reference_orders.append(phys_ordernr)
                reference_pixels.append(np.round(peak).astype(int))
                reference_waves.append(reference_interpolator(WaveSolution.eval_wavelengths(ordernr, peak)))

            else:
                #old value is better, do nothing
                pass


reference_array = np.stack([reference_orders, reference_pixels, reference_waves]).T

print(reference_array.shape)
print(reference_array)

np.savetxt(Out_filename, reference_array, fmt='%i;%i;%.3f', delimiter=';', header='# phys. ordernr ; pixel ; wavelength(A)',comments='# ')




