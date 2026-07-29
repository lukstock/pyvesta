#TODO: Create comments!

import os
import numpy as np
from itertools import count

from pyvesta import Spectra

def  _wavelength_to_rgb(w):
    #rgb conversion from https://stackoverflow.com/questions/3407942/rgb-values-of-visible-spectrum
    #adapted to grey color for invisible ranges and converted to angstrom

    if w >= 3800 and w < 4000:  #from gray to violet
        R = 0.698 + (1. - 0.698) * (w - 3800.) / (4000. - 3800.)
        G = 0.745 + (0. - 0.745) * (w - 3800.) / (4000. - 3800.)
        B = 0.710 + (1. - 0.710) * (w - 3800.) / (4000. - 3800.)
    elif w >= 4000 and w < 4400:
        R = -(w - 4400.) / (4400. - 4000.)
        G = 0.0
        B = 1.0
    elif w >= 4400 and w < 4900:
        R = 0.0
        G = (w - 4400.) / (4900. - 4400.)
        B = 1.0
    elif w >= 4900 and w < 5100:
        R = 0.0
        G = 1.0
        B = -(w - 5100.) / (5100. - 4900.)
    elif w >= 5100 and w < 5800:
        R = (w - 5100.) / (5800. - 5100.)
        G = 1.0
        B = 0.0
    elif w >= 5800 and w < 6450:
        R = 1.0
        G = -(w - 6450.) / (6450. - 5800.)
        B = 0.0
    elif w >= 6450 and w < 7800:
        R = 1.0
        G = 0.0
        B = 0.0
    elif w >= 7800 and w < 8000:    #from red to gray
        R = 0.698 - (1. - 0.698) * (w - 8000.) / (8000. - 7800.)
        G = 0.745 - (0. - 0.745) * (w - 8000.) / (8000. - 7800.)
        B = 0.710 - (0. - 0.710) * (w - 8000.) / (8000. - 7800.)
    else:   #gray
        R = 0.698
        G = 0.745
        B = 0.710

    return np.array([R,G,B])

def _many_wavelengths_to_rgb(wavelengths):
    rgbs = np.zeros(shape=(3, len(wavelengths)))

    for i in range(len(wavelengths)):
        rgbs[:,i] = _wavelength_to_rgb(wavelengths[i])

    return rgbs

def colorize_image(image, Trace_data, SpectraList, reverse_orders = True, max_limit = 0.995):
    SpectraList = SpectraList.copy()

    if reverse_orders:
        SpectraList.reverse_orders()

    if Trace_data.nr_of_fibers() != SpectraList.nr_of_spectra():
        raise ValueError('Traces and spectra don\'t have the same length!')

    image_data = image.data

    #normalize data
    if max_limit <= 0 or max_limit > 1:
        max_limit = 1.

    max_value = np.sort(image_data.flatten())[int(max_limit * image_data.size)]

    image_data /= max_value
    image_data[image_data >= 1.] = 1.
    image_data[image_data <  0.] = 0.

    color_image = np.zeros(shape=(image_data.shape[0], image_data.shape[1], 3))
    grey_color = np.array([178,190,181])/255.

    for i in range(3):
        color_image[:,:,i] = image_data * grey_color[i]


    for fiber_nr in range(Trace_data.nr_of_fibers()):
        spectrum     = SpectraList[fiber_nr]
        fiber_traces = Trace_data.traces[fiber_nr]

        if not isinstance(spectrum, Spectra.Spectrum):
            continue

        if spectrum.nr_of_orders() != len(fiber_traces):
            raise ValueError('Traces and spectrum of fiber {} do not match!'.format(fiber_nr))


        for order_nr in range(spectrum.nr_of_orders()):
            order       = spectrum[order_nr]
            order_trace = fiber_traces.traces[order_nr]

            wavelengths   = order.wave

            order_trace.compute_centers(np.arange(len(wavelengths)))

            order_sigma   = order_trace.sigma
            order_centers = order_trace.Centers


            if len(wavelengths) != len(order_centers) or len(wavelengths) != image_data.shape[1]:
                raise ValueError('Invalid shape of wavelengths, centers and image!')

            if order_sigma < 0:
                order_sigma = 2

            order_width = np.around(3 * order_sigma).astype(int)

            for x in range(len(order_centers)):
                center = order_centers[x]
                w      = wavelengths[x]

                min_ind = np.max([0, center - order_width]).astype(int)
                max_ind = np.min([image_data.shape[0] -1, center + order_width]).astype(int)

                rgb = _wavelength_to_rgb(w)

                color_image[min_ind:max_ind,x,:] = np.outer(image_data[min_ind:max_ind, x], rgb)

    return color_image


def getnextfilename(folder, filename, extension):
    # return filename folder + filename_i + extension for smallest i that does not exist yet.
    for i in count(0):
        path = os.path.join(folder, filename + '_{}'.format(i) + extension)

        if not os.path.exists(path):
            return path

