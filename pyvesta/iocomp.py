######################################################################
# Here we define the input / output functionalities
#
#
#
######################################################################


import h5py                      #saving / loading .hdf5 files  (Help data / config / extracted spectra)
from astropy.io import fits      #saving / loading .fits files  (Images)
from astropy.table import QTable #astropy table
import os                        #folder structures
import numpy as np               #data structures

from pyvesta import Spectra

def None_filter(x):
    if x is None:
        return np.nan
    else:
        return x

#reverse None_filter
def NaN_filter(x, return_nan=False):
    if isinstance(x, str):
        return x

    elif np.isscalar(x[()]):
        if x.dtype == str:
            return str(x[()])
        elif x.dtype == int:
            return int(x[()])
        elif x.dtype == float:
            return float(x[()])
    else:
        try:
            arr = np.zeros(shape=x.shape)
            x.read_direct(arr)
               
            return arr
        except:        
            try:
                arr = np.zeros(shape=x.shape, dtype=object)
                x.read_direct(arr)
                   
                return arr
            except:        
                if return_nan:
                    return np.nan
                else:
                    return None


def _read_dataset(dataset):
    if len(dataset) < 1:
        return np.array([])

    arr = np.zeros(dataset.shape)
    dataset.read_direct(arr)
    
    return arr.astype(np.float32)

def save_traces(filename, Traces):
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
        
    if not isinstance(Traces, Spectra.Trace_data):
        raise ValueError('Trace must be a Spectra.Fiber_traces object!')
        
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
        
    if filename[-5:] != '.hdf5':
        filename += '.hdf5'
    
    
    hdf5_file = h5py.File(filename, 'w')
    
    data_group = hdf5_file.create_group('data')
    data_dst   = data_group.create_dataset('params', dtype='i')
    data_dst.attrs['nr_of_orders']          = None_filter(Traces.nr_of_orders)
    data_dst.attrs['order_multiplicity']    = None_filter(Traces.nr_of_fibers())
    data_dst.attrs['filename']              = None_filter(Traces.filename)
    data_dst.attrs['filename_of_spec']      = None_filter(Traces.filename_of_spec)
    data_dst.attrs['ron_flat']              = None_filter(Traces.ron_flat)
    data_dst.attrs['gain_flat']             = None_filter(Traces.gain_flat)
    data_dst.attrs['ron_bias']              = None_filter(Traces.ron_bias)
    data_dst.attrs['gain_bias']             = None_filter(Traces.gain_bias)
    
    
    #go through fibers
    for i in range(Traces.nr_of_fibers()):
        fiber_trace = Traces.traces[i]
        
        fiber_group = hdf5_file.create_group('fiber{}'.format(i))
        fiber_dst   = fiber_group.create_dataset('fiber_data', dtype='i')
        
        fiber_dst.attrs['nr_of_orders'] = None_filter(fiber_trace.nr_of_orders)
        fiber_dst.attrs['type']         = None_filter(fiber_trace.type)
        
        #go through traces of single orders:
        for order_nr, trace in enumerate(fiber_trace.all_traces()):
            order_group = fiber_group.create_group('order{}'.format(order_nr))
            
            order_dst   = order_group.create_dataset('order_data', dtype='i')
            order_dst.attrs['order_number'] = order_nr
            
            #convert order positions (list of pixels) to 2D numpy array
            #order_pos[pixel_nr, idx] with idx: 0: x value, 1: y value, 2: pixel value
            
            order_positions = [pix.to_array() for pix in trace.order_positions]
            order_positions = np.vstack(order_positions)


            coeff_dst     = order_group.create_dataset('pol_coefficients', data=None_filter(trace.pol_coefficients))
            pos_dst       = order_group.create_dataset('order_positions', data=None_filter(order_positions))
            maxpix_dts    = order_group.create_dataset('maxpix', data=None_filter(trace.maxpix))
            Centers_dst   = order_group.create_dataset('centers', data=None_filter(trace.Centers))
            pix_range_dst = order_group.create_dataset('pix_range', data=None_filter(trace.pix_range))
            tilt_dst      = order_group.create_dataset('tilt', data=None_filter(trace.tilt))
            sigma_dst     = order_group.create_dataset('sigma', data=None_filter(trace.sigma))

        
        
    #save
    hdf5_file.flush()
    hdf5_file.close()
        
def load_traces(filename):
    ################
    # Load a Spectra.Trace_data object from given file
    #
    #
    #
    ################

    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
        
    hdf5_file = h5py.File(filename, 'r')
    
    
    Trace_data = Spectra.Trace_data(filename=filename)
    
    data_dst   = hdf5_file['data']['params'] 
    Trace_data.nr_of_orders         = 0
    Trace_data.order_multiplicity   = NaN_filter(data_dst.attrs['order_multiplicity'])
    Trace_data.filename             = NaN_filter(data_dst.attrs['filename'])
    Trace_data.filename_of_spec     = NaN_filter(data_dst.attrs['filename_of_spec'])
    Trace_data.ron_flat             = NaN_filter(data_dst.attrs['ron_flat'])
    Trace_data.gain_flat            = NaN_filter(data_dst.attrs['gain_flat']) 
    Trace_data.ron_bias             = NaN_filter(data_dst.attrs['ron_bias'])
    Trace_data.gain_bias            = NaN_filter(data_dst.attrs['gain_bias'])
    
    #create Fiber traces
    for fiber_nr in range(Trace_data.order_multiplicity):
        fiber_trace = Spectra.Fiber_traces(filename=filename)
    
        fiber_group = hdf5_file['fiber{}'.format(fiber_nr)]
        fiber_dst   = fiber_group['fiber_data']
        
        fiber_trace.nr_of_orders    = 0
        fiber_trace.type            = NaN_filter(fiber_dst.attrs['type'])
        
        #load traces of single orders
        for order_nr in range(fiber_dst.attrs['nr_of_orders']):
            order_group = fiber_group['order{}'.format(order_nr)]
            
            Centers             = NaN_filter(order_group['centers'])
            pol_coefficients    = NaN_filter(order_group['pol_coefficients'])
            maxpix              = NaN_filter(order_group['maxpix'])
            order_positions     = NaN_filter(order_group['order_positions'])
            pix_range           = NaN_filter(order_group['pix_range'])
            tilt                = NaN_filter(order_group['tilt'])
            sigma               = NaN_filter(order_group['sigma'])

            order_positions = [Spectra.Pixel.from_array(arr) for arr in order_positions]
            
            trace = Spectra.Trace(pol_coefficients, order_positions, maxpix, pix_range=pix_range)
            trace.Centers   = Centers
            trace.tilt      = tilt
            trace.sigma     = sigma
            
            fiber_trace.add_trace_class(trace)
   
        if not fiber_trace.nr_of_orders == fiber_dst.attrs['nr_of_orders']:
            raise IOError('Error while creating fiber trace!')
            
        Trace_data.add_traces_obj(fiber_trace)
        
    if not Trace_data.nr_of_orders == data_dst.attrs['nr_of_orders']:
            raise IOError('Error while creating Trace_data object!')
    
    #close file
    hdf5_file.close()
    
    return Trace_data
    

def save_weights(filename, Weights):
    assert isinstance(Weights, Spectra.ExtractionWeights)
    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
    
    if filename[-5:] != '.hdf5':
        filename += '.hdf5'
    
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
        

    hdf5_file = h5py.File(filename, 'w')
    
    data_group = hdf5_file.create_group('data')
    data_dst   = data_group.create_dataset('params', dtype='i')
    data_dst.attrs['nr_of_fibers']          = None_filter(Weights.nr_of_fibers)
    data_dst.attrs['nr_of_orders']          = None_filter(Weights.nr_of_orders)
    data_dst.attrs['nr_of_pixels']          = None_filter(Weights.nr_of_pixels)    
    
    
    my_type = h5py.vlen_dtype(np.float32)
    #go through fibers
    for fiber_nr in range(Weights.nr_of_fibers):
        fiber_group = hdf5_file.create_group('fiber{}'.format(fiber_nr))
        fiber_dst   = fiber_group.create_dataset('data', dtype='i')

        fiber_dst.attrs['fiber_nr']     = fiber_nr
        
        #go through orders
        for order_nr in range(Weights.nr_of_orders):
            order_group = fiber_group.create_group('order{}'.format(order_nr))
            order_dst   = order_group.create_dataset('data', dtype='i')

            order_dst.attrs['order_nr']     = order_nr
            
            #boundaries must have the same shape for all pixels, the weights not!
            order_boundaries = Weights.get_boundaries_order(fiber_nr, order_nr)
            ordershape       = Weights.get_ordershape(fiber_nr, order_nr)
            shapes = []
            
            weights_dst = order_group.create_dataset('weights', (Weights.nr_of_pixels,), dtype=my_type)  

            for x in range(Weights.nr_of_pixels):    
                weights =  Weights.get_weights(fiber_nr, order_nr, x)
                shapes.append(weights.shape)
                weights_dst[x] = weights.ravel()


                
            shapes = np.vstack(shapes)
            shapes_dst     = order_group.create_dataset('shapes', data=shapes)
            boundaries_dst = order_group.create_dataset('boundaries', data=order_boundaries)
            ordershape_dst = order_group.create_dataset('ordershape', data=ordershape)
        
    #save
    hdf5_file.flush()
    hdf5_file.close()
    

def load_weights(filename):    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    hdf5_file  = h5py.File(filename, 'r')
    data_group = hdf5_file['data']
    data_dst   = data_group['params']
    
    nr_of_fibers = NaN_filter(data_dst.attrs['nr_of_fibers'])
    nr_of_orders = NaN_filter(data_dst.attrs['nr_of_orders'])
    nr_of_pixels = NaN_filter(data_dst.attrs['nr_of_pixels'])
    
    Weights = Spectra.ExtractionWeights(nr_of_fibers, nr_of_orders, nr_of_pixels)
    
    
    #go thorugh fibers
    for fiber_nr in range(Weights.nr_of_fibers):
        fiber_group = hdf5_file['fiber{}'.format(fiber_nr)]
        #go through orders
        for order_nr in range(Weights.nr_of_orders):
            order_group = fiber_group['order{}'.format(order_nr)]
            
            boundaries = NaN_filter(order_group['boundaries'])
            ordershape = NaN_filter(order_group['ordershape'])
            shapes     = NaN_filter(order_group['shapes']).astype(int)
            
            w          = NaN_filter(order_group['weights'])
            
            if w is not None:
                for x in range(shapes.shape[0]):
                    w[x] = w[x].reshape(shapes[x]).astype(np.float32)

                Weights.set_weights_order(fiber_nr, order_nr, w) 
                Weights.set_boundaries_order(fiber_nr, order_nr, boundaries)
                Weights.set_ordershape(fiber_nr, order_nr, ordershape)
            
    #close file
    hdf5_file.close()
    
    return Weights
     
def save_spectra_hdf5(filename, spectralist):
    assert isinstance(spectralist, Spectra.SpectraList)
    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
    
    if filename[-5:] != '.hdf5':
        filename += '.hdf5'
    
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
    
    #remove previous file
    if os.path.exists(filename):
        os.remove(filename)
    
    
    try:
        hdf5_file = h5py.File(filename, 'w')
        
        data_group = hdf5_file.create_group('data')
        data_dst   = data_group.create_dataset('params', dtype='i')
        data_dst.attrs['nr_of_spectra'] = spectralist.nr_of_spectra()
        
        #go through spectra
        for spectrum_nr in range(spectralist.nr_of_spectra()):
            spec = spectralist[spectrum_nr]
        
            spectrum_group = hdf5_file.create_group('spectrum{}'.format(spectrum_nr))
            spectrum_dst   = spectrum_group.create_dataset('params', dtype='i')   #empty dataset, only for attributes

            spectrum_dst.attrs['spectrum_nr']  = spectrum_nr
            spectrum_dst.attrs['nr_of_orders'] = spec.nr_of_orders()
            
            if isinstance(spec, Spectra.Spectrum):
                spectrum_dst.attrs['reference']    = spec.reference
            
            header_dst = spectrum_group.create_dataset('header', dtype='i')  #empty dataset, only for attributes
            
            if spec.header is not None:
                for key in spec.header.keys():
                    header_dst.attrs[key] = str(spec.header[key])
            
            if isinstance(spec, Spectra.RawSpectrum):
                spectrum_dst.attrs['type'] = 'raw'
            else:
                spectrum_dst.attrs['type'] = 'calibrated'
            
            #go through orders
            for order_nr in range(spec.nr_of_orders()):
                order_group = spectrum_group.create_group('order{}'.format(order_nr))
                order_dst   = order_group.create_dataset('params', dtype='i')    #empty dataset, only for attributes
                
                order_dst.attrs['order_nr']     = order_nr
                
                order = spec[order_nr]
                
                flux_dst    = order_group.create_dataset('flux', data=order.flux)
                err_dst     = order_group.create_dataset('errors', data=order.errors)           
                
                if isinstance(order, Spectra.RawSpectralOrder):
                    order_dst.attrs['type'] = 'raw'
                    
                    pix_dst     = order_group.create_dataset('pixels', data=order.pixels)
                    wav_dst     = order_group.create_dataset('wavelength', dtype='i')   #empty dataset
                else:
                    order_dst.attrs['type'] = 'calibrated'
                    
                    pix_dst     = order_group.create_dataset('pixels', dtype='i')       #empty dataset
                    wav_dst     = order_group.create_dataset('wavelength', data=order.wave)  
    
    finally:
        #save
        hdf5_file.flush()
        hdf5_file.close()
    
    
def load_spectra_hdf5(filename):    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    hdf5_file  = h5py.File(filename, 'r')
    data_group = hdf5_file['data']
    data_dst   = data_group['params']
    
    nr_of_spectra = data_dst.attrs['nr_of_spectra']
    
    spectralist = Spectra.SpectraList([])
       
       
    #go thorugh spectra
    for spectrum_nr in range(nr_of_spectra):
        spectrum_group = hdf5_file['spectrum{}'.format(spectrum_nr)]
        spectrum_dst   = spectrum_group['params']
        
        nr_of_orders  = spectrum_dst.attrs['nr_of_orders']
        spectrum_type = spectrum_dst.attrs['type']
        

        
        if spectrum_type == 'raw':
            spectrum = Spectra.RawSpectrum()
        else:
            spectrum = Spectra.Spectrum()
            spectrum.reference = spectrum_dst.attrs['reference'] if 'reference' in spectrum_dst.attrs.keys() else 'AIR'
        
        header = {}
        header_dst = spectrum_group['header']
                
        for key in header_dst.attrs.keys():
            header[key] = header_dst.attrs[key]
        
        if len(header) == 0:
            header = None
            
        spectrum.header = header    
        
        #go through orders
        for order_nr in range(nr_of_orders):
            order_group = spectrum_group['order{}'.format(order_nr)]
            order_dst   = order_group['params']
            
            order_type = order_dst.attrs['type']
            
            flux_dst = order_group['flux']
            err_dst  = order_group['errors']
            wav_dst  = order_group['wavelength']
            pix_dst  = order_group['pixels']
            
            if order_type == 'raw':                      
                flux = _read_dataset(flux_dst)
                errs = _read_dataset(err_dst)
                pix  = _read_dataset(pix_dst)
                
                order = Spectra.RawSpectralOrder(pix, flux, errors=errs)
            
            else:
                flux = _read_dataset(flux_dst)
                errs = _read_dataset(err_dst)
                wave  = _read_dataset(wav_dst)
                
                order = Spectra.SpectralOrder(wave, flux, errors=errs)
            
            spectrum.addOrder(order)

        spectralist.addSpectrum(spectrum)
        
    #close file
    hdf5_file.close()
    
    return spectralist   
    
def save_spectra_fits(filename, spectralist):
    assert isinstance(spectralist, Spectra.SpectraList)
    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
    
    if filename[-5:] != '.fits':
        filename += '.fits'
    
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
    
    #remove previous file
    if os.path.exists(filename):
        os.remove(filename)
    
    
    try:
        PrimaryHDU = fits.PrimaryHDU()
        
        PrimaryHDU.header['nr_of_spectra'] = (spectralist.nr_of_spectra(), 'Number of spectra')
        
        hdus = []
        hdus.append(PrimaryHDU)
        
        #go through spectra
        for spectrum_nr in range(spectralist.nr_of_spectra()):
            spec = spectralist[spectrum_nr]
        
        
            orders = np.array(dtype=np.int)
            pixels = np.array(dtype=np.int)
            wavs   = np.array(dtype=float)
            flux   = np.array(dtype=float)
            errs   = np.array(dtype=float)
            
            #go through orders
            for order_nr in range(spec.nr_of_orders()):
                order = spec[order_nr]
            
                orders = np.append(orders, np.ones_like(order.flux) * order_nr)
                flux   = np.append(flux,   order.flux)
                errs   = np.append(errs,   order.errors)
                        
                
                if isinstance(order, Spectra.RawSpectralOrder):                    
                    pixels = np.append(pixels, order.pixels)
                    wavs   = np.append(wavs  , np.zeros_like(order.flux))   #no information
                else:
                    pixels = np.append(pixels, np.zeros_like(order.flux))   # no information
                    wavs   = np.append(wavs  , order.wave)      
        
        
        
            HDU = fits.BinTableHDU.from_columns(    
                    [fits.Column(name='order'     , array=orders),
                    fits.Column(name='pixels'     , array=pixels),
                    fits.Column(name='wavelengths', array=wavs),
                    fits.Column(name='flux'       , array=flux),
                    fits.Column(name='flux_errors', array=errs)])
            
            
            #modify header
            HDU.header['spectrum_nr']  = (spectrum_nr, 'Spectrum number')
            HDU.header['nr_of_orders'] = (spec.nr_of_orders(), 'Number of orders')
            
            
            if spec.header is not None:
                for key in spec.header.keys():
                    HDU.header[key] = str(spec.header[key])
            
            if isinstance(spec, Spectra.RawSpectrum):
                HDU.header['reference'] = ('pixels', 'reference frame')
                HDU.header['type']      = 'raw'
            else:
                HDU.header['reference'] = (spec.reference, 'reference frame')
                HDU.header['type']      = 'calibrated'
            
            hdus.append(HDU)
           
        HDUList = fits.HDUList(hdus)
        HDUList.writeto(filename)
    
    finally:
        #do nothing
        pass
    
    
def load_spectra_fits(filename):    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    
    try:
        with fits.open(filename) as HDUList:
            nr_of_spectra = int(HDUList[0].header['nr_of_spectra'])
            
            assert len(HDUList) > nr_of_spectra
            
            spectralist = Spectra.SpectraList([])
               
            #go thorugh spectra
            for spectrum_nr in range(nr_of_spectra):   
                hdu = HDUList[spectrum_nr + 1]
            
                nr_of_orders  = int(hdu.header['nr_of_orders'])
                spectrum_type = str(hdu.header['type'])
                               
                if spectrum_type == 'raw':
                    spectrum = Spectra.RawSpectrum()
                else:
                    spectrum = Spectra.Spectrum()
                    spectrum.reference = spectrum_dst.attrs['reference'] if 'reference' in spectrum_dst.attrs.keys() else 'AIR'
                
                header = {}
                        
                for key in hdu.header.keys():
                    header[key] = hdu.header[key]
                
                if len(header) == 0:
                    header = None
                    
                spectrum.header = header    
                
                orders = np.array(hdu.data['order'])
                pixels = np.array(hdu.data['pixels'])
                wavs   = np.array(hdu.data['wavelengths'])
                flux   = np.array(hdu.data['flux'])
                errs   = np.array(hdu.data['flux_errors'])
                                
                #go through orders
                for order_nr in range(nr_of_orders):
                    good_inds = np.where(orders == order_nr)[0]
                    
                    order_flux = flux[good_inds]
                    order_errs = errs[good_inds]
                    
                    #calibrated
                    if np.all(pixels[good_inds] == 0):
                        order_wave = wavs[good_inds]
                        
                        order = Spectra.SpectralOrder(order_wave, order_flux, errors=order_errs)
                    
                    else:
                        order_pixels = pixels[good_inds]
                        
                        order = Spectra.RawSpectralOrder(order_pixels, order_flux, errors=order_errs)
                    
                    spectrum.addOrder(order)

                spectralist.addSpectrum(spectrum)
                
            return spectralist   

    except:
        return None

def load_header_hdf5(filename):    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    hdf5_file  = h5py.File(filename, 'r')
    data_group = hdf5_file['data']
    data_dst   = data_group['params']
    
    nr_of_spectra = data_dst.attrs['nr_of_spectra']
    
    headerlist = []       
       
    #go thorugh spectra
    for spectrum_nr in range(nr_of_spectra):
        spectrum_group = hdf5_file['spectrum{}'.format(spectrum_nr)]
        
        header = {}
        header_dst = spectrum_group['header']
                
        for key in header_dst.attrs.keys():
            header[key] = header_dst.attrs[key]
        
        if len(header) == 0:
            header = None
            
        headerlist.append(header)
            
    #close file
    hdf5_file.close()
    
    return headerlist  

def load_header_fits(filename):    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    headerlist = []  
    
    with fits.open(filename) as HDUList:      
        nr_of_spectra = int(HDUList[0].header['nr_of_spectra'])
        
        assert len(HDUList) > nr_of_spectra     
           
        #go thorugh spectra
        for spectrum_nr in range(nr_of_spectra):
            hdu = HDUList[spectrum_nr +1]
            
            header = {}
                    
            for key in hdu.header.keys():
                header[key] = hdu.header[key]
            
            if len(header) == 0:
                header = None
                
            headerlist.append(header)
    
    return headerlist  

def save_image_hdf5(filename, image):
    assert isinstance(image, Spectra.Image)
    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
    
    if filename[-5:] != '.hdf5':
        filename += '.hdf5'
    
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
       
    #remove previous file
    if os.path.exists(filename):
        os.remove(filename)   
       
    try:
        hdf5_file = h5py.File(filename, 'w')
        
        data_group = hdf5_file.create_group('data')
        params_dst = data_group.create_dataset('params', dtype='i')
        params_dst.attrs['gain'] = image.gain
        params_dst.attrs['RON']  = image.RON
        
        header_dst = spectrum_group.create_dataset('header', dtype='i')  #empty dataset, only for attributes
            
        if spec.header is not None:
            for key in spec.header.keys():
                header_dst[key] = spec.header[key]
        
        data_dst   = data_group.create_dataset('data', data=image.data.astype(np.float32))
        errors_dst = data_group.create_dataset('errors', data=image.errors.astype(np.float32))
         
    finally:
        #save
        hdf5_file.flush()
        hdf5_file.close()

def save_image_fits(filename, image):
    assert isinstance(image, Spectra.Image)
    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
    
    if filename[-5:] != '.fits':
        filename += '.fits'
    
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
      
    #remove previous file
    if os.path.exists(filename):
        os.remove(filename)  
      

    if image.errors is None:
        save_data = image.data.astype(np.float32) #32 bit is enough, no need for 64 bit (normal float)
    else:
        save_data = np.zeros(shape=(2, image.data.shape[0], image.data.shape[1]))
        save_data[0, :, :] = image.data
        save_data[1, :, :] = image.errors
        save_data = save_data.astype(np.float32)


    data_hdu   = fits.PrimaryHDU(data=save_data)

    if image.header is not None:
        data_hdu.header = image.header
    
    
    data_hdu.header['BITPIX']   = 32
    data_hdu.header['BZERO']    = 0
    
    hdul = fits.HDUList([data_hdu])
    
    hdul.writeto(filename)
    hdul.close()


def getFitsHeader(filename):
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
        
    return fits.getheader(filename)


def image_from_file_fits(filename, transpose=False, index = 0):
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))

    with fits.open(filename) as hdu:  
        if index >= len(hdu):
            index = 0

        while index < len(hdu):
            data = hdu[index].data

            if data is not None:
                break
            else:
                index += 1



        data   = hdu[index].data.astype(np.float32)

        if len(data.shape) == 2:
            image  = data
            errors = None
        elif len(data.shape) == 3:
            image  = data[0,:,:]
            errors = data[1,:,:]

        #always get first header, if that is None use header of image
        header = hdu[0].header

        if header is None:
            header = hdu[index].header

        #data = hdu[0].data.astype(np.float32)
        #errors = None if len(hdu) < 2 else hdu[1].data.astype(np.float32)
        #header = hdu[0].header

    if transpose:
        if image is not None:
            image   = data.T
        if errors is not None:
            errors = errors.T
    
    return image, errors, header
    
def image_from_file_hdf5(filename):
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    hdf5_file  = h5py.File(filename, 'r')
    data_group = hdf5_file['data']
    params_dst = data_group['params']
       
    gain = params_dst.attrs['gain']
    RON  = params_dst.attrs['RON']
    
    header = {}
    header_dst = spectrum_group['header']
                
    for key in header_dst.attrs.keys():
        header[key] = header_dst.attrs[key]
        
    if len(header) == 0:
        header = None
        
    data_dst = data_group['data']
    errs_dst = data_group['errors']
    
    data  = np.zeros(shape=data_dst.shape)
    errs  = np.zeros(shape=errs_dst.shape)
    
    data_dst.read_direct(data)
    errs_dst.read_direct(errs)
    
    data = data.astype(np.float32)
    errs = errs.astype(np.float32)
    
    #close file
    hdf5_file.close()
    
    return data, errs, gain, RON, header

def save_final_wavesolution(filename, FinalWaveSolution):
    assert isinstance(FinalWaveSolution, Spectra.FinalWavelengthSolution)
    
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if filename[0] != '/' and filename[:2] != './':
        filename = './' + filename
    
    if filename[-5:] != '.hdf5':
        filename += '.hdf5'
    
    #check that parent directory of filename exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('directory {} does\'nt exists!'.format(os.path.dirname(filename)))
      
    #remove previous file
    if os.path.exists(filename):
        os.remove(filename)  

    try:
        hdf5_file = h5py.File(filename, 'w')
        
        params_group = hdf5_file.create_group('params')
        params_dst   = params_group.create_dataset('params', dtype='i')
        params_dst.attrs['m0']               = FinalWaveSolution.m0
        params_dst.attrs['min_pix']          = FinalWaveSolution.min_pix
        params_dst.attrs['max_pix']          = FinalWaveSolution.max_pix
        params_dst.attrs['min_order']        = FinalWaveSolution.min_order
        params_dst.attrs['max_order']        = FinalWaveSolution.max_order
        params_dst.attrs['min_useful_order'] = FinalWaveSolution.min_useful_order
        params_dst.attrs['max_useful_order'] = FinalWaveSolution.max_useful_order
        params_dst.attrs['nord']             = FinalWaveSolution.nord
        params_dst.attrs['npix']             = FinalWaveSolution.npix
        params_dst.attrs['mjd']              = FinalWaveSolution.mjd
        params_dst.attrs['rms']              = FinalWaveSolution.rms
        params_dst.attrs['final_rms']        = FinalWaveSolution.final_rms if FinalWaveSolution.final_rms is not None else -1
        

        header_dst = params_group.create_dataset('header', dtype='i')  #empty dataset, only for attributes

        if FinalWaveSolution.ThAr_header is not None:
            for key in FinalWaveSolution.ThAr_header.keys():
                header_dst.attrs[key] = str(FinalWaveSolution.ThAr_header[key])

        lines_table = np.zeros(shape=(len(FinalWaveSolution.lines_matched), 3)).astype(np.float32)
        
        for i, line in enumerate(FinalWaveSolution.lines_matched):
            lines_table[i, :] = [line.order, line.pixel, line.wavelength]
        
        all_peaks, all_orders = FinalWaveSolution.allPeaks.allPeaks()
        widths                = FinalWaveSolution.allPeaks.getWidths()
        peaks = np.stack([all_orders, all_peaks])
        
        data_group = hdf5_file.create_group('data')
        coeffs_dst   = data_group.create_dataset('coeffs', data=FinalWaveSolution.coeffs)
        atlas_dst    = data_group.create_dataset('lineatlas', data=FinalWaveSolution.lineatlas) 
        matched_dst  = data_group.create_dataset('matched_lines', data=lines_table) 
        peaks_dst    = data_group.create_dataset('all_peaks', data=peaks) 
        widths_dst   = data_group.create_dataset('peak_widths', data=widths) 
        
    finally:
        #save
        hdf5_file.flush()
        hdf5_file.close()


def load_final_wavesolution(filename):
    if not type(filename) is str:
        try:
            filename = str(filename)
        except:
            raise ValueError('filename must be a string!')
        
    if not os.path.exists(filename):
        raise ValueError('File {} doesn\'t exists!'.format(filename))
    
    hdf5_file  = h5py.File(filename, 'r')
    
    params_group = hdf5_file['params']
    params_dst   = params_group['params']
    m0               = params_dst.attrs['m0']
    min_pix          = params_dst.attrs['min_pix']
    max_pix          = params_dst.attrs['max_pix']
    min_order        = params_dst.attrs['min_order']
    max_order        = params_dst.attrs['max_order']
    min_useful_order = params_dst.attrs['min_useful_order']
    max_useful_order = params_dst.attrs['max_useful_order']
    nord             = params_dst.attrs['nord']
    npix             = params_dst.attrs['npix']
    mjd              = params_dst.attrs['mjd']
    rms              = params_dst.attrs['rms']
    final_rms        = params_dst.attrs['final_rms']
    
    if final_rms < 0:
        final_rms = None
    
    data_group = hdf5_file['data']
    coeffs_dst  = data_group['coeffs']
    atlas_dst   = data_group['lineatlas']
    matched_dst = data_group['matched_lines']
    peaks_dst    = data_group['all_peaks']
    widths_dst   = data_group['peak_widths']
    

    coeffs  = _read_dataset(coeffs_dst)
    atlas   = _read_dataset(atlas_dst)
    matched = _read_dataset(matched_dst)
    peaks   = _read_dataset(peaks_dst)
    widths  = _read_dataset(widths_dst)
    
    
    assert coeffs.shape == (nord+1, npix+1)
    assert matched.shape[1] == 3
    assert peaks.shape[0] == 2
    
    peaks, orders = peaks[1, :], peaks[0,:]
    
    ThAr_Peaks = Spectra.ThArPeaks()
    ThAr_Peaks.setWidths(widths)
    
    for order in np.unique(orders):
        inds = np.where(orders == order)[0]
        
        ThAr_Peaks.addPeaks(order, peaks[inds])


    found_lines = []
    for i in range(matched.shape[0]):
        found_lines.append(Spectra.MatchedLine(matched[i,0], matched[i,1], matched[i,2]))
    
    header_dst = params_group['header']

    header = {}
    for key in header_dst.attrs.keys():
        header[key] = header_dst.attrs[key]

    if len(header) == 0:
        header = None
    
    #close file
    hdf5_file.close()
    
    Solution = Spectra.FinalWavelengthSolution(coeffs, m0, atlas, found_lines, (min_order, max_order),  \
                                               (min_useful_order, max_useful_order),(min_pix, max_pix), \
                                                ThAr_Peaks, rms, mjd)
    
    Solution.final_rms = final_rms
    Solution.ThAr_header = header

    return Solution
    
