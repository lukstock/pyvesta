# pyvesta
PyVesta - Versatile Echelle Spectra exTraction and Analysis pipeline

# PyVesta

## About PyVesta

A versatile extraction routine that can extract raw data from various fiber fed echelle spectrographs such as HARPS and FEROS. Special effort was made to improve the accuracy of small instruments such as the Shelyak eShel II.  
If you use this software in your reseach please cite **TODO**


PyVesta can extract data from almost all fiber fed echelle spectrographs, as long as the dispersion (wavelength) direction is approximately perpendicular to the cross-dispersion (order) direction. So far, PyVesta has been tested with the ESO instruments HARPS and FEROS as well as our own Shelyak eShel II. PyVesta can easily be adapted to other instruments. For reference see chapter **Adjust PyVesta to your instrument**

## Installing PyVesta

For now, the easiest way is to download PyVesta directly as source code from this website. In the future, PyVesta will also be available via pip.  
    
It is highly recommended to install a new conda environment for PyVesta. You can find an instruction how to install conda [here](https://www.anaconda.com/docs/getting-started/installation "Anaconda installation guide").  

To create a new environment, use the command
    
    conda create --name <myenv>
    
To switch to the new environment, type
    
    conda activate <myenv>
    
You can install all dependencies via the command
    
    conda install astropy h5py numpy matplotlib scipy conda-forge::astroscrappy conda-forge::astroquery


## Using PyVesta
Only the basic routines are located in the pyvesta directory. The complete pipeline is provided as a .py scripy file in this repository. The files are named `PyVesta_INSTRUMENT.py`.  

Before using PyVesta, make sure you have switched to the correct conda environment.

To start PyVesta, navigate to the folder containing the `pyvesta` source folder and the `PyVesta_INSTRUMENT.py` file. You can specify which raw data folder to process and where to save the results in the `config.py` file. Note that you should always process data from only one night of observations at a time and should not mix data from different nights.

PyVesta, shown here as an example for our Shelyak eShel II, can be easily started by 

    python3 PyVesta_eShel.py

In addition to specifying them in the config.py file, the reduction parameters can also be passed via a command-line option. In this case, the values in config.py are overridden. In this example, the number of usable processor cores is set to 16:
    
    python3 PyVesta_eShel.py --npools 16


## Output

PyVesta creates a folder structure in the output directory. Each type of calibration frame or science image is placed in its own subfolder. It is organized as follows:

- Extracted_dir
    - Extracted_dir/Bias
    - Extracted_dir/Dark
    - Extracted_dir/Flat
    - Extracted_dir/Light
    - Extracted_dir/Orderdef
    - Extracted_dir/Plots
    - Extracted_dir/ThAr
    - Extracted_dir/wavelength_solutions
        

The combined master files, the reference wavelength solution, the order traces, and the extraction weights are saved directly to the "Extracted_dir" directory.  

We believe that each intermediate step of the extraction process should be saved separately. For this reason, each subfolder (except for wavelength_solutions) is itself divided into several subfolders. However, all extraction steps are performed only on the light files, so many subfolders are empty for the calibration frames.

- background     : This is where the backgrounds for the individual frames are stored.
- cal            : This is where the wavelength-calibrated spectra are stored.
- CCDcorr        : This is where the images are saved after the CCD corrections.
- continuum      : The calculated continuum is stored here as a spectrum.
- contnorm       : This is where the continuum normalized and merged spectrum is stored.
- contnorm_orders: This is where the continuum normalized but not merged spectrum is stored.
- ext            : The extracted spectrum is saved here, still in pixel scale.
- filtered       : This is where the spectrum filtered from cosmics is stored.
- flat           : The blaze corrected spectrum is stored here.
- merged         : The merged spectrum is stored here.

PyVesta can save the result spectra in two different ways. The default option is a custom HDF5 format. This has the advantage of being more flexible and adaptable to various characteristics of the spectrographs, such as multiple fibers. The spectra saved in this way can be viewed using the `Plot_HDF5.py` file located in the "Additional_scripts" folder.
Alternatively, it is also possible to save the files in FITS format.

## Get test data
A test dataset from our Shelyak eShel II can be downloaded here **TODO**. Test data for HARPS and FEROS can be downloaded from the [ESO Raw Data Archive](https://archive.eso.org/eso/eso_archive_main.html "ESO Raw Data Archive"). Please make sure that, in addition to the science frames, all calibration frames are downloaded as well.


## Adjust PyVesta to your instrument

All instrument specific parameters are stored in the `instruments.py` file. All parameters regarding extraction, e.g. number of processes and what figures to plot, are stored in the `config.py` file.

If you need any help, don't hesitate to contact me. Also, if you successfully adapted PyVesta to your instrument, please reach out to me, so that I can add your instrument and other people may also profit from that.


