This is a test application for the GPU capon beamformer written in Python.

Prior to running this application you must build the COS-framework.
See INSTALL.txt.

Before the application can run you also have to run 
> python load_db.py

Load_db uses the Config.py script found in framework together with the fileLUT.py script in order to read a given dataset and convert it to a hdf database.

In load_db.py there is a tag ('motion phantom 16x 3.4 MHz 20dB30dB') which refers to a tag found in fileLUT.py.    This dataset can be downloaded from http://folk.uio.no/jpaasen/motion_phantom_16x_3-4MHz_20dB30dB/, and all files must be placed in the data-folder and within a folder matching the path found in fileLUT.py (motion_phantom_16x_3-4MHz_20dB30dB).

When the database is generated run the application with 
> python capon_on_speed_gui.py

