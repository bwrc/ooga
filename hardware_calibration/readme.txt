
*** Hardware calibration ***

You can use the Qt version to calibrate the cameras. It also computes the transformation matrix between eye and scene cameras.
Open with ./cali

There is the option to flip the streamed scene camera images (upside down).
However, don't flip the scene camera images because the code currently assumes it to be upside down. 

The code of LedCalibration compiles at least with Opencv 2.4.* and Qt 4.

Note: the translation vector of the resulting transformation matrices are in units of 'cm'




*** Forming the corresponding glint grid from example images ***

Only Matlab code currently.
Annotate manually with annotate_glints.
From the grid models with form_glint_grid_model.
