# EKF-MonoSLAM-ESE-650
The instruction to run this project are given below:

## Camera Calibration

Run the following to capture 10 images of a checkerboard for calibration. Use a checkerboard of size (7, 9) or change this parameter later in calibration.py. Hit '0' key until 10 captures are finished.

```bash
cd calibration
python capture.py
```

Now run the calibration sequence to generate intrinsics.yaml containing calibration parameters of the camera.

```bash
python calibration.py
```

## Running MonoSLAM
The MonoSLAM uses an initialization sequence to generate initial features for the map from a fixed checkerboard pattern. Fix a checkerboard in a space which has other corner features and turn the camera towards it before running the following commands.

Additionally, in main.py, change the VideoCapture() argument to your respective camera. By default the argument is 0.

```bash
cd ..
python main.py
```
After the initialization sequence runs, the SLAM will begin. At this point, you should see features being plotted in an open window. To close, press ESC key.

Move the camera slowly around to see the new features being added to the map.

