## Centerstage
Python package to crop in and center a subject's face from an input video

### Setup and Use
Make sure Anaconda is installed and run `conda env create -f environment.yaml`

[Download the shape predictor file](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) from the dlib website, unpack and place in the `models` folder.

Then run `python centerstage.py -f path/to/video.mp4 -o optional/output/folder`