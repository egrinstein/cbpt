# cbpt
Color Based Probabilistic Tracking using Python + OpenCV

Functional object tracking implementation of Perez et al.'s article entitled "Color Based Probabilistic Tracking",
which uses a particle filter and histogram comparison for a robust object tracking.

This program tries to mimic the exact algorithm descripted in the aforementioned article. Some features were however 
approximated. Some considerations:
* Using the exact measure of similarity between the current and candidate histograms
* For the control update, the idea of transition used is x[t+1] = x[t]*I + x[t-1]*I + v, 
where I represents the identity matrix and v is a decorrelated gaussian vector.
* For the histogram's computation, only considerable values of hue/saturation are taken into account (>20%). The histogram is normalized.
* The ROI is computed by averaging the current distribution.
* The first distribution is considered to be a distribution with all particles in the location of the first ROI's central points, to be given as input of the program.
