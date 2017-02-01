# cbpt
Color Based Probabilistic Tracking using Python + OpenCV

Functional object tracking implementation of Perez et al.'s article entitled "Color Based Probabilistic Tracking",
which uses a particle filter and histogram comparison for a robust object tracking.

This program tries to mimic the exact algorithm descripted in the aforementioned article. Some features were however 
approximated. Some considerations:
* Using the exact measure of similarity between the current and candidate histograms
* For the control update: State is represented by the vector (x,y,square_size). The transition goes as follows: X[t+1] = X[t] + V[t]dt + N[t], where V[t] represents the current velocity of the state and N[t] is a gaussian vector.
* For the histogram's computation, only considerable values of hue/saturation are taken into account (>20%). The histogram is normalized.
* The ROI is computed by averaging the current distribution.
* The first distribution is considered to be a distribution with all particles in the location of the first ROI's central points, to be given as input of the program.

![alt text](http://i.imgur.com/EVnTXz3.gif "Visual Tracking")
