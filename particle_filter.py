import numpy as np
import cv2


class ParticleFilter:
    def __init__(
        self,
        x_0,
        y_0,
        first_frame,
        n_particles=1000,
        dt=0.04,
        window_size=(480, 640),
        square_size=20,
    ):
        self.n_particles = n_particles

        self.state = np.array([x_0, y_0, square_size])
        # state =[
        #   X[t],Y[t],S[t],
        #   X[t-1],Y[t-1],S[t-1]
        # ]
        self.std_state = np.array([15, 15, 1])

        self.window_size = window_size

        self.max_square = window_size[0] * 0.5
        self.min_square = window_size[0] * 0.1

        self.A = np.array([[1 + dt, 0, 0], [0, 1 + dt, 0], [0, 0, 1 + dt / 4]])

        self.B = np.array([[-dt, 0, 0], [0, -dt, 0], [0, 0, -dt / 4]])

        self.particles = _init_particles(self.state, n_particles)
        self.last_particles = np.array(self.particles)

        self.hist = _calc_hist(_get_view(first_frame, x_0, y_0, square_size))

    def next_state(self, frame):

        # 1. Predict the next state of the particles
        control_prediction = self.transition()
        control_prediction = self.filter_borders(control_prediction)
        
        # 2. Compute the histograms around the particles
        hists = self.candidate_histograms(control_prediction, frame)

        # 3. Compute the weights of the particles
        #    based on the histogram comparison
        weights = self.compare_histograms(hists, self.hist)

        # 4. Resample the particles
        self.last_particles = np.array(self.particles)
        self.particles = self.resample(control_prediction, weights)
        
        # 5. Compute the new state
        self.state = np.mean(self.particles, axis=0)
        self.hist = _calc_hist(
            _get_view(frame, self.state[0], self.state[1], self.state[2])
        )

        return (
            int(self.state[0]),
            int(self.state[1]),
            int(self.state[2]),
            self.particles,
            control_prediction,
        )

    def transition(self):
        """Predict the next state of the particles
        using the transition model and some noise.

        the model is:
        X[t] = A*X[t-1] + B*X[t-2] + noise
        
        return: A numpy array of shape (n_particles,3)"""
        n_state = self.state.shape[0]
        n_particles = self.particles.shape[0]
        noises = self.std_state * np.random.randn(n_particles, n_state)
        particles = (
            np.dot(self.particles, self.A)
            + np.dot(self.last_particles, self.B)
            + noises
        )
        return particles

    def candidate_histograms(self, predictions, image):
        "Compute histograms for all candidates"
        hists = np.array([
            _calc_hist(
                _get_view(image, x[0], x[1], x[2]))
                for x in predictions
        ])

        return hists

    def compare_histograms(self, hists, reference_hist):
        "Compare the histogram of the current reference histogram with those of all candidate hists"

        weights = np.array([
            _comp_hist(x, reference_hist)
            for x in hists
        ])

        return weights / np.sum(weights)

    def resample(self, predictions, weights):
        "Scatter new particles according to the weights of the predictions"
        indexes = np.arange(weights.shape[0])
        inds = np.random.choice(indexes, self.n_particles, p=weights)
        return predictions[inds]

    def filter_borders(self, predictions):
        "Remove candidates that will not have the correct square size."
        np.clip(
            predictions[:, 0],
            self.state[2] + 1,
            self.window_size[0] - (1 + self.state[2]),
            predictions[:, 0],
        )
        np.clip(
            predictions[:, 1],
            self.state[2] + 1,
            self.window_size[1] - (1 + self.state[2]),
            predictions[:, 1],
        )
        np.clip(predictions[:, 2], self.min_square, self.max_square, predictions[:, 2])

        return predictions


def _init_particles(state, n):
    return np.array([state]* n)
    
def _get_view(image, x, y, sq_size):
    """
    Get a smaller image, centered at (x,y) with size (sq_size x sq_size)
    """

    # with numpy arrays this is an O(1) operation
    view = image[
        int(x - sq_size / 2) : int(x + sq_size / 2),
        int(y - sq_size / 2) : int(y + sq_size / 2),
        :,
    ]
    return view


def _calc_hist(image):
    """
    Computes the color histogram of an image (or from a region of an image).

    image: 3D Numpy array (X,Y,RGB)

    return: One dimensional Numpy array
    """

    mask = cv2.inRange(
        image, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0))
    )
    hist = cv2.calcHist([image], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, norm_type=cv2.NORM_MINMAX)
    return hist


def _comp_hist(hist1, hist2):
    """
    Compares two histograms together using the article's metric

    hist1,hist2: One dimensional numpy arrays
    return: A number
    """
    lbd = 20
    return np.exp(lbd * np.sum(hist1 * hist2))
