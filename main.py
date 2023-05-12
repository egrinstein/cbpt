import cv2
import numpy as np

from particle_filter import ParticleFilter


def _create_legend(img, pt1, pt2):
    text1 = "Before resampling"
    cv2.putText(img, text1, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    text2 = "After resampling"
    cv2.putText(img, text2, pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


def main():
    cap = cv2.VideoCapture(0)
    ret, first_frame = cap.read()

    print("Resolution:", first_frame.shape[:2])
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

    x_0, y_0 = first_frame.shape[0]//2, first_frame.shape[1]/2 # Start tracking in the middle of the frame
    pf = ParticleFilter(x_0, y_0, first_frame, n_particles=1000, square_size=50, dt=0.20,
                        window_size=first_frame.shape[:2])

    while True:
        ret, frame = cap.read()
        orig = np.array(frame)
        img = frame
        norm_factor = 255.0 / (np.sum(frame, axis=2)[:, :, np.newaxis] + 1e-6)

        frame = frame * norm_factor
        frame = cv2.convertScaleAbs(frame)
        frame = cv2.blur(frame, (5, 5))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get the next state
        x, y, sq_size, distrib, distrib_control = pf.next_state(frame)
        p1 = (int(y - sq_size), int(x - sq_size))
        p2 = (int(y + sq_size), int(x + sq_size))

        # before resampling
        for x2, y2, scale2 in distrib_control:
            x2 = int(x2)
            y2 = int(y2)
            cv2.circle(img, (y2, x2), 1, (255, 0, 0), thickness=10)
        # after resampling
        for x1, y1, scale1 in distrib:
            x1 = int(x1)
            y1 = int(y1)
            cv2.circle(img, (y1, x1), 1, (0, 0, 255), thickness=10)

        cv2.rectangle(img, p1, p2, (0, 0, 255), thickness=5)

        alpha = 0.5
        cv2.addWeighted(orig, alpha, img, 1 - alpha, 0, img)
        _create_legend(img, (40, 40), (40, 20))

        cv2.imshow("frame", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
