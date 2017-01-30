
import cv2
from particle_filter import ParticleFilter

def main():

    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    print("MAIN:",frame.shape)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    x,y = 240,420
    pf = ParticleFilter(x,y,frame,n_particles=800,square_size=80)
    while(True):        
        ret, frame = cap.read()
        img = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x,y,sq_size = pf.next_state(frame)
        p1 = (int(x-sq_size/2),int(y-sq_size/2))
        p2 = (int(x+sq_size/2),int(y+sq_size/2))
        
        cv2.circle(img, (y,x), sq_size, (255,255,0), thickness=5)        
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break          
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
