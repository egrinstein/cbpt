import numpy as np
import cv2


def init_particles(state,n):
    particles = np.array([state,]*n)
    return particles
    

def get_view(image,x,y,sq_size):
    
    # with numpy arrays this is an O(1) operation

    view = image[int(x-sq_size/2):int(x+sq_size/2),
                 int(y-sq_size/2):int(y+sq_size/2),:]
    return view
    
def calc_hist(image):
    

    mask = cv2.inRange(image, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    hist = cv2.calcHist([image],[0],mask,[180],[0,180])
    #hist = cv2.calcHist(image,[0,1],None,[10,10],[0,180,0,255])
    cv2.normalize(hist,hist,0,1,norm_type=cv2.NORM_MINMAX)
    return hist

def comp_hist(hist1,hist2):

    lbd = 20
    return np.exp(lbd*np.sum(hist1*hist2))
    

class ParticleFilter(object):
    def __init__(self,x,y,first_frame,n_particles=1000,dt=0.04,
                    window_size=(480,640),square_size=20):
        self.n_particles = n_particles
        self.n_iter = 0
        self.state = np.array([x,y,square_size]) 
        # state =[X[t],Y[t],S[t],X[t-1],Y[t-1],S[t-1]]
        self.std_state = np.array([15,15,1])

        self.window_size = window_size
        
        self.max_square = window_size[0]*0.5
        self.min_square = window_size[0]*0.1

        self.A = np.array([[1+dt,0,0],
                           [0,1+dt,0],
                           [0,0,1+dt/4]])


        self.B = np.array([[-dt,0,0],
                           [0,-dt,0],
                           [0,0,-dt/4]])


        self.particles = init_particles(self.state,n_particles)
        self.last_particles = np.array(self.particles)                                
                                        
        self.hist = calc_hist(get_view(first_frame,x,y,square_size))
        
     
    def next_state(self,frame):       
      
        control_prediction = self.transition()
        control_prediction = self.filter_borders(control_prediction)
       
        hists = self.candidate_histograms(control_prediction,frame)

        weights = self.compare_histograms(hists,self.hist)
        self.last_particles = np.array(self.particles)
        self.particles = self.resample(control_prediction,weights)
        self.state = np.mean(self.particles,axis=0)


        self.last_frame = np.array(frame)
        self.n_iter += 1
        self.hist = calc_hist(get_view(frame,self.state[0],self.state[1],self.state[2]))
        

        
        return int(self.state[0]),int(self.state[1]),int(self.state[2]),self.particles,control_prediction
        
        
    def transition(self):

        n_state = self.state.shape[0]
        n_particles = self.particles.shape[0]   
        noises = self.std_state*np.random.randn(n_particles,n_state)
        particles = np.dot(self.particles,self.A) + np.dot(self.last_particles,self.B) + noises
        return particles

    def candidate_histograms(self,predictions,image):
        
        hists = [] 

        for x in predictions:
            v = get_view(image,x[0],x[1],x[2])
            hists.append(calc_hist(v))
        return hists
        
    def compare_histograms(self,hists,last_hist):
        
        weights = np.array(list(map(lambda x: comp_hist(x,last_hist),hists)))
        return weights/np.sum(weights)

    def resample(self,predictions,weights):
        indexes = np.arange(weights.shape[0])
        inds = np.random.choice(indexes,self.n_particles,p=weights)
        return predictions[inds]
    def filter_borders(self,predictions):  
        ""
        np.clip(predictions[:,0],self.state[2]+1,self.window_size[0]-(1+self.state[2]),predictions[:,0])        
        np.clip(predictions[:,1],self.state[2]+1,self.window_size[1]-(1+self.state[2]),predictions[:,1])
        np.clip(predictions[:,2],self.min_square,self.max_square,predictions[:,2])
        
        return predictions