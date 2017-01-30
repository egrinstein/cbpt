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
        
    h= cv2.calcHist(image,[0,1],None,[10,10],[36,180,52,256])
    return h/np.sum(h)

def comp_hist(hist1,hist2):

    lbd = 20
    return np.exp(lbd*np.sum(hist1*hist2))


class ParticleFilter(object):
    def __init__(self,x,y,first_frame,n_particles=1000,dt=0.01,
    				window_size=(480,640),square_size=20):
        self.n_particles = n_particles
        self.n_iter = 0
        self.square_size = square_size
        vx = vy = vsquare_size = 0
        self.state = np.array([x,y,square_size]) 
        # state =[X[t],Y[t],S[t],X[t-1],Y[t-1],S[t-1]]
        self.std_state = np.array([20,20,.5])

        self.window_size = window_size
        
        self.max_square = window_size[0]/3
        self.min_square = window_size[0]/10

        self.A = np.array([[1+dt,0,0],
                           [0,1+dt,0],
                           [0,0,1+dt/4]])


        self.B = np.array([[-dt,0,0],
                           [0,-dt,0],
                           [0,0,-dt/4]])

        self.particles = init_particles(self.state,n_particles)
        self.last_particles = init_particles(self.state,n_particles)                                
                                        
        self.hist = calc_hist(get_view(first_frame,x,y,square_size))
        
     
    def next_state(self,frame):       
        """ AR Model for the "prediction" step AKA Control update 
         Predicts x(t) ~ p(x(t)| u(t),x(t-1))
         Where u(t) represents the control of the system/the dynamics
        
         This simplified model uses a recursion of order 1 and fixes the 
         window size. Its transition is expressed in the variable "transition_matrix"
        """
        
        
      
        control_prediction = self.transition()
        hists = self.candidate_histograms(control_prediction,frame)
        
        weights = self.compare_histograms(hists,self.hist)
        self.last_particles = np.array(self.particles)
        self.particles = self.resample(control_prediction,weights)
        self.state = np.mean(self.particles,axis=0)


        self.last_frame = frame
        self.n_iter += 1
        self.square_size = self.state[2]
        self.hist = calc_hist(get_view(frame,self.state[0],self.state[1],self.state[2]))
        
        
        return int(self.state[0]),int(self.state[1]),int(self.state[2])
        
        
    def transition(self):
        n_state = self.state.shape[0]
        n_particles = self.particles.shape[0]
        noises = np.zeros(self.particles.shape)
        for i in range(n_state):
            noises[:,i] = self.std_state[i]*np.random.randn(1,n_particles)
        particles = np.dot(self.A,self.particles.T).T
        particles = particles + np.dot(self.B,self.last_particles.T).T
        particles = particles + noises

        particles[:,0] = np.clip(particles[:,0],self.square_size,self.window_size[0]-1-self.square_size,particles[:,0])
        particles[:,1] = np.clip(particles[:,1],self.square_size,self.window_size[1]-1-self.square_size,particles[:,1])
        particles[:,2] = np.clip(particles[:,2],self.min_square,self.max_square)
        return particles

    def candidate_histograms(self,predictions,image):
        views = map(lambda x: get_view(image,x[0],x[1],self.square_size),predictions)
        return map(calc_hist,views)
    def compare_histograms(self,hists,last_hist):
        
        weights = list(map(lambda x: comp_hist(x,last_hist),hists))
        weights = weights/np.sum(weights)
        return weights
        
    def resample(self,predictions,weights):
        indexes = np.arange(weights.shape[0])
        inds = np.random.choice(indexes,self.n_particles,p=weights)
        return predictions[inds]
             
