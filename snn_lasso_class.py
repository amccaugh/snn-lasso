#%%
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.optimize import minimize
from scipy.optimize import brentq

def classo(x,lam,Phi,s):
    # Evaluates (C)LASSO objective
    f = 1/2*np.linalg.norm(s-Phi@x,2)**2 + lam*np.linalg.norm(x,1)
    return f

def classo_solve(x0,lam,Phi,s):
    # Solve problem by direct optimization
    f_opt = lambda x: classo(x,lam,Phi,s)
    res = minimize(f_opt, x0, 
                   bounds = [(0,None)]*len(x0))
    solution =  res.x
    return solution
# print("Direct LASSO solution =", np.round(solution_direct,2))

def convergence(solution_snn, solution_direct, lam, Phi, s):
    E_star = classo(solution_direct, lam, Phi, s)
    E = classo(solution_snn,lam,Phi,s)
    return (E-E_star)/E_star


@njit
def timestep(w,b,lam,vf,vr,N,X,v,L,Lidx,t,dt):
    # Compute the per-neuron membrane current
    alpha = np.sum(np.exp(-(t-L)), axis = 1)
    mu = b - alpha @ w
    # Increase the membrane potential by the membrane current
    v = v + dt*(mu - lam)
    # Check which neurons have exceeded the membrane potential
    # is_spiking = v > vf
    # if sum(is_spiking > 0):
    #     # Reset those neurons
    #     v[is_spiking] = vr
    #     # Update the last-spike-time for each synapse in L
    #     for n in range(N):
    #         if is_spiking[n] == True:
    #             L[n,Lidx[n]] = t
    #             Lidx[n] = (Lidx[n] + 1) % X
    return v, mu, L, Lidx, is_spiking
    
@njit
def run(w,b,lam,vf,vr,N,X,v,L,Lidx,total_time, dt):
    times = np.arange(0,total_time,dt)

    data_v = np.zeros((len(times), N))
    data_mu = np.zeros((len(times), N))
    data_spikes = np.zeros((len(times), N))
    
    for i,t in enumerate(times):
        v, mu, L, Lidx, is_spiking = timestep(w,b,lam,vf,vr,N,X,v, L, Lidx, t, dt)
        
        # Save data
        data_v[i,:] = v
        data_mu[i,:] = mu
        data_spikes[i,:] = is_spiking
    
    return data_v, data_mu, data_spikes


class SNN(object):
    
    def __init__(self,*args,**kwargs):
        self.initialize(*args, **kwargs)
    
    def initialize(self, N, M, seed = None):
        # Set neuron membrane potential parameters
        self.v = np.zeros(N) # v is the membrane potential
        self.vf = 1 # Threshold membrane potential
        self.vr = 0 # Reset membrane potential
        self.lam = 0.1 # Lambda = static bias value for membrane current

        self.N = N # Number of neurons / number of dictionary items
        self.M = M # Dimension of input target / output vector

        # Create a randomized dictionary `Phi` and target state `s`
        if seed is not None:
            np.random.seed(seed)
        self.Phi = np.random.uniform(0,1,size = (M,N))
        self.s = np.random.uniform(0,2, size = M)
        # Normalize each dictionary atom (phi_i in Phi) to unit Euclidean norm
        for i in range(N):
            self.Phi[:,i] /= np.linalg.norm(self.Phi[:,i])

        # Create bias vector `b` and weight matrix `w` from dictionary `Phi`
        self.b = self.Phi.T @ self.s # Bias vector for the neurons (size N)
        self.w = self.Phi.T @ self.Phi # Weight matrix between the neurons (size NxN)
        self.w_mask = 1 - np.identity(N)
        self.w *= self.w_mask

        # L holds the last X times a spike arrived to neuron i
        self.X = 10
        self.L = np.ones((N,self.X))*(-np.inf)
        self.Lidx = np.zeros(N, dtype = np.int_)

        self.t = 0
    
    
    def vt(self, dt):
        """ Computes the membrane potential `dt` time away from now """
        # Compute the per-neuron membrane current
        # In Euler time-stepping the equation is this:
            # integral of d(mu)/dt = dt*(b + alpha*w)
        # But now that we're doing arbitrary time-stepping, we have to use the
        # full integral:
            # integral of d(mu)/dt = b*dt + alpha*w*(1-np.exp(-dt)
        alpha = np.sum(np.exp(-(self.t-self.L)), axis = 1)
        delta_mu = self.b*dt - (alpha @ self.w) * (1-np.exp(-dt))
        # Increase the membrane potential by the membrane current
        v = self.v + delta_mu - self.lam*dt
        return v
        
    
    def _next_spike_objective(self, dt):
        v = self.vt(dt)
        return np.sum(v > self.vf) - 0.5
        


    def timestep_to_next_spike(self, dt_tol, time_to_scan = 1):
        # Use "brentq" zero-crossing algorithm to find where first spike happens
        dt = brentq(f = self._next_spike_objective, a = 0, b = time_to_scan, xtol = dt_tol)


        # Move forward in time `dt + dt_tol` because just moving `dt` may
        # not put `v` quite above threshold
        self.v = self.vt(dt + dt_tol)
        self.t += dt + dt_tol
        
        # Check which neuron `i` has reached threshold and reset it
        is_spiking = self.v > self.vf
        assert sum(is_spiking) > 0
        # Reset those neurons
        self.v[is_spiking] = self.vr
        # Update the last-spike-time for each synapse in L
        for i in range(self.N):
            if is_spiking[i] == True:
                self.L[i, self.Lidx[i]] = self.t
                self.Lidx[i] = (self.Lidx[i] + 1) % self.X
        
        return self.t, np.where(is_spiking)[0].tolist()
        
time_start = time.time()
snn = SNN(N = 500, M = 500, seed = 3)

spikes = []
for n in range(4000):
    # print(snn.v)
    t, s = snn.timestep_to_next_spike(1e-6)
    spikes.extend(s)
solution_snn = np.array([spikes.count(n) for n in range(snn.N)])/t
solution_direct =classo_solve(solution_snn, snn.lam, snn.Phi, snn.s)
print(convergence(solution_snn, solution_direct, snn.lam, snn.Phi, snn.s))
print(time.time() - time_start)


# %timeit -r10 -n10 find_next_spike(snn, dt_tol = 1e-6, time_to_scan = 1)
#%%

