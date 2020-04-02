#%%
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.optimize import minimize





# Define the dictionary and starting state
# Phi = dictionary
Phi = np.array([[0.3313,0.8148,0.4364],
                [.8835,0.3621,0.2182],
                [0.3313,0.4527,0.8729]])
N = 3 # Number of neurons / number of dictionary items
M = 3 # Dimension of input target / output vector
vf = 1 # Threshold membrane potential
vr = 0 # Reset membrane potential

# s = target (desired output)
s = np.array([0.5,1,1.5])
# Lambda = static bias value
lam = 0.1

# Bias vector for the neurons
b = Phi.T @ s
# Weight matrix between the neurons
w = Phi.T @ Phi
w_mask = 1 - np.identity(N)
w *= w_mask



# L holds the last X times a spike arrived to neuron i
X = 4


dt = 1e-3
times = np.arange(0,10,dt)

data = {}
data_v = np.zeros([len(times), N])
data_mu = np.zeros([len(times), N])
data_spikes = np.zeros([len(times), N])




@njit
def timestep(v, L, Lidx, t, dt):
    # Compute the per-neuron membrane current
    alpha = np.sum(np.exp(-(t-L)), axis = 1)
    mu = b - alpha @ w
    # Increase the membrane potential by the membrane current
    v += dt*(mu - lam)
    # Check which neurons have exceeded the membrane potential
    is_spiking = v > vf
    if sum(is_spiking > 0):
        # Reset those neurons
        v[is_spiking] = vr
        # Update the last-spike-time for each synapse in L
        for n in range(N):
            if is_spiking[n] == True:
                L[n,Lidx[n]] = t
                Lidx[n] = (Lidx[n] + 1) % X
    return v, mu, L, Lidx, is_spiking
    
@njit
def run(total_time, dt):
    times = np.arange(0,total_time,dt)
    
    v = np.zeros(N) # v is the membrane potential
    
    # L holds the last X times a spike arrived to neuron i
    L = np.ones((N,X))*(-100)
    Lidx = np.zeros(X, dtype = np.int_)

    data_v = np.zeros((len(times), N))
    data_mu = np.zeros((len(times), N))
    data_spikes = np.zeros((len(times), N))
    
    for i,t in enumerate(times):
        v, mu, L, Lidx, is_spiking = timestep(v, L, Lidx, t, dt)
        
        # Save data
        data_v[i,:] = v
        data_mu[i,:] = mu
        data_spikes[i,:] = is_spiking
    
    return data_v, data_mu, data_spikes

total_time = 10

data_v, data_mu, data_spikes = run(10, dt)
#
#plt.figure()
#plt.plot(times, data_v[:,0])
#plt.plot(times, data_v[:,1])
#plt.plot(times, data_v[:,2])
#
#
#plt.figure()
#plt.plot(times, data_mu[:,0])
#plt.plot(times, data_mu[:,1])
#plt.plot(times, data_mu[:,2])
#
#plt.figure()
#plt.plot(times, data_spikes[:,0]*1, '+')
#plt.plot(times, data_spikes[:,1]*1.1, '+')
#plt.plot(times, data_spikes[:,2]*1.2, '+')

print("SNN LASSO solution =", np.sum(data_spikes,axis = 0)/total_time)

# =============================================================================
# Solve using LASSO objective
# =============================================================================
def lasso(x,lam,Phi,b):
    # Evaluates Lasso objective
    f = 1/2*np.linalg.norm(b-Phi@x,2)**2 + lam*np.linalg.norm(x,1)
    return f

# Solve problem by direct optimization
f_opt = lambda x: lasso(x,lam,Phi,s)
x0  = np.ones(N)
res = minimize(f_opt, x0, method='nelder-mead', options={'xtol':1e-6})
print("Direct LASSO solution =", np.round(res.x,2))
