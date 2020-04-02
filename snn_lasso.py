#%%
import numpy as np
#from numba import njit

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
L = np.ones([N,X])*(-100)
Lidx = np.zeros(X, dtype = int)


# v is the membrane potential
v = np.zeros(N)

dt = 1e-4
times = np.arange(0,10,dt)

data = {}
data['v'] = np.zeros([len(times), N])
data['mu'] = np.zeros([len(times), N])
data['alpha'] = np.zeros([len(times), N])
data['spikes'] = np.zeros([len(times), N])

for i,t in enumerate(times):
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
    
    # Save data
    data['v'][i,:] = v
    data['mu'][i,:] = mu
    data['alpha'][i,:] = alpha
    data['spikes'][i,is_spiking] = 1

figure()
plot(times, data['v'][:,0])
plot(times, data['v'][:,1])
plot(times, data['v'][:,2])


figure()
plot(times, data['mu'][:,0])
plot(times, data['mu'][:,1])
plot(times, data['mu'][:,2])

figure()
plot(times, data['spikes'][:,0]*1, '+')
plot(times, data['spikes'][:,1]*1.1, '+')
plot(times, data['spikes'][:,2]*1.2, '+')

figure()
plot(times, data['alpha'][:,0])
plot(times, data['alpha'][:,1])
plot(times, data['alpha'][:,2])