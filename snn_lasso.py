#%%
import numpy as np
# Define the dictionary and starting state
# Phi = dictionary
Phi = np.array([[0.3313,0.8148,0.4364],
                [.8835,0.3621,0.2182],
                [0.3313,0.4527,0.8729]]).T
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


# L_ij holds the last time a spike arrived to neuron i from neuron j
L = np.ones(N)*(-100)

# v is the membrane potential
v = np.zeros(N)

dt = 1e-4
times = np.arange(0,10,dt)

data = {}
data['v'] = np.zeros([len(times),len(v)])

for n,t in enumerate(times):
    # Compute the per-neuron membrane current
    alpha = np.exp(-(t-L))
    mu = b - alpha @ w
    # Increase the membrane potential by the membrane current
    v += dt*mu
    # Check which neurons have exceeded the membrane potential
    is_spiking = v > vf
    # Reset those neurons
    v[is_spiking] = vr
    # Update the last-spike-time for each synapse in L
    L[is_spiking] = t
    
    # Save data
    data['v'][n,:] = v

figure()
plot(times, data['v'][:,0])
plot(times, data['v'][:,1])
plot(times, data['v'][:,2])