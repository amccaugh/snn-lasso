#%%
import numpy as np
from matplotlib import pyplot as plt
from numba import njit
from scipy.optimize import minimize



# =============================================================================
# Target parameters and initialization
# =============================================================================
np.random.seed(1)
N = 400 # Number of neurons / number of dictionary items
M = 400 # Dimension of input target / output vector

# Create a randomized dictionary `Phi` and target state `s`
Phi = np.random.uniform(0,1,size = (M,N))
s = np.random.uniform(0,2, size = M)

# Normalize each dictionary atom (phi_i in Phi) to unit Euclidean norm
for i in range(N):
    Phi[:,i] /= np.linalg.norm(Phi[:,i])

# Set neuron membrane potential parameters
vf = 1 # Threshold membrane potential
vr = 0 # Reset membrane potential
lam = 0.1 # Lambda = static bias value for membrane current

# Create bias vector `b` and weight matrix `w` from dictionary `Phi`
b = Phi.T @ s # Bias vector for the neurons (size N)
w = Phi.T @ Phi # Weight matrix between the neurons (size NxN)
w_mask = 1 - np.identity(N)
w *= w_mask

# X=4 means the alpha term integrates the latest 4 spikes (and ignores all spikes before that)
X = 10


total_time = 150
dt = 1e-3

times = np.arange(0,total_time,dt)



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
    return v, mu, L, Lidx, is_spiking, alpha
    
@njit
def run(total_time, dt):
    times = np.arange(0,total_time,dt)
    
    v = np.zeros(N) # v is the membrane potential
    
    # L holds the last X times a spike arrived to neuron i
    L = np.ones((N,X))*(-np.inf)
    Lidx = np.zeros(N, dtype = np.int_)

    data_v = np.zeros((len(times), N))
    data_mu = np.zeros((len(times), N))
    data_spikes = np.zeros((len(times), N))
    data_alpha = np.zeros((len(times), N))
    
    for i,t in enumerate(times):
        v, mu, L, Lidx, is_spiking, alpha = timestep(v, L, Lidx, t, dt)
        
        # Save data
        data_v[i,:] = v
        data_mu[i,:] = mu
        data_spikes[i,:] = is_spiking
        data_alpha[i,:] = alpha
    
    return data_v, data_mu, data_spikes, data_alpha


time_start = time.time()
data_v, data_mu, data_spikes, data_alpha = run(total_time, dt)
# Compute spike rate
data_spike_rates = (np.cumsum(data_spikes,0).T/times).T
print(time.time() - time_start)

# fig, axs = plt.subplots(4,1, sharex = True, sharey = False)
# p = 0
# [axs[p].plot(times, data_v[:,n]) for n in range(N)]
# p += 1
# [axs[p].plot(times, data_mu[:,n]) for n in range(N)]
# p += 1
# [axs[p].plot(times, np.cumsum(data_spikes[:,n]),'.') for n in range(N)]
# p += 1
# [axs[p].plot(times, data_alpha[:,n],'.') for n in range(N)]
# p += 1
# plt.tight_layout()

solution_snn = data_spike_rates[-1,:]
print("SNN LASSO solution = %s" % np.round(np.sum(data_spikes,axis = 0)/total_time,2))

# =============================================================================
# Solve using LASSO objective
# =============================================================================
def classo(x,lam,Phi,s):
    # Evaluates Lasso objective
    f = 1/2*np.linalg.norm(s-Phi@x,2)**2 + lam*np.linalg.norm(x,1)
    return f

# Solve problem by direct optimization
f_opt = lambda x: classo(x,lam,Phi,s)
x0  = solution_snn
res = minimize(f_opt, x0, 
               bounds = [(0,None)]*N)
solution_direct =  res.x
print("Direct LASSO solution =", np.round(solution_direct,2))

#%%

# Calculate convergence
def convergence_vs_time(times, data_spike_rates, solution_direct):
    num_points = 1000
    downsample = len(times)//num_points
    E_star = classo(solution_direct, lam, Phi, s)
    convergence = []
    times_downsampled = []
    for n in range(downsample,len(times),downsample):
        E_t = classo(data_spike_rates[n,:],lam,Phi,s)
        convergence.append((E_t-E_star)/E_star)
        times_downsampled.append(times[n])
    return times_downsampled, convergence


t, c = convergence_vs_time(times, data_spike_rates, solution_direct)
fig, ax = plt.subplots()
ax.plot(t,c,'.')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('Time')
ax.set_ylabel('Convergence (E(t)-E*)/E*')
