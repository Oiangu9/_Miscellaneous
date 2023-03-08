'''
Monte Carlo Markov Chain Phase Transition in a Lennard-Jones-like Potential with Gravity

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from time import time
from mpl_toolkits.axes_grid.inset_locator import InsetPosition
from time import sleep

# PARAMETERS ##############################################################
m=3e-26 #kg
epsLJ=1
sigmaLJ=1

epsLJ_SI = 23.3e3/(6.022e23) #J hydrogen bond between water molecules
sigmaLJ_SI = 0.197e-9+0.275e-9 #m

Ls=[ 45,100 ] # units of sigma
t_mfp = 1e-10 # seconds
v_max = min(Ls)/t_mfp
kB=1.38e-23 #J/K
N_particles=15**2
N_iterations=1000
N_neighbors=10
height_density_bins = 10

plot_every=2
N_iterations=1000
g=float(input("Please, choose a gravity value in m/s^2, ex. 9.8: "))
Temp_adim = float(input("Please choose a temperature value in eps/kB, ex. 0.3 : "))


height_bins = np.linspace(-Ls[1]/2, Ls[1]/2, height_density_bins+1)
height_bin_c=np.linspace(-Ls[1]/2+Ls[1]/height_density_bins/2, 
                         Ls[1]/2-Ls[1]/height_density_bins/2, height_density_bins)


# ROUTINES ###################################################################
def get_initialized_state( N_particles, L_x, L_y, v_max, 
                        mode_position='a_sixth_of_room', mode_speeds='slow'):
    v_magnitudes =  np.random.uniform(0, 1, size=N_particles)  # [N_particles]
    v_angles = 2*np.pi*np.random.uniform(0,1, size=N_particles) #[N_particles]
    if mode_speeds=='slow':
        v_magnitudes *=(v_max/40)
    elif mode_speeds=='still':
        v_magnitudes *= 0
    else: # randomly
        v_magnitudes *=(v_max)
    
    if mode_position=='a_sixth_of_room':    
        return np.vstack( (np.random.uniform(-Ls[0]/2,-Ls[0]/2+Ls[0]/6, N_particles),
                         np.random.uniform(Ls[1]/2-Ls[1]/6,Ls[1]/2, N_particles),
                         v_magnitudes*np.cos(v_angles),
                         v_magnitudes*np.sin(v_angles) ) ).T  #[N_particles, 4]
    elif mode_position=='lattice':
        n=int(np.sqrt(N_particles)) # aranged in nxn lattice
        if n**2!=N_particles:
            raise ValueError
        xy = np.meshgrid(np.arange(0,n), np.arange(0,n))
        return np.vstack( (2**(1/6)*xy[0].flatten(), 2**(1/6)*xy[1].flatten(),
                         v_magnitudes*np.cos(v_angles),
                         v_magnitudes*np.sin(v_angles) ) ).T  #[N_particles, 4]
    
    else: # mode_position=='randomly':    
        return np.vstack( (np.random.uniform(-Ls[0]/2,Ls[0]/2, N_particles),
                         np.random.uniform(-Ls[1]/2,Ls[1]/2, N_particles),
                         v_magnitudes*np.cos(v_angles),
                         v_magnitudes*np.sin(v_angles) ) ).T  #[N_particles, 4]
        

def E_intrinsic(state, m=m):
    return 0.5*m*np.sum(state[:,2]**2+state[:,3]**2)*(sigmaLJ_SI)**2 # units of joules

def pairwise_distances(xy_positions_listed): # expected to be [N_particles, 2] the input
    # first computed [N_particles, 1, 2]-[1, N_particles, 2]->[N_particles, N_particles, 2]
    # then apply norm along the last coordinate-> a matrix [N_particles, N_particles] of pairwise distances 
    return np.linalg.norm( xy_positions_listed[:, np.newaxis, :] - 
                              xy_positions_listed[np.newaxis, : , :], axis=-1 )
      
def E_pair_wise(state, epsLJ=epsLJ, sigmaLJ=sigmaLJ ): # Lennard-Jones
    distance_ij = pairwise_distances(state[:,:2]) # [N_particles, N_particles]
    r = distance_ij[ np.tril_indices(distance_ij.shape[0], k=-1) ] # only select the lower triangular part
    return np.sum(4*epsLJ*((sigmaLJ/r)**12-(sigmaLJ/r)**6))*epsLJ_SI # in J

def E_external(state, m=m, g=g): # gravity y the y direction
    return m*g*np.sum(state[:, 1])*sigmaLJ_SI # in J
    
def compute_E(state):
    return E_intrinsic(state)+E_pair_wise(state)+E_external(state) # in J

def get_average_density(xy_positions_listed, n_neighbors=N_neighbors):
    # average distance to the n_neighbors closest neighbors
    distance_ij = pairwise_distances(xy_positions_listed)
    return np.mean(np.sort(distance_ij, axis=1)[:,:n_neighbors])**-1
    
def get_average_and_height_density(xy_positions_listed, n_neighbors=N_neighbors, bins=height_density_bins):
    distance_ij = pairwise_distances(xy_positions_listed)
    closest_dists=np.mean(np.sort(distance_ij, axis=1)[:,:n_neighbors], axis=1) #[N_particles]
    height_density=[np.mean(closest_dists[
            np.where((xy_positions_listed[:,1]>height_bins[k])&(xy_positions_listed[:,1]<height_bins[k+1]))])
                   for k in range(height_density_bins)]
    return np.mean(closest_dists)**-1, np.array(height_density)**-1



# MAIN ###########################################################################
state = get_initialized_state( N_particles, *Ls, v_max, #)
                        mode_position='random', mode_speeds='random')

t0=time()
T=Temp_adim*epsLJ_SI/kB
print(f"Using T_adim={Temp_adim} -> T={T:.3}K")
state = get_initialized_state( N_particles, *Ls, v_max, #)
                        mode_position='random', mode_speeds='random')
state_new = state.copy() #[N_particles, 4]
E_state=compute_E(state)
average_Energies=[]
average_Densities=[]

for it in range(N_iterations):
    # in each time iteration all particles will have the chance to be displaced
    # sample random point non-uniformly in the circle of radious v_max, but uniformly in params
    new_v_magnitudes =  np.random.uniform(0, 1, size=state.shape[0])*v_max # [N_particles]
    new_v_angles = 2*np.pi*np.random.uniform(0,1, size=state.shape[0]) #[N_particles]
    new_vx = new_v_magnitudes*np.cos(new_v_angles)
    new_vy = new_v_magnitudes*np.sin(new_v_angles)

    new_x = state[:, 0]+t_mfp*new_vx
    new_y = state[:, 1]+t_mfp*new_vy
    # bouncing/reflecting boundaries for the particles so whenever a particle 
    # escapes one of the boundaries make it go back by the ammount it got out
    # dont need to iterate since v_max is such that tau*v_max=L but else do a while loop
    new_x[ np.where(new_x>Ls[0]/2) ] = Ls[0]-new_x[np.where(new_x>Ls[0]/2) ]
    new_y[ np.where(new_y>Ls[1]/2) ] = Ls[1]-new_y[np.where(new_y>Ls[1]/2) ]
    new_x[ np.where(new_x<-Ls[0]/2) ] = -Ls[0]-new_x[np.where(new_x<-Ls[0]/2) ]
    new_y[ np.where(new_y<-Ls[1]/2) ] = -Ls[1]-new_y[np.where(new_y<-Ls[1]/2) ]

    E_old=E_state
    old_state=state.copy()
    not_changed = np.ones(state.shape[0], dtype=bool)
    for k in range(state.shape[0]):
        state_new[k,:]=np.array([new_x[k], new_y[k], new_vx[k], new_vy[k]])
        E_state_new = compute_E(state_new) # J
        delta_E = E_state_new-E_state
        P=np.exp(-(delta_E)/Temp_adim/epsLJ_SI)
        if (delta_E < 0) or (np.random.rand() < P): 
            not_changed[k]=False
            state[k,:] = state_new[k,:]
            E_state = E_state_new
        else:
            state_new[k,:] = state[k,:]
    state[not_changed, 2:] = 0
    average_Energies.append(E_state/N_particles/epsLJ_SI)
    av, height_Density = get_average_and_height_density(state[:,:2])
    average_Densities.append(av)

    if it%plot_every==0:
        #clear_output(wait=True)
        sleep(0.2)
        plt.close()
        fig = plt.figure(figsize=(18,9))
        print(f"It={it} Taken computer time={time()-t0}s")
        ax = fig.add_subplot(121)
        ax.add_patch(Rectangle((-Ls[0]/2, -Ls[1]/2), Ls[0], Ls[1], fill=False))
        #ax.quiver(old_state[:,0],old_state[:,1],state[:,2]*t_mfp,state[:,3]*t_mfp,
        delta = state[:,:2]-old_state[:,:2]
        ax.quiver(old_state[:,0],old_state[:,1],delta[:,0],delta[:,1], 
                scale_units='xy', angles='xy', scale=1, alpha=0.7, color="#348ABD") # si multiplicas por la tau será en las unidades de distancia
        ax.scatter(state[:,0], state[:,1], s=5, c="r")
        ax.set_xlim(-Ls[0]/2-Ls[0]/10, Ls[0]/2+Ls[0]/10)
        ax.set_ylim(-Ls[1]/2-Ls[1]/10, Ls[1]/2+Ls[1]/10)

        ax.set_xlabel(f'sigma ({sigmaLJ_SI:.3} m)')
        ax.set_ylabel(f'sigma ({sigmaLJ_SI:.3}m)')
        ax.set_title(f"Time Iteration {it} at t_mfp={t_mfp}s\nAverage Energy {E_state/epsLJ_SI/state.shape[0]:.3} eps")
        ax = fig.add_subplot(233)
        ax.plot(average_Energies, color="#348ABD")
        ax.set_title("Average Energy vs Iterations")
        ax.set_ylabel(f"Averga Energy in eps ({epsLJ_SI:.3} J)")
        box = ax.get_position()
        ax.set_position([box.x0 - 0.03,box.y0,box.width*1.15, box.height])
        #ax.set_xlabel("Iteration")
        ax2 = plt.axes([0,0,1,1])
        # Manually set the position and relative size of the inset axes within ax1
        ip = InsetPosition(ax, [0.4,0.4,0.6,0.6])
        ax2.set_axes_locator(ip)
        ax2.plot(average_Energies[-200:])
        ax2.set_xlabel("Last 200 iterations")

        ax = fig.add_subplot(2,3,6)
        ax.plot(average_Densities, color="#A60628")
        ax.set_title("Average Density vs Iterations")
        ax.set_ylabel(f"Mean dist. to {N_neighbors} closest neighbs.^-1 (sigma)")
        ax.set_xlabel("Iteration")
        box = ax.get_position()
        ax.set_position([box.x0 - 0.03,box.y0,box.width*1.15, box.height])
        ax2 = plt.axes([0,0,1,1])
        ip = InsetPosition(ax, [0.4,0.0,0.6,0.6])
        ax2.set_axes_locator(ip)
        ax2.plot(average_Densities[-200:], color="#A60628")
        ax2.xaxis.tick_top()
        ax2.set_xlabel("Last 200 its")
        ax2.xaxis.set_label_position('top') 
        ax = fig.add_subplot(1,6,4)
        ax.plot(height_Density, height_bin_c, 'o-', color="#A60628")
        ax.set_title("Height-Density")
        #ax.set_ylabel(f"y")
        ax.set_xlabel("Density")
        ax.set_ylim(-Ls[1]/2-Ls[1]/10, Ls[1]/2+Ls[1]/10)
        ax.set_yticks(height_bins)
        ax.grid(axis='y')
        box = ax.get_position()
        box.x0 = box.x0 - 0.05
        box.x1 = box.x1 - 0.05
        ax.set_position(box)
        ax.yaxis.set_ticklabels([])
        fig.suptitle(f"Adimensionalized Temperature = {Temp_adim}; g={g:.2}m/s^2; N_particles={N_particles}")
        plt.show(block=False)

fig = plt.figure(figsize=(18,9))
ax = fig.add_subplot(121)
ax.add_patch(Rectangle((-Ls[0]/2, -Ls[1]/2), Ls[0], Ls[1], fill=False))
#ax.quiver(old_state[:,0],old_state[:,1],state[:,2]*t_mfp,state[:,3]*t_mfp,
delta = state[:,:2]-old_state[:,:2]
ax.quiver(old_state[:,0],old_state[:,1],delta[:,0],delta[:,1], 
        scale_units='xy', angles='xy', scale=1, alpha=0.7, color="#348ABD") # si multiplicas por la tau será en las unidades de distancia
ax.scatter(state[:,0], state[:,1], s=5, c="r")
ax.set_xlim(-Ls[0]/2-Ls[0]/10, Ls[0]/2+Ls[0]/10)
ax.set_ylim(-Ls[1]/2-Ls[1]/10, Ls[1]/2+Ls[1]/10)

ax.set_xlabel(f'sigma ({sigmaLJ_SI:.3} m)')
ax.set_ylabel(f'sigma ({sigmaLJ_SI:.3}m)')
ax.set_title(f"Time Iteration {it} at t_mfp={t_mfp}s\nAverage Energy {E_state/epsLJ_SI/state.shape[0]:.3} eps")
ax = fig.add_subplot(233)
ax.plot(average_Energies, color="#348ABD")
ax.set_title("Average Energy vs Iterations")
ax.set_ylabel(f"Averga Energy in eps ({epsLJ_SI:.3} J)")
box = ax.get_position()
ax.set_position([box.x0 - 0.03,box.y0,box.width*1.15, box.height])
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip = InsetPosition(ax, [0.4,0.4,0.6,0.6])
ax2.set_axes_locator(ip)
ax2.plot(average_Energies[-200:])
ax2.set_xlabel("Last 200 its")
#ax.set_xlabel("Iteration")
ax = fig.add_subplot(2,3,6)
ax.plot(average_Densities, color="#A60628")
ax.set_title("Average Density vs Iterations")
ax.set_ylabel(f"Mean dist. to {N_neighbors} closest neighbs.^-1 (sigma^-1)")
ax.set_xlabel("Iteration")
box = ax.get_position()
ax.set_position([box.x0 - 0.03,box.y0,box.width*1.15, box.height])
ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax, [0.4,0.0,0.6,0.6])
ax2.set_axes_locator(ip)
ax2.plot(average_Densities[-200:], color="#A60628")
ax2.xaxis.tick_top()
ax2.set_xlabel("Last 200 its")
ax2.xaxis.set_label_position('top') 
ax = fig.add_subplot(1,6,4)
ax.plot(height_Density, height_bin_c, 'o-', color="#A60628")
ax.set_title("Height-Density")
#ax.set_ylabel(f"y")
ax.set_xlabel("Density")
ax.set_ylim(-Ls[1]/2-Ls[1]/10, Ls[1]/2+Ls[1]/10)
ax.set_yticks(height_bins)
ax.grid(axis='y')
box = ax.get_position()
box.x0 = box.x0 - 0.05
box.x1 = box.x1 - 0.05
ax.set_position(box)
ax.yaxis.set_ticklabels([])
fig.suptitle(f"Adimensionalized Temperature = {Temp_adim}; g={g:.2}m/s^2; N_particles={N_particles}")
np.save(f"./T_{Temp_adim}_g_{g:.2}_N_its_{N_iterations}_N_parts_{N_particles}.npy", state)
plt.savefig(f"./T_{Temp_adim}_g_{g:.2}_N_its_{N_iterations}_N_parts_{N_particles}_ct_{time()-t0:.2}s.png", dpi=120)

