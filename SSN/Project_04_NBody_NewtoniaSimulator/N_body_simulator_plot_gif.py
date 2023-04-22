'''
N BODY IN 3D SPACE NEWTONIAN PARTICLE SIMULATOR

This version of the code shows the animation and 
then generates a .gif animation

''' 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
import os

def gravity_force_on_particle_1_by_2(x1, x2, m1, m2, q1, q2, G=1.0):
    '''
    x1 and x2 are expected to be arrays of 3 elements (1 dimension) or 3x1 or 1x3 
    It returns a 3 element array back (the force vector)
    '''
    unit_vector = (x2-x1)/np.linalg.norm(x2-x1) # pointing from particle 1 to 2
    return (G*m1*m2/( np.linalg.norm(x2-x1)**2))*unit_vector

def coulomb_force_on_particle_1_by_2(x1, x2, m1, m2, q1, q2, k=1.0):
    '''
    x1 and x2 are expected to be arrays of 3 elements (1 dimension) or 3x1 or 1x3 
    It returns a 3 element array back (the force vector)
    '''
    unit_vector = -(x2-x1)/np.linalg.norm(x2-x1) # pointing from particle 2 to 1 (here if charges equal force must be repulsive thus the minus sign)
    return (k*q1*q2/( np.linalg.norm(x2-x1)**2))*unit_vector


def run_N_Body_simulator(N, positions_now, velocities_now, masses, additional_parameters,
                    force_list, t0, tf, timeIts, plotEvery, limits, J_trace, exp_name="", dpi=90, fps=10, show_frames=False):
    assert J_trace>=3, "For the Verlet algorithm, saving at least 3 time iterations is necessary, so choose J_trace>=3"
    os.makedirs("./temp", exist_ok=True)
    image_paths= []
    # initialize the figure
    fig = plt.figure(figsize=(10,10))
    
    # Compute the times in which the simulator will compute a step:
    times = np.linspace(start=t0, stop=tf, num=timeIts)
    # Get time increment delta t
    dt = times[1]-times[0]
    
    positions = np.zeros((J_trace, positions_now.shape[0], positions_now.shape[1]))
    # for the first time iteration we will estimate the previous position of the
    # particle using a simple Euler rule given the initial velocities
    positions[:] = positions_now-dt*velocities_now
    positions[0,:,:] = positions_now # copy the same position in all J

    dt2_masses = dt**2/(np.array(masses)[:,np.newaxis])
    
    for it, t in enumerate(times):
        # a Nx3 array (matrix) where we will save the forces in each time
        forces = np.zeros(positions_now.shape)
    
        # Step 1, compute the total force on each particle
        for k, xk in enumerate(positions[0]):
            for j, xj in enumerate(positions[0]): # each of the other particles
                if j!=k: # does not self-interact!
                    for force in force_list:
                        forces[k,:] += force(xk, xj, masses[k], masses[j], additional_parameters[k], additional_parameters[j])
        
        # Step 2, compute the position of the particles in the next time
        # first move all the positions one step onward
        positions[1:, :, :] = positions[:-1,:,:] # copy in the slots from 1 to J the ones that were in 0 to J-1
        
        positions[0,:,:] = 2*positions[1,:,:] - positions[2,:,:] + forces*dt2_masses
        
        # Step 3, compute the velcity in the next time for plotting purposes
        velocities_next = (positions[0,:,:]-positions[1,:,:])/dt
        
        # Plot the particles
        if it%plotEvery==0:
            print(f"{100*(it/timeIts):.3}%")
            output_image_path = f"./temp/NBody_it_{it}.png"
            plot_N_particles_and_save_it(fig, output_image_path, positions, velocities_next, t,
                             limits['xmin'], limits['xmax'], limits['ymin'],
                             limits['ymax'], limits['zmin'], limits['zmax'], dpi=dpi, fps=fps, show_frames=show_frames)
            image_paths.append(output_image_path)
        
        # prepare for the next time iteration
        velocities_now = velocities_next
        # for the position it is already prepared
    
    print("Generating GIF...")
    images_for_animation = [ imageio.imread(image_path) for image_path in image_paths]
    imageio.mimsave(f'NBody_Simulation_{exp_name}.gif', images_for_animation, fps=fps)
    
    print("Erasing mess...")
    for image_path in image_paths:
        os.remove(image_path)
    os.rmdir('./temp/')
    print("\nDone!!")
    

def plot_N_particles_and_save_it(fig, output_image_path, positions, velocities, t,
                                 xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, dpi=150, fps=10, show_frames=False):
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    cmap = matplotlib.cm.get_cmap('hsv') # to have the same colors of th trace and the body
    
    # Plot particles
    ax.scatter3D(positions[0,:,0], positions[0,:,1], positions[0,:,2], 
                 c=cmap(np.arange(positions.shape[1])/positions.shape[1]), s=50)
    
    # plot velocity vectors
    ax.quiver(positions[0,:,0], positions[0,:,1], positions[0,:,2], 
              velocities[:,0], velocities[:,1], velocities[:,2], length=0.2, normalize=True)
    
    
    # plot the traces
    for j in range(positions.shape[1]): # for each particle its trace separately
        ax.plot3D(positions[:,j,0], positions[:,j,1], positions[:,j,2], '-', c=cmap(j/positions.shape[1]) )
    
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.set_zlim((zmin, zmax))
    ax.set_title(f"{positions.shape[1]} Body simulation time t={t:4.3}")

    plt.savefig(output_image_path, dpi=dpi) # choose the resolution of the images with dpi
    # If you want to see the figures uncomment this
    if show_frames:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    


if __name__ == "__main__":
    example = int(input("Choose 1 or 2 to see different examples: "))
    show = bool(int(input("Do you want to see the frames while generating them? This is slightly slower. If yes write 1, else 0: ")))
    if show:
        plt.ion()
    if example==1:
        '''Example 1: Cool 3 Body Orbit'''
        print("Example 1: Cool 3 Body Stable Orbit")
        run_N_Body_simulator(N=3, positions_now=np.array([[0.97000436,-0.24308753,0], 
                                                    [-0.97000436,+0.24308753,0], 
                                                    [0,0,0]
                                                    ]), 
                        velocities_now=np.array([[0.93240737/2, 0.86473146/2, 0], 
                                                [0.93240737/2, 0.86473146/2, 0],
                                                [-0.93240737, -0.86473146, 0]]),
                        masses=[1, 1, 1,], 
                        additional_parameters=[0, 0, 0], 
                        force_list=[gravity_force_on_particle_1_by_2, coulomb_force_on_particle_1_by_2],
                        t0=0, tf=7, timeIts=300, plotEvery=5,
                        limits={'xmin':-1, 'xmax':1, 'ymin':-1, 'ymax':1, 'zmin':-1, 'zmax':1},
                        J_trace=30, exp_name="Example1", show_frames=show
                        )
    else:
        '''Example 2: Random 50 Body orbit'''
        print("Example 2: Random 50 body orbit\nRandom masses and charges - Gravity+Coulomb force")
        n=50
        run_N_Body_simulator(N=n, positions_now=np.random.randn(n, 3)/2, 
                        velocities_now=np.random.randn(n, 3),
                        masses=np.abs(np.random.randn(n)), 
                        additional_parameters=np.random.randn(n), 
                        force_list=[gravity_force_on_particle_1_by_2, coulomb_force_on_particle_1_by_2],
                        t0=0, tf=1, timeIts=150, plotEvery=1,
                        limits={'xmin':-2, 'xmax':2, 'ymin':-2, 'ymax':2, 'zmin':-2, 'zmax':2},
                        J_trace=100, exp_name="Example2", show_frames=show
                        )

