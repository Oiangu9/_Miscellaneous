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
import json


def ask_user_return_x_v_initial_conditions(initial_conds):
    print("Authors:\n")
    for i, author in enumerate(initial_conds.keys()):
        print(f"({i}) {author}")
    author = list(initial_conds.keys())[int(input("Choose the Author number:"))]

    print("\nConditions:\n")
    for i, cond in enumerate( initial_conds[author].keys()):
        print(f"({i}) {cond}")
    condition = list(initial_conds[author].keys())[int(input("Choose the Condition type:"))]

    print("\nMode:\n")
    for i, mode in enumerate( initial_conds[author][condition].keys()):
        print(f"({i}) {mode}")
    mode = list(initial_conds[author][condition].keys())[int(input("Choose the Mode:"))]
    
    selected_condition_dict = initial_conds[author][condition][mode]
    
    x = np.concatenate( (np.array( selected_condition_dict['x'] ), np.array([[0],[0],[0]])), axis=1 )
    v = np.concatenate( (np.array( selected_condition_dict['v'] ), np.array([[0],[0],[0]])), axis=1 )
    return x,v, f"{condition}_{mode}_{author}"

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
    show = bool(int(input("Do you want to see the frames while generating them? This is slightly slower. If yes write 1, else 0: ")))
    print("\n\nWelcome to the 3 Body Simulator!\nToday's star dish is (0) Šuvakov -> (3) IVa - Moth I -> (0)\nBut you can choose whatever you prefer from our menu:\n\n")
    with open("threeBodyInitialConditions.json", 'r') as file:
        initial_conds = json.load(file)
        
    x, v, exp_name = ask_user_return_x_v_initial_conditions(initial_conds)
    run_N_Body_simulator(N=3, positions_now=x, velocities_now=v,
                    masses=[1, 1, 1], 
                    additional_parameters=[0, 0, 0], 
                    force_list=[gravity_force_on_particle_1_by_2],
                    t0=0, tf=12, timeIts=60000, plotEvery=1000,
                     limits={'xmin':-1.5, 'xmax':1.5, 'ymin':-1.5, 'ymax':1.5, 'zmin':-1.5, 'zmax':1.5},
                     J_trace=50000, exp_name=exp_name, fps=7, dpi=70)
