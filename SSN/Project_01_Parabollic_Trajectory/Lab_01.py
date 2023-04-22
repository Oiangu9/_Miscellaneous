#
# Example parabolic motion
#

# Here we import the mathematical library and the plots library
import numpy as np
import matplotlib.pyplot as plt
 
#
# FUNCTION DRAW A TRAJECTORY FOR PARABOLIC MOTION
# Input: velocity and angle 
#
def draw_trajectory(us, theta_deg, phi_deg):
    #convert angle in degrees to rad
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    #gravity acceleration in m/s2
    g = 9.8
    # Initialize figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    
    for u in us:
        # Time of flight
        t_flight = 2*u*np.sin(theta)/g
        # find time intervals
        intervals = np.arange(0, t_flight, 0.001)
        # create an empty list of x,y and z coordinates
        x = []
        y = []
        z = []
        #Do a loop over time calculating the coordinates
        for t in intervals:
            x.append(u*np.cos(theta)*np.cos(phi)*t)
            y.append(u*np.cos(theta)*np.sin(phi)*t)
            z.append(u*np.sin(theta)*t - 0.5*g*t*t)
        #Plot the results
        cmap = ax.scatter3D(x,y,z, label=f"t0 Speed {u}m/s; tf={t_flight:.2}s") # c=intervals,cmap='winter', 
    #fig.colorbar(cmap, ax=ax)
    ax.set_xlabel("x distance (m)")
    ax.set_ylabel("y distance (m)")
    ax.set_zlabel("z distance (m)")
    ax.set_title(f"Projectile motion\nInitial theta={theta_deg:.4}deg phi={phi_deg:.4}deg")
    ax.legend()
    plt.show()

#--------------------------------------------------------------------------------
# Main Program: give specific values and call to the function draw_trajectory
#--------------------------------------------------------------------------------

print("Parabolic motion of a projectile\n")

#Ask the user for angle
print("Enter desired launch polar angle in degrees (recommended 45 degrees):")
theta=float(input())
print("Enter desired launch azimuthal angle in degrees (for example 0 degrees):")
phi=float(input())


# list of three different initial velocity in m/s
u_list = [20, 40, 60]

draw_trajectory(u_list, theta, phi)

