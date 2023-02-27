# ----------------------------
# 3D Time Independent Schrödinger Equation Solver (Hydrogen Atom example)
# ----------------------------
# Generalization of the finite differences method developed by Truhlar JCP 10 (1972) 123-132
#
# by Xabier Oiangurne Asua
#
# The Schrödinger Equation is assumed to be in atomic units
#  
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from time import time

# ------------
# ROUTINES
# ------------
#Potential as a function of position
def V(x,y,z):
    return -1.0/np.sqrt(x**2+y**2+z**2) # Hidrogen Atom


#Discretized Hamiltonian as a Sparse Matrix
def get_discrete_H(Nx, Ny, Nz, dx, dy, dz, xs, ys, zs):
    main_diagonal = -0.5*(-2.0)*(1/dx**2+1/dy**2+1/dz**2)*np.ones(Nx*Ny*Nz)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                main_diagonal[i*Nz*Ny+j*Nz+k] += V(xs[i], ys[j], zs[k])
    z_diagonals = -0.5*1.0/dz**2*np.ones(Nx*Ny*Nz-1)
    y_diagonals = -0.5*1.0/dy**2*np.ones(Nx*Ny*Nz-Nz)
    x_diagonals = -0.5*1.0/dx**2*np.ones(Nx*Ny*Nz-Nz*Ny)
    # There are some zeros we need to place in these diagonals
    for j in range(Ny-1):
        z_diagonals[(j+1)*Nz-1] = 0
    for i in range(Nx-1):
        y_diagonals[ (i+1)*Nz*Ny-Nz:(i+1)*Nz*Ny ] = 0
    
    return sp.diags( diagonals=
        [main_diagonal, z_diagonals, z_diagonals,y_diagonals, y_diagonals, x_diagonals, x_diagonals],
        offsets=[0, 1, -1, Nz, -Nz, Nz*Ny, -Nz*Ny] )

use_saved=1
#-------------------------
# Main program
#-------------------------
# Number of Eigenvalues
num_eig=5

# Intervals for calculating the wave function [-L/2,L/2] (in atomic units)
Ls = np.array([ 8,8,8 ]) # (Lx, Ly, Lz) Bohr Radii
xlowers = -Ls/2.0
xuppers = Ls/2.0

# Number of points to be used per dimension
Ns = [200,200,200] # (Nx, Ny, Nz)

# Increments to be used per dimension
dxs = [(xuppers[j]-xlowers[j])/(Ns[j]-1) for j in range(3)] # (dx, dy, dz)


#Create coordinates at which the solution will be calculated
nodes = [np.linspace(xlowers[j], xuppers[j], Ns[j]) for j in range(3)] # (xs, ys, zs)

print("> Grid Settings:")
print(f"  Using Nx={Ns[0]} Ny={Ns[1]} Nz={Ns[2]} grid points.")
print(f"  Using dx={dxs[0]:.4} dy={dxs[1]:.4} dz={dxs[2]:.4} increments (atomic units)\n")

file1="eigenVectors.npy"
file2="eigenValues.npy"

if not use_saved:
    #Calculation of discrete form of Schrodinger Equation
    t0=time()
    print("> Calculating discretized Hamiltonian...")
    H=get_discrete_H(*Ns, *dxs, *nodes)
    t1=time()
    print(f"  Done! Taken {t1-t0:.4}s\n")

    # Diagonalize the matrix F
    print("> Diagonalizing Hamiltonian...")
    eigenValues, eigenVectors = lg.eigsh(H, k=num_eig, which='SM', maxiter=200, tol=0.01)
    print(f"  Done! Taken {time()-t1:.4}s\n")
    
    print("> Normalising the eigenstates...")
    # We will normalise the states "the lazy way":
    # instead of integrating each dimension with its own discretization, we will use the average one
    dx_av = np.mean(dxs)
    for k in range(0, num_eig):
        eigenVectors[:,k] /= np.sqrt((np.dot(eigenVectors[:,k], eigenVectors[:,k])*dx_av**3))
    
    # Reshape the eigen-vectors to a 3D array each (in their natural indexing)
    eigenStates=eigenVectors.T.reshape((num_eig, *Ns))

    del eigenVectors
    print("  Done!\n")

    np.save(file1, eigenStates)
    np.save(file2, eigenValues)
else:
    eigenStates=np.load(file1)
    eigenValues=np.load(file2)


# Print Energy Values
print("> Resulting Energies:")
for k in range(0,num_eig):
	print(f"State {k} -> Energy = {eigenValues[k]:.4} Hartrees")


#Plot Wave functions
print("> Plotting...")
every=2 # Only take one data point every this number in each axis to plot
grid = np.array(np.meshgrid(*nodes))[:,::every, ::every, ::every]
print(grid.shape)
for j in range(0, num_eig):
    fig = plt.figure( figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    colormap = ax.scatter3D(*grid, c=eigenStates[j, ::every, ::every, ::every], cmap='seismic',
            s=0.003, alpha=0.4 ) #, antialiased=True)
    fig.colorbar(colormap, fraction=0.04, location='left')
    ax.set_xlabel("x (Bohr Radii)")
    ax.set_ylabel("y (Bohr Radii)")
    ax.set_zlabel("z (Bohr Radii)")
    ax.set_title(f"Real Part of the {j}-th Energy Eigenstate")
    plt.savefig(f"Re_Eig_{j}_Ener_{eigenValues[j]:.4}_N_{Ns[0]}_L_{Ls[0]}.png")
    #plt.show()
    
    fig = plt.figure( figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    colormap = ax.scatter3D(*grid, c=np.abs(eigenStates[j, ::every, ::every, ::every])**2, cmap='hot',
            s=0.003, alpha=0.4 ) #, antialiased=True)
    fig.colorbar(colormap, fraction=0.04, location='left')
    ax.set_xlabel("x (Bohr Radii)")
    ax.set_ylabel("y (Bohr Radii)")
    ax.set_zlabel("z (Bohr Radii)")
    ax.set_title(f"Magnitude Squared of the {j}-th Energy Eigenstate")
    plt.savefig(f"pdf_Eig_{j}_Ener_{eigenValues[j]:.4}_N_{Ns[0]}_L_{Ls[0]}.png")
    #plt.show()
    print(f"{j}-th done!")
