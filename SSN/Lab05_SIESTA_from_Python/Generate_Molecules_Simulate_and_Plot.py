# Code Based on the SISL - SIESTA tutorial
# https://sisl.readthedocs.io/en/latest/tutorials/tutorial_siesta_1.html

import numpy as np
import matplotlib.pyplot as plt
import os, sys
import sisl
from Pseudopotential_file_generator import *

def string_to_float_list_of_3lists(string):
    '''Converts the list of 3 lists as a string to an 
    actual list of lists. 
    '''
    if string in ["[]", "[[]]"]:
        return []
    string = string.split("[[")[-1].split("]]")[0].split(",")
    l=[]
    for k,st in enumerate(string):
        if k%3==0:
            l.append([])
        l[-1].append(st.split("[")[-1].split("]")[0].split(",")[0])
    lout=[]
    for sub in l:
        subb = []
        for el in sub:
            subb.append(float(el))
        lout.append(subb)
    return lout

def get_max_diam(ls):
    max_p = -np.inf
    min_n = np.inf
    for l in ls:
        for sub in l:
            for k in sub:
                if k<min_n:
                    min_n=k
                if k>max_p:
                    max_p=k
    return max_p-min_n
    
    

if __name__ == "__main__":
    if str(sys.argv[1]) in ["-benzene", "-b"]:
        exp_label="Benzene"
        # Define positions of atoms
        CC = 1.39 #A
        CH = 1.09 #A
        positions_C = [
            [0, CC, 0], [0, -CC, 0],
            [np.sqrt(3)/2*(CC), (CC)/2, 0], [np.sqrt(3)/2*(CC), -(CC)/2, 0],
            [-np.sqrt(3)/2*(CC), (CC)/2, 0], [-np.sqrt(3)/2*(CC), -(CC)/2, 0]
        ]

        positions_H = [
            [0, CC+CH, 0], [0, -(CC+CH), 0],
            [np.sqrt(3)/2*(CC+CH), (CC+CH)/2, 0], [np.sqrt(3)/2*(CC+CH), -(CC+CH)/2, 0],
            [-np.sqrt(3)/2*(CC+CH), (CC+CH)/2, 0], [-np.sqrt(3)/2*(CC+CH), -(CC+CH)/2, 0],
            ]
        positions_N=[]
        positions_O=[]
        orbs="SZP"
    else:
        exp_label = str(sys.argv[5])
        positions_C = string_to_float_list_of_3lists(sys.argv[1])
        positions_H = string_to_float_list_of_3lists(sys.argv[2])
        positions_N = string_to_float_list_of_3lists(sys.argv[3])
        positions_O = string_to_float_list_of_3lists(sys.argv[4])
        try:
            orbs=sys.argv[6]
        except:
            orbs = "SZP"

    os.makedirs(exp_label, exist_ok=True)
    os.chdir(exp_label)

    cell_side=6*get_max_diam([positions_C, positions_H, positions_N, positions_O])
    # Generate molecule instance
    benzene = sisl.Geometry(positions_C+positions_H+positions_N+positions_O,
                [sisl.Atom('C')]*len(positions_C)+[sisl.Atom('H')]*len(positions_H)+[sisl.Atom('N')]*len(positions_N)+[sisl.Atom('O')]*len(positions_O),
                sc=sisl.SuperCell(cell_side, origin=[-cell_side/2] * 3))
    # Sanity check
    sisl.plot(benzene)
    plt.show()

    # Generate SIESTA configuration file
    print("\n\n>> Generating SIESTA configuration Files...")
    with open(f'RUN_{exp_label}.fdf', 'w') as f:
        f.write(f"""%include STRUCT_{exp_label}.fdf
                    SystemLabel siesta_{exp_label}
                    PAO.BasisSize {orbs}
                    MeshCutoff 250. Ry
                    CDF.Save true
                    CDF.Compress 9
                    SaveHS true
                    SaveRho true
                    """)
    # Generate molecule structure file
    benzene.write(f"STRUCT_{exp_label}.fdf")

    # Generate Pseudopotential Files
    print("\n\n>> Generating Pseudopotential Files...")
    generate_C_psf("./C.psf")
    generate_H_psf("./H.psf")
    if len(positions_N)!=0:
        generate_N_psf("./N.psf")
    if len(positions_O)!=0:
        generate_O_psf("./O.psf")

    # Run SIESTA
    # first change the conda environment
    print("\n\n>> Running Siesta...")
    com = f"siesta RUN_{exp_label}.fdf > siesta_terminal_log.txt"
    os.system(com)

    # Import results
    print("\n\n>> Importing Results...")
    fdf = sisl.get_sile(f'RUN_{exp_label}.fdf')
    H = fdf.read_hamiltonian()
    # Create a short-hand to handle the geometry
    benzene = H.geometry
    print(f"\n\n\nObtained result geometry : \n\n{H} \n\n")

    # We plot the fitted orbitals
    print("\n\n>> Generating Orbital Plots (and their .cube versions) for two example atoms...")
    plot_atom(benzene.atoms[0])
    plot_atom(benzene.atoms[-1])
    plt.show()


    print("\n\n>> Generating Molecular Orbiotals and .cube versions for HOMO and LUMO...")
    #Function integrate
    def integrate(g):
        print('Real space integrated wavefunction: {:.4f}'.format((np.absolute(g.grid) ** 2).sum() * g.dvolume))


    #Eigenstates
    es = H.eigenstate()

    # We specify an origin to center the molecule in the grid
    benzene.sc.origin = [-cell_side/2]*3

    # Reduce the contained eigenstates to only the HOMO and LUMO
    # Find the index of the smallest positive eigenvalue
    idx_lumo = (es.eig > 0).nonzero()[0][0]
    es = es.sub([idx_lumo - 1, idx_lumo])
    g = sisl.Grid(0.2, sc=benzene.sc)

    #HOMO
    es.sub(0).wavefunction(g)
    integrate(g)
    g.write('HOMO.cube')

    g.fill(0) # reset the grid values to 0

    #LUMO
    es.sub(1).wavefunction(g)
    integrate(g)
    g.write('LUMO.cube')
    
    # tried to generate the density file as well...but
    print("\n\n>> Generating Density .cube File....")
    density = sisl.get_sile(f"siesta_{exp_label}.nc").read_grid(name='Rho')
    density.set_geometry(fdf.read_geometry())
    density.write("DENSITY.cube")
    #os.system("vmd DENSITY.cube")
    
    
    # call vmd
    os.system("vmd HOMO.cube")
    
