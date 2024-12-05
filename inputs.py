import numpy as np
import pyscf.gto as gto

def coordinate():
    """
    Description: 
                This code is meant to provide the user with the coordinates of a water molecules aranged in a circular configuration.   
    
                # Inputs required: 1) Oxygen-Oxygen distance(in Angstrom)
                                  2) number of water molecules to constitute the circular configuration
                                  (Note: Be careful as the number of water molecules must be at least 3)
    
                # Output: coordinate of the water molecules in the following format:
                                    Atom    x-coordinate        y-coordinate        z-coordinate
                                     O      1.73205081          0.00000000          0.00000000
                                     H      0.90257168          0.47890000          0.00000000
                                     H      2.31071918          0.52265508          0.55619729
    
                           This same code have also been written such that it write the coordinates in a text file rather than giving it as an output in another file named coordinate.py             
    """              
    

    roo = float(input("O-O distance in Angstrom:  "))
    n = int(input("Number of water in ring:   "))

    rc = roo / (2 * np.sin(np.pi / n))

    roh = 0.9578  # O-H distance
    ahoh = 104.5  # H-O-H angle in each monomer
    apln = 35.5  # Angle that one of the H's makes with O plane
    assert n >= 3
    an = 180. * (n - 2) / n   # Interior angle of n-gon
    a0 = 180. - an / 2.

    # Solve for at
    f0 = np.cos(ahoh * np.pi / 180.) / np.cos(apln * np.pi / 180.)
    at = a0 - (180 / np.pi) * np.arccos(f0)

    # Some useful rotation matrices
    R1 = np.eye(3)
    R1[0, 0] = np.cos(a0 * np.pi / 180.)
    R1[1, 1] = np.cos(a0 * np.pi / 180.)
    R1[0, 1] = -np.sin(a0 * np.pi / 180.)
    R1[1, 0] = np.sin(a0 * np.pi / 180.)
    R2 = np.eye(3)
    R2[0, 0] = np.cos(at * np.pi / 180.)
    R2[1, 1] = np.cos(at * np.pi / 180.)
    R2[0, 1] = -np.sin(at * np.pi / 180.)
    R2[1, 0] = np.sin(at * np.pi / 180.)
    R3 = np.eye(3)
    R3[0, 0] = np.cos(apln * np.pi / 180.)
    R3[2, 2] = np.cos(apln * np.pi / 180.)
    R3[0, 2] = -np.sin(apln * np.pi / 180.)
    R3[2, 0] = np.sin(apln * np.pi / 180.)

    # Atom coordinate array
    atom_coords = []
    for i in range(n):
        theta = i * (2 * np.pi / n)
        R0 = np.eye(3)
        R0[0, 0] = np.cos(theta)
        R0[1, 1] = np.cos(theta)
        R0[0, 1] = -np.sin(theta)
        R0[1, 0] = np.sin(theta)

        t = np.zeros((3, 1))
        t[0] = 1.

        # O
        O_coords = np.dot(R0, t)[:, 0] * rc
        atom_coords.append(('O', O_coords))

        # H1
        R01 = np.dot(R1, R0)
        H1_coords = O_coords + np.dot(R01, t)[:, 0] * roh
        atom_coords.append(('H', H1_coords))

        # H2
        R302 = np.dot(R2, np.dot(R0, R3))
        H2_coords = O_coords + np.dot(R302, t)[:, 0] * roh
        atom_coords.append(('H', H2_coords))

    return atom_coords

def mol():
    """
    Description:
                This function will be responsible for the taking the output from the coordinate() function above and making individual water molecule. 
                The basis set and units for the calculation is also written here. So, be careful to check that before running any calculation. 
    """
    atom_coords = coordinate()
    #print(atom_coords)
    molecules = []
    for i in range(0, len(atom_coords), 3):
        coords = []
        for j in range(3):
            atom, (x, y, z) = atom_coords[i + j]
            coords.append([atom, (float(x), float(y), float(z))])
        molx = gto.M(atom=coords, basis='aug-cc-pvdz', unit='angstrom')
        molecules.append(molx)
    return molecules

