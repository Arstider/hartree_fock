
import multiprocessing
import sys
import numpy as np
from pyscf import gto,scf
from scipy import special,linalg
from scipy.linalg import cholesky,solve_triangular
from scipy.optimize import minimize
#import matplotlib.pyplot as plt
#from tabulate import tabulate

class HFsolver:

  def __init__(self):
    self.molecule=None
    self.density=None
    self.orbital=None
    self.fock=None
    self.energy=None
    #self.gradient=None

  def run_HF(self,molecule):

    self.molecule=molecule
    nup = molecule.nelectron//2

    enr = molecule.energy_nuc()

    # Calculating the integrals
    nb = molecule.nao_nr()
    S = molecule.intor_symmetric('int1e_ovlp')
    T = molecule.intor_symmetric('int1e_kin')
    V = molecule.intor_symmetric('int1e_nuc')
    i2s = molecule.intor('int2e', aosym='s1')

    # Calculating the Transformation matrices
    v, D = np.linalg.eigh(S)
    s = 1./np.sqrt(v)
    X = D*s @ D.T  # Transformation matrices

    # Initial guess of the Primitive Gaussian
    mol_scf = scf.RHF(molecule)
    P = mol_scf.get_init_guess()

    #2e portion of Fock matrix
    i = S.shape[0]
    G = np.zeros((i, i))  # 2e portion of Fock matrix
    G += np.einsum('ls,mnsl->mn',P,i2s)
    G -= np.einsum('ns,mnsl->ml',P,i2s)*0.5

    # Calculating the core Hamiltonian matrix#
    H_core = T + V  # Core Hamiltonian

    # Fock matrix
    F = H_core + G #Fock matrix
    F_transformed = X.T @ F @ X
    e, C_prime = np.linalg.eigh(F_transformed)
    C = X @ C_prime #orbitals
    # C is initial guess orbitals

    #Thouless projection
    noc = nup
    nvt = self.molecule.nao_nr() - nup
    np.random.seed(0)

    Z=np.random.rand(nvt,noc)*0.01

    def orbital_update (Z,C0):
      L=linalg.cholesky(np.eye(noc)+(Z.T)@Z,lower=True)
      U=linalg.cholesky(np.eye(nvt)+Z@Z.T,lower=True)

      L_inv=linalg.inv(L)
      U_inv=linalg.inv(U)

      C_occ=C[:,0:nup]
      C_vrt=C[:,nup:nb]

      # Apply the Thouless projection to each virtual orbital

      C_occ_new = (C_occ + C_vrt@Z)@L_inv.T

      # Apply the inverse Thouless projection to each virtual orbital

      C_vrt_new = (C_vrt - C_occ@Z.T)@U_inv.T
      C_new = np.hstack((C_occ_new,C_vrt_new))
      return C_new, L_inv, U_inv

    self.orbital= orbital_update(Z,C)

    def build_density (C):
      P = 2*C[:,:nup]@C[:,:nup].T
      return P

    self.density=build_density(C)

    def build_fock (P):
      G = np.zeros_like(P)
      G += np.einsum('ls,mnsl->mn',P,i2s)
      G -= np.einsum('ns,mnsl->ml',P,i2s)*0.5

      F = H_core + G #Fock matrix
      return F

    self.fock=build_fock(P)

    def hf_energy (P,F):
      # Energy Calculation
      E_elec = 0.5*np.trace(P@(H_core + F))  # np.sum is being used to sum of all elements because we get a matrix after the calculation

      # Total energy
      E_total = E_elec + enr
      return E_total

    C_new, L_inv, U_inv = orbital_update (Z,C)
    P = build_density (C_new)
    F = build_fock (P)
    Gloc = 4*C_new[:,nup:].T@F@C_new[:,:nup]
    Ggl  = U_inv.T@Gloc@L_inv

    def objective(Z):
      Z=Z.reshape(nvt,noc)
      C_new,L_inv,U_inv=orbital_update(Z,C)
      P = build_density (C_new)
      F = build_fock (P)
      Gloc = 4*C_new[:,nup:].T@F@C_new[:,:nup]
      Ggl  = U_inv.T@Gloc@L_inv
      E=hf_energy(P,F)
      return E, Ggl.flatten()

    #self.gradient=objective

    Z0 = np.zeros_like(Z.flatten())
    result=minimize(objective,Z0,method='BFGS',jac=True)
    #print ('converged E for molecule = ', result.fun)

    Z=result.x.reshape(nvt,noc)
    C_new,L_inv,U_inv=orbital_update(Z,C)
    P = build_density (C_new)
    return C_new, P

HF=HFsolver()

def dimer(mol1,mol2,C1_initial=None,C2_initial=None):
  super_mol =gto.M()
  super_mol = mol1 + mol2
  molecules=[mol1,mol2]

  S = super_mol.intor_symmetric('int1e_ovlp')
  T = super_mol.intor_symmetric('int1e_kin')
  V = super_mol.intor_symmetric('int1e_nuc')
  H_core = T+V
  enr=super_mol.energy_nuc()

  def initial(mol):
    nb = mol.nao_nr()
    noc = mol.nelectron // 2
    nvt = mol.nao_nr() - noc
    return nb, nvt, noc

  def get_orbitals (Z,C,noc,nvt):
    L=linalg.cholesky(np.eye(noc)+(Z.T)@Z,lower=True)
    U=linalg.cholesky(np.eye(nvt)+Z@Z.T,lower=True)
    L_inv=linalg.inv(L)
    U_inv=linalg.inv(U)

    C_occ=C[:,:noc]
    C_vrt=C[:,noc:]
    C_occ_new = (C_occ + C_vrt@Z)@L_inv.T
    C_vrt_new = (C_vrt - C_occ@Z.T)@U_inv.T

    C_new = np.hstack((C_occ_new,C_vrt_new))
    return C_new, L_inv, U_inv,

  def density(C,noc):
    P = 2*C[:,:noc]@C[:,:noc].T
    return P

  def get_G(P1, P2,):
    nb1, nb2 = P1.shape[0], P2.shape[0]

    _P1 = np.block([[P1, np.zeros((nb1, nb2))], [np.zeros((nb2, nb1)), np.zeros((nb2, nb2))]])
    _P2 = np.block([[np.zeros((nb1, nb1)), np.zeros((nb1, nb2))], [np.zeros((nb2, nb1)), P2]])

    JJ, KK = scf.rhf.get_jk(super_mol, dm=(_P1, _P2))

    J1, J2 = JJ[0], JJ[1]
    K1, K2 = KK[0], KK[1]

    G1 = J1 - 0.5 * K1 + J2
    G2 = J2 - 0.5 * K2 + J1

    return G1[:nb1, :nb1], G2[nb1:, nb1:]

  def get_fock(G1,G2):
    f1=H_core[:nb1,:nb1]+G1
    f2=H_core[nb1:,nb1:]+G2
    return f1,f2

  def get_energy(P1,P2,G1,G2):
    Eer_dimer=np.trace(H_core[:nb1,:nb1]@P1)+np.trace(H_core[nb1:,nb1:]@P2)+0.5*np.trace(G1@P1)+0.5*np.trace(G2@P2)
    E=Eer_dimer+enr
    return E

  nb1, nvt1, noc1,  = initial(mol1)
  nb2, nvt2, noc2,  = initial(mol2)

  if C1_initial is None and C2_initial is None:
    C1,P1=HF.run_HF(mol1)
    C2,P2=HF.run_HF(mol2)
  else:
    C1 = C1_initial if C1_initial is not None else C1
    C2 = C2_initial if C2_initial is not None else C2
    P1=density(C1,noc1)
    P2=density(C2,noc2)

  def get_HLE():
    #Random variables remove later on
    nup = (super_mol.nelectron//2)
    nup1  = mol1.nelectron//2

    #not random
    c1,_=HF.run_HF(mol1)
    c2,_=HF.run_HF(mol2)

    C_tilda=linalg.block_diag(c1[:,:nup1],c2[:,:nup-nup1])
    X=C_tilda.T@S@C_tilda
    P= np.dot(C_tilda,linalg.inv(X)) @ C_tilda.T
    J,K = scf.rhf.get_jk(super_mol, dm=(P))
    G=J-0.5*K
    hle= 2*np.trace(H_core@P)+2*np.trace(np.dot(P,G))+enr
    return hle

  HL_E=get_HLE()
  #print(HL_E)


  np.random.seed(0)
  Z=np.zeros(nvt1*noc1+nvt2*noc2)
  Z1  = Z[:nvt1*noc1].reshape((nvt1,noc1,))
  Z2  = Z[nvt1*noc1:].reshape((nvt2,noc2,))

  C_new1, L_inv1, U_inv1, = get_orbitals(Z1,C1,noc1,nvt1)
  C_new2, L_inv2, U_inv2, = get_orbitals(Z2,C2,noc2,nvt2)
  P1  = density(C1,noc1)
  P2  = density(C2,noc2)

  print()

  G1,G2=get_G(P1,P2,)
  unphysical_1=get_energy(P1,P2,G1,G2)

  final_C_new1 = None
  final_C_new2 = None
  final_p1,final_p2=None,None

  def physical_energy(final_C_new1,final_C_new2):
    nup = (super_mol.nelectron//2)
    nup1  = mol1.nelectron//2
    C_tilda=linalg.block_diag(final_C_new1[:,:nup1],final_C_new2[:,:nup-nup1])
    #S= super_mol.intor_symmetric('int1e_ovlp')
    s=C_tilda.T@S@C_tilda

    P= np.dot(C_tilda,linalg.inv(s)) @ C_tilda.T
    #P=np.dot(C,C_tilda.T)
    J,K = scf.rhf.get_jk(super_mol, dm=(P))
    G=J-0.5*K
    hf= 2*np.trace(H_core@P)+2*np.trace(np.dot(P,G))+enr
    #print(hf)
    return hf

  def objective(Z):
    nonlocal final_C_new1, final_C_new2,final_p1,final_p2,iterations
    #Z=Z.reshape(nvt,noc)
    iterations+=1
    Z1  = Z[:nvt1*noc1].reshape((nvt1,noc1,))
    Z2  = Z[nvt1*noc1:].reshape((nvt2,noc2,))
    C_new1, L_inv1, U_inv1, = get_orbitals(Z1,C1,noc1,nvt1)
    C_new2, L_inv2, U_inv2, = get_orbitals(Z2,C2,noc2,nvt2)
    P1  = density(C_new1,noc1)
    P2  = density(C_new2,noc2)
    G1,G2=get_G(P1,P2,)
    f1,f2=get_fock(G1,G2)

    G_loc1=C_new1[:,noc1:].T@f1@C_new1[:,:noc1]
    G_gbl1=U_inv1.T@G_loc1@L_inv1

    G_loc2=C_new2[:,noc2:].T@f2@C_new2[:,:noc2]
    G_gbl2=U_inv2.T@G_loc2@L_inv2

    E=get_energy(P1,P2,G1,G2)

    Z=linalg.block_diag(Z1,Z2)
    final_C_new1  = C_new1
    final_C_new2  = C_new2
    final_p1=P1
    final_p2=P2
    physical=physical_energy(final_C_new1,final_C_new2)
    #print(f"For iteration {iterations}")
    #print(f"Unphysical Energy: {E}  |||||| Physical Energy : {physical}")
    return E, np.hstack((G_gbl1.flatten(),G_gbl2.flatten(),))

  iterations=0
  Z0 = np.zeros_like(Z.flatten())
  result=minimize(objective,Z0,method='BFGS',jac=True,options={'gtol':1e-9})
  E=result.fun

  #print(f"Final converged Unphysical energy : {result.fun}")

  hf=physical_energy(final_C_new1,final_C_new2)

  print(f"Unphysical energy at 1st iteration : {unphysical_1}|| Final convergence : {result.fun}")
  print(f"Physical energy at 1st iteration : {HL_E}|| Final convergence : {hf}")

  return E,hf,HL_E,final_C_new1,final_C_new2
  #return E,hf,HL_E

def parse_input(input_str):
    atoms = []
    for line in input_str.split('\n'):
        if line.strip():
            tokens = line.split()
            symbol = tokens[0]
            coords = list(map(float, tokens[1:4]))
            atoms.append([symbol, tuple(coords)])
    return atoms

def create_molecule(atoms, basis='cc-pvdz', unit='au'):
    mol = gto.M(atom=atoms, basis=basis, unit=unit)
    return mol

# Function to parse the output file and find the last completed iteration
def find_last_completed_iteration(file_path):
    last_completed_iteration = 0 try: with open(file_path, "r") as f: lines =
    f.readlines() for line in reversed(lines):
                if line.startswith("Result:"):
                    last_completed_iteration += 1
    except FileNotFoundError:
        pass
    return last_completed_iteration

# Function to perform dimer calculations
def perform_dimer_calculations(mol_str_list, output_file):
    # Get the index of the last completed iteration
    last_completed_iteration = find_last_completed_iteration(output_file)

    # Open file in append mode to append results
    with open(output_file, "a") as f:
        for i, (mol1_str, mol2_str) in enumerate(mol_str_list[last_completed_iteration:], start=last_completed_iteration):
            mol1_atoms = parse_input(mol1_str)
            mol1 = create_molecule(mol1_atoms, basis='aug-cc-pvtz')

            mol2_atoms = parse_input(mol2_str)
            mol2 = create_molecule(mol2_atoms, basis='aug-cc-pvtz')

            result = dimer(mol1, mol2)

            # Write the result to the file
            f.write(f"Iteration: {i+1}\n")
            f.write(f"Molecule 1: {mol1_str}\n")
            f.write(f"Molecule 2: {mol2_str}\n")
            f.write(f"Result: {result}\n")
            f.write("-" * 30 + "\n")

    # Printing confirmation message
    print("Dimer calculations completed and saved to", output_file)

# Define your molecule string list
mol_str_list = [
    ("""O -1.2506330 -0.0300900 -0.0114770
H -1.7816830 0.4139150 -0.6725950
H -1.8610820 -0.2136560 0.7025870
""",
"""O 1.2491430 -0.0119010 0.0166490
H 1.7614750 0.7525960 0.2798820
H 1.8932130 -0.6169280 -0.3512460
"""),
    ("""O -1.3006460 -0.0300330 -0.0111980
H -1.8319010 0.4114690 -0.6738250
H -1.9107660 -0.2104700 0.7039440
""",
"""O 1.2991430 -0.0119360 0.0165410
H 1.8118950 0.7534410 0.2763770
H 1.9427930 -0.6186900 -0.3492420
"""),
    ("""O -1.3506570 -0.0299770 -0.0109380
H -1.8821010 0.4092040 -0.6749560
H -1.9604770 -0.2075180 0.7051840
""",
"""O 1.3491440 -0.0119710 0.0164400
H 1.8622790 0.7542080 0.2731350
H 1.9924090 -0.6203110 -0.3473800
"""),
    ("""O -1.4006680 -0.0299220 -0.0106960
H -1.9322840 0.4071010 -0.6759990
H -2.0102120 -0.2047760 0.7063220
""",
"""O 1.3991440 -0.0120060 0.0163450
H 1.9126300 0.7549080 0.2701290
H 2.0420570 -0.6218080 -0.3456450
"""),
    ("""O -1.4506780 -0.0298690 -0.0104700
H -1.9824520 0.4051440 -0.6769620
H -2.0599680 -0.2022250 0.7073680
""",
"""O 1.4491440 -0.0120400 0.0162550
H 1.9629530 0.7555480 0.2673340
H 2.0917330 -0.6231950 -0.3440260
"""),
    ("""O -1.5006870 -0.0298170 -0.0102580
H -2.0326060 0.4033190 -0.6778550
H -2.1097420 -0.1998450 0.7083340
""",
"""O 1.4991450 -0.0120740 0.0161710
H 2.0132500 0.7561350 0.2647300
H 2.1414340 -0.6244820 -0.3425110
"""),
    ("""O -1.5506950 -0.0297670 -0.0100590
H -2.0827490 0.4016140 -0.6786850
H -2.1595340 -0.1976200 0.7092270
""",
"""O 1.5491450 -0.0121070 0.0160900
H 2.0635250 0.7566760 0.2622990
H 2.1911580 -0.6256790 -0.3410930
"""),
    ("""O -1.6007020 -0.0297190 -0.0098720
H -2.1328810 0.4000170 -0.6794580
H -2.2093410 -0.1955360 0.7100550
""",
"""O 1.5991450 -0.0121390 0.0160140
H 2.1137790 0.7571740 0.2600240
H 2.2409020 -0.6267960 -0.3397610
"""),
    ("""O -1.6507090 -0.0296720 -0.0096960
H -2.1830040 0.3985190 -0.6801790
H -2.2591610 -0.1935810 0.7108260
""",
"""O 1.6491450 -0.0121700 0.0159420
H 2.1640150 0.7576360 0.2578910
H 2.2906640 -0.6278400 -0.3385090
"""),
    ("""O -1.7007160 -0.0296260 -0.0095300
H -2.2331180 0.3971110 -0.6808530
H -2.3089940 -0.1917430 0.7115440
""",
"""O 1.6991450 -0.0122010 0.0158740
H 2.2142350 0.7580640 0.2558880
H 2.3404420 -0.6288170 -0.3373310
"""),
    ("""O -1.7507220 -0.0295820 -0.0093730
H -2.2832240 0.3957860 -0.6814850
H -2.3588380 -0.1900130 0.7122150
""",
"""O 1.7491450 -0.0122300 0.0158090
H 2.2644400 0.7584620 0.2540030
H 2.3902350 -0.6297350 -0.3362190
"""),
    ("""O -1.8007270 -0.0295400 -0.0092240
H -2.3333240 0.3945370 -0.6820780
H -2.4086920 -0.1883810 0.7128430
""",
"""O 1.7991450 -0.0122590 0.0157460
H 2.3146310 0.7588320 0.2522270
H 2.4400420 -0.6305970 -0.3351690
"""),
    ("""O -1.8507320 -0.0294990 -0.0090840
H -2.3834170 0.3933570 -0.6826350
H -2.4585540 -0.1868390 0.7134320
""",
"""O 1.8491450 -0.0122870 0.0156870
H 2.3648110 0.7591780 0.2505510
H 2.4898600 -0.6314090 -0.3341760
"""),
    ("""O -1.9007370 -0.0294590 -0.0089500
H -2.4335040 0.3922420 -0.6831600
H -2.5084260 -0.1853810 0.7139850
""",
"""O 1.8991450 -0.0123150 0.0156310
H 2.4149790 0.7595020 0.2489660
H 2.5396900 -0.6321740 -0.3332360
"""),
    ("""O -1.9507420 -0.0294210 -0.0088230
H -2.4835860 0.3911850 -0.6836560
H -2.5583050 -0.1839990 0.7145060
""",
"""O 1.9491450 -0.0123410 0.0155770
H 2.4651370 0.7598060 0.2474660
H 2.5895290 -0.6328980 -0.3323440
"""),
    ("""O -2.0007460 -0.0293830 -0.0087020
H -2.5336630 0.3901830 -0.6841240
H -2.6081900 -0.1826890 0.7149970
""",
"""O 1.9991450 -0.0123670 0.0155250
H 2.5152860 0.7600900 0.2460440
H 2.6393780 -0.6335820 -0.3314970
"""),
    ("""O -2.0507500 -0.0293470 -0.0085870
H -2.5837360 0.3892320 -0.6845670
H -2.6580830 -0.1814450 0.7154600
""",
"""O 2.0491440 -0.0123920 0.0154750
H 2.5654270 0.7603580 0.2446950
H 2.6892360 -0.6342300 -0.3306920
"""),
    ("""O -2.1007540 -0.0293130 -0.0084780
H -2.6338050 0.3883280 -0.6849860
H -2.7079810 -0.1802620 0.7158990
""",
"""O 2.0991440 -0.0124160 0.0154280
H 2.6155590 0.7606110 0.2434120
H 2.7391010 -0.6348450 -0.3299260
"""),
    ("""O -2.1507570 -0.0292790 -0.0083730
H -2.6838700 0.3874680 -0.6853850
H -2.7578850 -0.1791360 0.7163140
""",
"""O 2.1491440 -0.0124400 0.0153820
H 2.6656850 0.7608490 0.2421910
H 2.7889740 -0.6354300 -0.3291950
"""),
    ("""O -2.2007610 -0.0292460 -0.0082730
H -2.7339320 0.3866480 -0.6857630
H -2.8077930 -0.1780630 0.7167070
""",
"""O 2.1991440 -0.0124630 0.0153390
H 2.7158040 0.7610740 0.2410290
H 2.8388530 -0.6359850 -0.3284990
"""),
    ("""O -2.2507640 -0.0292150 -0.0081770
H -2.7839900 0.3858660 -0.6861220
H -2.8577070 -0.1770390 0.7170810
""",
"""O 2.2491440 -0.0124850 0.0152970
H 2.7659160 0.7612860 0.2399200
H 2.8887390 -0.6365140 -0.3278340
"""),
]


output_file = "dimer_output.txt"

# Call the function to perform dimer calculations
perform_dimer_calculations(mol_str_list, output_file)

