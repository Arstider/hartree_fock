import argparse
import sys
import numpy as np
from pyscf import gto,scf
from scipy import special,linalg
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
from block_diagonalize import *
from input_system import *  
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

  def run_HF(self,molecule,return_energy_only=False):

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
      L=cholesky(np.eye(noc)+(Z.T)@Z,lower=True)
      U=cholesky(np.eye(nvt)+Z@Z.T,lower=True)

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
    print ('converged E for molecule = ', result.fun)

    if return_energy_only:
      return result.fun
    else:
      Z=result.x.reshape(nvt,noc)
      C_new,_,_=orbital_update(Z,C)
      P = build_density (C_new)
      return C_new, P,

HF=HFsolver()

def generalized_dimer(molecules,guess_C=None,only_HL_energy=False):
  def initial(mol):
    nb = mol.nao_nr()
    noc = mol.nelectron // 2
    nvt = mol.nao_nr() - noc
    return nb, nvt, noc

  def get_orbitals (Z_list,C_list,noc_list,nvt_list):
    orbital,L_inv, U_inv=[],[],[]

    for i in range(len(Z_list)):
      Z, C, noc, nvt = Z_list[i], C_list[i], noc_list[i], nvt_list[i]
      L=cholesky(np.eye(noc)+(Z.T)@Z,lower=True)
      U=cholesky(np.eye(nvt)+Z@Z.T,lower=True)
      l_inv = solve_triangular(L, np.eye(noc), lower=True)
      u_inv = solve_triangular(U, np.eye(nvt), lower=True)
      L_inv.append(l_inv)
      U_inv.append(u_inv)

      C_occ, C_vrt = C[:, :noc], C[:, noc:]

      C_occ_new = (C_occ + C_vrt@Z)@l_inv.T
      C_vrt_new = (C_vrt - C_occ@Z.T)@u_inv.T
      orbital.append(np.hstack((C_occ_new,C_vrt_new)))

    return orbital, L_inv, U_inv,

  def get_density(C,noc):
    density=[]

    for C,noc in zip(C,noc):
      P = 2*C[:,:noc]@C[:,:noc].T
      density.append(P)

    return density

  def initialize_molecules(molecules,custom_Z=None):
    guess, basis,  = [], [],
    np.random.seed(0)
    super_mol = gto.M()
    n = 0

    for mol in molecules:
      super_mol += mol
      guess_C, _, = HF.run_HF(mol)
      guess.append(guess_C)
      nb, nvt, noc = initial(mol)
      basis.append((nb, nvt, noc,))
      n+=1

    return super_mol, guess, basis, n,

  def get_G(density):
    _density = []
    d = 0
    nb_total = 0
    i=0

    # Calculate total number of elements
    for P in density:
      nb = P.shape[0]
      d += nb
      nb_total += nb

      i +=  1

    # Fill _P with values from density matrices
    offset = 0
    for P in density:
      nb=P.shape[0]
      _P = np.zeros((d, d))
      _P[offset:offset + nb, offset:offset + nb] = P
      offset += nb
      _density.append(_P)

    JJ, KK = scf.rhf.get_jk(super_mol, dm=(_density))

    J=0
    for j in JJ:
      J +=  j

    G=[]
    for i in range(len(KK)):
      g=J - 0.5 * KK[i]
      G.append(g)

    return G

  def get_fock(G_list,H_core):
      F = []

      for G_val in G_list:
          f = H_core + G_val
          F.append(f)

      return F


  def get_Z(Z0=None):
    Z=[]
    ofst=0
    if Z0 is None:
      for i in range(len(basis)):
        _,nvt,noc=basis[i]
        z=np.zeros((nvt*noc)).reshape((nvt,noc,))
        Z.append(z)
        ofst+=(nvt*noc)
    else:
      for i in range(len(basis)):
        _,nvt,noc=basis[i]
        z=Z0[ofst:ofst+(nvt*noc)].reshape((nvt,noc,))
        Z.append(z)
        ofst+=(nvt*noc)
    return Z

  def get_energy(G,density_list):
    offset = 0
    Eer_dimer = 0

    for i, g in enumerate(G):
      P = density_list[i]
      nb = P.shape[0]
      g = G[i]

      Eer_dimer += np.trace(H_core[offset:offset + nb, offset:offset + nb] @ P) + 0.5 * np.trace(g[offset:offset + nb, offset:offset + nb] @ P)
      offset += nb

    E = Eer_dimer+enr
    return E

  def objective(Z0):
    nonlocal final_C_new
    Z=get_Z(Z0)
    count=0
    nb_list  = [item[0] for item in basis]
    nvt_list = [item[1] for item in basis]
    noc_list = [item[2] for item in basis]

    if  count==0:
      C, L_inv,U_inv = get_orbitals(Z,guess,noc_list,nvt_list,)
    else:
      C, L_inv,U_inv = get_orbitals(Z,orbital,noc_list,nvt_list,)

    #raise
    density = get_density(C,noc_list)
    G = get_G(density)
    F = get_fock(G,H_core,)
    offset = 0
    Eer_dimer = 0

    for i, g in enumerate(G):
      P = density[i]
      nb = P.shape[0]
      g = G[i]

      Eer_dimer += np.trace(H_core[offset:offset + nb, offset:offset + nb] @ P) + 0.5 * np.trace(g[offset:offset + nb, offset:offset + nb] @ P)
      offset += nb

    E = Eer_dimer+enr
    #print(E)
    G_global=[]
    ofst=0
    for c,L_inv,U_inv,F,noc, in zip(C,L_inv,U_inv,F,noc_list,):
      nb=c.shape[0]
      G_loc=c[:,noc:].T@F[ofst:(ofst+nb),ofst:(ofst+nb)]@c[:,:noc]
      G_gbl=U_inv.T@G_loc@L_inv

      G_global.append(G_gbl.flatten())
      ofst += nb

    Z0=block_diagonalize(Z)
    count+=0
    orbital=C
    final_C_new=C
    return E, np.hstack(G_global)

  def Heitler_london(guess_list,noc_list):
      p=orthogonal_density(guess_list,noc_list)
      J,K =scf.rhf.get_jk(super_mol,dm=(p))
      g=J-0.5*K
      HL_E=2*np.trace(H_core@p)+2*np.trace(np.dot(p,g))+enr
      return HL_E
    
  def orthogonal_density(guess_list,noc_list):
      orbital=[]
      for C,noc in zip(guess_list,noc_list):
          #nup=noc//2
          c=C[:,:noc]
          orbital.append(c)
      C_tilda=linalg.block_diag(*orbital)
      s=C_tilda.T@S@C_tilda
      P=np.dot(C_tilda,linalg.inv(s))@C_tilda.T 
      return P

  def physical_energy(final_C_new,noc_list):
    P=orthogonal_density(final_C_new,noc_list)
    J,K = scf.rhf.get_jk(super_mol, dm=(P))
    G=J-0.5*K
    hf= 2*np.trace(H_core@P)+2*np.trace(np.dot(P,G))+enr
    #print(hf)
    return hf


  final_C_new = None
  #final_C_new2 = None
  #final_p1,final_p2=None,None

  super_mol,guess, basis,n = initialize_molecules(molecules)

  nb_list  = [item[0] for item in basis]
  nvt_list = [item[1] for item in basis]
  vrt = sum(nvt_list)
  noc_list = [item[2] for item in basis]
  occ=sum(noc_list)

  # Compute Hamiltonians
  T = super_mol.intor_symmetric('int1e_kin')
  V = super_mol.intor_symmetric('int1e_nuc')
  S = super_mol.intor_symmetric('int1e_ovlp')
  H_core = T + V
  enr = super_mol.energy_nuc()

  Z=get_Z(None)

  #Compute orbital and density
  """
  if guess_C is None:
    C, L_inv,U_inv = get_orbitals(Z,guess,noc_list,nvt_list,)
  else:
    C, L_inv,U_inv = get_orbitals(Z,guess_C,noc_list,nvt_list,)
  """
  #C, L_inv,U_inv = get_orbitals(Z,guess,noc_list,nvt_list,)
  #print(guess)
  density = get_density(guess,noc_list)

  # Compute G and F matrix
  G = get_G(density)
  #print(G)

  e = get_energy(G,density)
  print(noc_list)
  HL_E = Heitler_london(guess,noc_list)
  
  if only_HL_energy:
      return HL_E 

  Z0 = np.zeros((vrt*occ)//n)

  print(f'1st unphysical energy for dimer is {e} hartree.')
  print(f'Heitler london energy for dimer is {HL_E} hartree.')
  result=minimize(objective,Z0,method='BFGS',jac=True,options={'gtol':1e-9})
  E=result.fun

  print (f'converged unphysical energy for dimer is {E} hartree.')
  hf=physical_energy(final_C_new,noc_list)
  print(f"Converged physical energy for dimer is {hf} hartree")
  A=scf.HF(super_mol)
  energy=A.kernel()
  print(f"SCF energy from pyscf:    {energy} Hartree")

  return E, HL_E,hf

mol1 = gto.M(atom=[['H', (-0.7, 0.0, 0.0)], ['H', (0.7, 0.0, 0.0)]], basis='cc-pvdz', unit='au')
mol2 = gto.M(atom=[['H', (-0.7, 4.0, 0.0)], ['H', (0.7, 4.0, 0.0)]], basis='cc-pvdz', unit='au')
mol3 = gto.M(atom=[['H', (-0.7, -4.0, 0.0)], ['H', (0.7, -4.0, 0.0)]], basis='cc-pvdz', unit='au')
#mol1 = gto.M(atom=[['H', (-1.9207380, 0.1695600, 0.3600000)], ['O', (-2.8087660, 0.0289470, 0.0614980)], ['H', (-2.7094260, -0.3741760, -0.7889810)]], basis='cc-pvdz', unit='au')

#mol2 = gto.M(atom=[['O', (2.5969180, -0.0127240, -0.0289530)], ['H', (3.1696380, -0.6598260, 0.3566380)], ['H', (3.1553090, 0.7346610, -0.1880170)]], basis='cc-pvdz', unit='au')

#molecules=[mol1,]

#molecules=mol()
#generalized_dimer(molecules)

