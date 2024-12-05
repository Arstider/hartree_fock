import numpy as np
from pyscf import gto,scf
from scipy import linalg
from scipy.optimize import minimize
#from general import *
np.set_printoptions(suppress=True,precision=6,linewidth=1000000000000000000)


#Defining molecules for test purpose
mol1 = gto.M(atom=[['H', (-0.7, 0.0, 0.0)], ['H', (0.7, 0.0, 0.0)]], basis='cc-pvdz', unit='au')
mol2 = gto.M(atom=[['H', (-0.7, 4.0, 0.0)], ['H', (0.7, 4.0, 0.0)]], basis='cc-pvdz', unit='au')
mol3 = gto.M(atom=[['H', (-0.7, -4.0, 0.0)], ['H', (0.7, -4.0, 0.0)]], basis='cc-pvdz', unit='au')

mol=[mol1,mol2,mol3]

#creating a super molecule
sup_mol  =    gto.M()
sup_mol +=   mol1+mol2+mol3
sup_noc = sup_mol.nelectron//2  

mf  =   scf.RHF(sup_mol)
mf.kernel()
dm=mf.make_rdm1()
sup_orb = mf.mo_coeff[:,:sup_noc]


  
def ghost(mol_list):
    for i,mol in enumerate(mol_list):
        mod_super= gto.M()
        mod_super+=mol_list[i]
        for mol in mol_list:
            if mol_list.index(mol)!=i:
                mod_super += gto.M(atom=[(f'X:{atom[0]}',atom[1]) for atom in mol.atom],basis=mol.basis)
        
        _noc = mod_super.nelectron//2
        i2s=mod_super.intor('int2e',aosym='s1')
        H_core=mod_super.intor_symmetric('int1e_kin')+mod_super.intor_symmetric('int1e_nuc')
        #P = 2*sup_orb[:,:_noc]@sup_orb[:,:_noc].T
        P = 2*sup_orb@sup_orb.T
        G=np.zeros_like(P)
        G+=np.einsum('ls,mnsl->mn',P,i2s)
        G-=np.einsum('ns,mnsl->ml',P,i2s)*0.5
        F=H_core+G

        E=0.5*np.trace(P@(H_core+F))+mod_super.energy_nuc()
        print(E)
        
        raise
        
        #mf=scf.RHF(mod_super)
        #energy=mf.kernel(dm0=dm)
        #orbital=mf.mo_coeff
        
        x= linalg.solve(orbital,sup_orb)
        print(x[:,:15])
        print(x[:,15:])
        
        #raise
        """
        def objective(x):
            return np.sum((sup_orb-orbital@x)**2)
        
        x0=np.zeros(orbital.shape[1])
        result=minimize(objective,x0,method='BFGS')

        x_opt=result.x
        print(x_opt)
        """
    return 

ghost(mol)
