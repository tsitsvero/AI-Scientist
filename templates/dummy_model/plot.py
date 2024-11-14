from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


import ase
from xtb.ase.calculator import XTB
import numpy as np

import os


def render_molecule_rdkit(xyz_path, out_path):
    """Renders a molecule from an xyz file using RDKit and saves it as a png image
    
    Args:
        xyz_path (str): Path to input xyz file
        out_path (str): Path to save output png file
    """
    # Read XYZ file
    with open(xyz_path, "r") as f:
        xyz_data = f.readlines()

    # Parse number of atoms
    num_atoms = int(xyz_data[0])

    # Create RDKit molecule
    mol = Chem.rdchem.RWMol()
    conf = Chem.rdchem.Conformer(num_atoms)

    # Add atoms and coordinates
    for i in range(2, num_atoms+2):
        line = xyz_data[i].split()
        atom_symbol = line[0]
        x = float(line[1])
        y = float(line[2])
        z = float(line[3])
        
        # Add atom
        atom = Chem.rdchem.Atom(atom_symbol)
        atom_idx = mol.AddAtom(atom)
        
        # Set 3D coordinates
        conf.SetAtomPosition(atom_idx, (x, y, z))

    # Add the conformer after all atoms are added
    mol.AddConformer(conf)

    # Connect atoms based on distance
    for i in range(mol.GetNumAtoms()):
        for j in range(i+1, mol.GetNumAtoms()):
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = pos_i.Distance(pos_j)
            
            # Add bonds if atoms are close enough
            if dist < 1.7:  # Typical bond length threshold in Angstroms
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

    # Convert to regular molecule and render
    mol = mol.GetMol()
    img = Draw.MolToImage(mol)
    img.save(out_path)



out_dir = "."

render_molecule_rdkit(os.path.join(out_dir, "gen_0_react.xyz"), os.path.join(out_dir, "gen_0_react.png"))
render_molecule_rdkit(os.path.join(out_dir, "gen_0_ts.xyz"), os.path.join(out_dir, "gen_0_ts.png"))
render_molecule_rdkit(os.path.join(out_dir, "gen_0_prod.xyz"), os.path.join(out_dir, "gen_0_prod.png"))