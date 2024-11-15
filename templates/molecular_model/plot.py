from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


import ase
from xtb.ase.calculator import XTB
import numpy as np

import os


def render_molecule_2d(xyz_path, out_path):
    """Renders a molecule from an xyz file using RDKit and saves it as a png image
    
    Args:
        xyz_path (str): Path to input xyz file
        out_path (str): Path to save output png file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
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
        
    except FileNotFoundError:
        print(f"Warning: XYZ file not found at {xyz_path}")
        # Create blank image as fallback
        from PIL import Image
        img = Image.new('RGB', (300, 300), color='white')
        img.save(out_path)
        
    except Exception as e:
        print(f"Warning: Failed to render molecule {xyz_path}: {str(e)}")
        # Create blank image as fallback
        from PIL import Image
        img = Image.new('RGB', (300, 300), color='white')
        img.save(out_path)



out_dir = "."

render_molecule_2d(os.path.join(out_dir, "gen_0_react.xyz"), os.path.join(out_dir, "gen_0_react_2d.png"))
render_molecule_2d(os.path.join(out_dir, "gen_0_ts.xyz"), os.path.join(out_dir, "gen_0_ts_2d.png"))
render_molecule_2d(os.path.join(out_dir, "gen_0_prod.xyz"), os.path.join(out_dir, "gen_0_prod_2d.png"))



import ase
import ase.visualize
import numpy as np
import os

def render_molecule_3d(xyz_path, out_path):
    """Renders a molecule from an xyz file using ASE and saves it as a png image
    
    Args:
        xyz_path (str): Path to input xyz file
        out_path (str): Path to save output png file
    """
    # Create empty figure first
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Read XYZ file using ASE
        atoms = ase.io.read(xyz_path)
        
        # Create the visualization
        from ase.visualize.plot import plot_atoms
        
        # Create figure and render atoms
        fig, ax = plt.subplots()
        ax.set_axis_off()  # Hide axes
        
        # Plot atoms with element symbols
        plot_atoms(atoms, ax, rotation='45z,45x,45y', show_unit_cell=False, scale=0.5, radii=0.5)
               
        # Save figure
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Failed to render molecule {xyz_path}: {str(e)}")


render_molecule_3d(os.path.join(out_dir, "gen_0_react.xyz"), os.path.join(out_dir, "gen_0_react_3d.png"))
render_molecule_3d(os.path.join(out_dir, "gen_0_ts.xyz"), os.path.join(out_dir, "gen_0_ts_3d.png"))
render_molecule_3d(os.path.join(out_dir, "gen_0_prod.xyz"), os.path.join(out_dir, "gen_0_prod_3d.png"))
