import argparse
import inspect
import json
import math
import os
import pickle
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
args = parser.parse_args()


import random

# --- Importing necessary function ---
import torch
from torch.utils.data import DataLoader

from oa_reactdiff.trainer.pl_trainer import DDPMModule


from oa_reactdiff.dataset import ProcessedTS1x
from oa_reactdiff.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule

from oa_reactdiff.diffusion._normalizer import FEATURE_MAPPING
# from oa_reactdiff.analyze.rmsd import batch_rmsd

# from oa_reactdiff.utils.sampling_tools import (
#     assemble_sample_inputs,
#     write_tmp_xyz,
# )

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


import ase
from xtb.ase.calculator import XTB
import numpy as np

def get_molecule_and_chemical_formula(idx=0):
    print("Loading dataset...")

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    dataset = ProcessedTS1x(
        npz_path="./oa_reactdiff/data/transition1x/train.pkl",
        center=True,
        pad_fragments=0,
        device=device,
        zero_charge=False,
        remove_h=False,
        single_frag_only=False,
        swapping_react_prod=False,
        use_by_ind=True,
    )

    print("Creating data loader...")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    data_list = list(loader)  # Only use first 10 entries

    print("Getting molecule representations...")
    representations, res= data_list[idx]

    # Get positions from representations
    pos = representations[0]["pos"].detach().cpu().numpy()
    
    # Get atomic numbers from charge tensor in representations
    atomic_nums = representations[1]["charge"].detach().cpu().numpy().squeeze()
    
    print("Creating ASE Atoms object...")
    # Create ASE Atoms object
    atoms = ase.Atoms(numbers=atomic_nums, positions=pos)
    atoms.write("chemical_formula.xyz", format="xyz")
    
    print("Done getting molecule and chemical formula")    
    return representations, res, atoms.get_chemical_formula(), atoms


def generate_ts_and_products(representations, res, checkpoint_path="/home/mm/Downloads/pretrained-ts1x-diff.ckpt", device=None, out_dir="."):
    """Generates transition state and product structures using pre-trained DDPM model
    
    Args:
        representations: Molecule representations from dataset
        res: Results from dataset
        checkpoint_path (str): Path to model checkpoint
        device (torch.device): Device to run model on
        
    Returns:
        tuple: Generated molecule samples and masks
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    # Load model
    ddpm_trainer = DDPMModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device
    )
    ddpm_trainer = ddpm_trainer.to(device)

    print("Setting up noise schedule...")
    # Set up noise schedule
    noise_schedule = "polynomial_2"
    timesteps = 1
    precision = 1e-5

    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=precision,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module,
        norm_values=ddpm_trainer.ddpm.norm_values
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    ddpm_trainer = ddpm_trainer.to(device)

    print("Preparing inputs for generation...")
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [
        repre["size"] for repre in representations
    ]
    conditions = torch.tensor([[0] for _ in range(n_samples)], device=device)


    new_order_react = torch.randperm(representations[0]["size"].item())
    for k in ["pos", "one_hot", "charge"]:
        representations[0][k] = representations[0][k][new_order_react]
        
    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations
    ]

    print("Generating molecule...")
    # Generate molecule
    out_samples, out_masks = ddpm_trainer.ddpm.inpaint(
        n_samples=1,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=0,
        jump_length=5,
        timesteps=1,
        xh_fixed=xh_fixed,
        frag_fixed=[0, 2],  # Fix reactant and product, generate TS
    )

    # Convert out_samples tensor to ASE Atoms object
    pos = out_samples[0][2].detach().cpu().numpy()[:,:3] # Get positions for TS structure, take only first 3 coords
    # Get atomic numbers from representations
    atomic_nums = xh_fixed[1][:, -1].detach().cpu().numpy().astype(int)

    atoms = ase.Atoms(numbers=atomic_nums, positions=pos)

    # Save each structure (reactant, TS, product) as xyz files
    for idx, (sample, fixed) in enumerate([
        (out_samples[0][0], xh_fixed[0]),  # Reactant
        (out_samples[0][2], xh_fixed[1]),  # TS 
        (out_samples[0][1], xh_fixed[2])   # Product
    ]):
        # Get positions and atomic numbers
        pos = sample.detach().cpu().numpy()[:,:3]
        atomic_nums = fixed[:, -1].detach().cpu().numpy().astype(int)
        
        # Create ASE Atoms object
        atoms = ase.Atoms(numbers=atomic_nums, positions=pos)
        
        # Write to xyz file
        structure_type = ["react", "ts", "prod"][idx]
        xyz_path = os.path.join(out_dir, f"gen_0_{structure_type}.xyz")
        atoms.write(xyz_path, format="xyz")
    

    print("Generation complete")
    return out_samples, out_masks, xh_fixed


def perform_experiment(dataset, out_dir, seed_offset):
    """Dummy training function that returns random energy values"""
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    ## SET INDEX OF MOLECULE TO USE HERE:
    ###################################
    idx = 0
    ###################################

    representations, res, chemical_formula, atoms = get_molecule_and_chemical_formula(idx=idx)
    print("Chemical formula:", chemical_formula)
    
    # Write xyz file directly to output directory
    xyz_path = os.path.join(out_dir, "chemical_formula.xyz")
    atoms.write(xyz_path, format="xyz")
    
    
    out_samples, out_masks, xh_fixed = generate_ts_and_products(
        representations, 
        res,
        out_dir=out_dir
    )
    
    # Generate random energies in a reasonable range (in eV)
    reactant_energy = random.uniform(-10.0, -5.0)
    ts_energy = random.uniform(-5.0, 0.0) 
    product_energy = random.uniform(-8.0, -3.0)

    # Create dictionary with energies
    energy_info = {
        "activation_energy": {"means": ts_energy - reactant_energy},
        "reaction_energy": {"means": product_energy - reactant_energy},
        "energies": {
            "reactant": reactant_energy,
            "ts": ts_energy,
            "product": product_energy
        }
    }

    # Save energy info
    with open(os.path.join(out_dir, f"energy_info_{dataset}_{seed_offset}.json"), "w") as f:
        json.dump(energy_info, f)

    return energy_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="run_0")
    args = parser.parse_args()
    
    out_dir = args.out_dir
    all_results = {}
    final_infos = {}
    
    # Just use shakespeare_char dataset with one seed
    dataset = "molecules"
    
    # Generate once with seed_offset 0
    energy_info = perform_experiment(dataset, out_dir, 0)
    all_results[f"{dataset}_0_energy_info"] = energy_info
    
    # Store means similar to 2d_diffusion format
    final_infos[dataset] = {
        "means": {
            "activation_energy": energy_info["activation_energy"]["means"]
        }
    }

    print(final_infos)
    print(all_results)

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)
