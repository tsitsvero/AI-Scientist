#!/bin/bash

# Then activate environment and install required packages
mamba activate ai_scientist


# Rest of your script
python experiment.py --out_dir run_0; python plot.py

cd ../..
python launch_scientist.py --model "gpt-4o-mini-2024-07-18" --experiment molecular_model --num-ideas 1 --skip-novelty-check > OUTPUT.log 2>&1
