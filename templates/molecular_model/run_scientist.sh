#!/bin/bash

# Then activate environment and install required packages
mamba activate ai_scientist


# Rest of your script
python experiment.py --out_dir run_0; python plot.py

cd ../..
python launch_scientist.py --model "gpt-4o-mini-2024-07-18" --experiment molecular_model --num-ideas 1 --skip-novelty-check > OUTPUT.log 2>&1


# "Claude 3.5 Sonnet 2024-10-22"
# python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment molecular_model --num-ideas 1 --skip-novelty-check > OUTPUT.log 2>&1

# "claude-3-5-sonnet-20241022"
# gpt-4o-2024-08-06

