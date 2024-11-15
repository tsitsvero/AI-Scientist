

# gpt-4o-mini
# gpt-4o-mini-2024-07-18


pip install git+https://github.com/tsitsvero/OAReactDiff.git


# python experiment.py --out-dir "./dummy_model_runs"
python experiment.py --out_dir run_0; python plot.py
python launch_scientist.py --model "gpt-4o-mini-2024-07-18" --experiment molecular_model --num-ideas 1 --skip-novelty-check > OUTPUT.log
