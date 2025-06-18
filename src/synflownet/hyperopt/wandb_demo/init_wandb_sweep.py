import os
import sys
import time

import torch
import wandb

from synflownet.config import Config, init_empty
from synflownet.tasks.reactions_task import ReactionTrainer

TIME = time.strftime("%m-%d-%H-%M")
ENTITY = "asindhanai-montai-therapeutics"  # Your full wandb entity name
PROJECT = "synflownet"
SWEEP_NAME = f"{TIME}-synflownet-open-reproduce-qed"
# Use a local directory that actually exists
STORAGE_DIR = f"./wandb_sweeps/{SWEEP_NAME}"


# Define the search space of the sweep
sweep_config = {
    "name": SWEEP_NAME,
    "program": "init_wandb_sweep.py",
    "controller": {
        "type": "cloud",
    },
    "method": "grid",
    "parameters": {
        "config.seed": {"values": [0, 1, 2]},
    },
}


def wandb_config_merger():
    config = init_empty(Config())
    wandb_config = wandb.config

    # Set desired config values
    config.log_dir = f"{STORAGE_DIR}/{wandb.run.name}-id-{wandb.run.id}"
    config.print_every = 100
    config.validate_every = 500
    config.num_final_gen_steps = 100
    config.num_training_steps = 5000
    config.pickle_mp_messages = False
    config.overwrite_existing_exp = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.num_workers = 8
    config.algo.sampling_tau = 0.99
    config.algo.train_random_action_prob = 0.02
    config.algo.tb.Z_learning_rate = 1e-3
    config.algo.tb.Z_lr_decay = 2_000
    config.algo.tb.do_parameterize_p_b = True
    config.algo.tb.do_sample_p_b = False
    config.algo.max_len = 4
    config.algo.tb.backward_policy = "MaxLikelihood"
    config.algo.strict_bck_masking = False

    config.model.graph_transformer.fingerprint_type = "morgan_1024"
    config.model.graph_transformer.continuous_action_embs = True

    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [32.0]
    config.cond.weighted_prefs.preference_type = "dirichlet"
    config.cond.focus_region.focus_type = None
    config.replay.use = False

    config.reward = "qed"

    # Merge the wandb sweep config with the nested config from synflownet
    config.seed = wandb_config["config.seed"]
    
    # Ensure the log directory exists
    os.makedirs(config.log_dir, exist_ok=True)
    
    print(f"Config log_dir: {config.log_dir}")
    print(f"Wandb run name: {wandb.run.name}")
    print(f"Wandb run id: {wandb.run.id}")

    return config


if __name__ == "__main__":
    # Check if wandb is logged in
    try:
        api = wandb.Api()
        if ENTITY is None:
            # Get the current user's entity from wandb
            ENTITY = api.default_entity
        print(f"Using Wandb entity: {ENTITY}")
    except Exception as e:
        print(f"Error connecting to Wandb: {e}")
        print("Please run 'wandb login' first to authenticate with Wandb")
        sys.exit(1)
    
    # if there no arguments, initialize the sweep, otherwise this is a wandb agent
    if len(sys.argv) == 1:
        # Create storage directory if it doesn't exist
        os.makedirs(STORAGE_DIR, exist_ok=True)
        
        if os.path.exists(os.path.join(STORAGE_DIR, "sweep_id.txt")):
            print(f"Sweep already exists in {STORAGE_DIR}")
            with open(os.path.join(STORAGE_DIR, "sweep_id.txt"), "r") as f:
                sweep_id = f.read().strip()
            print(f"Existing sweep ID: {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
            print(f"Created new sweep with ID: {sweep_id}")
            # Save sweep ID for future reference
            with open(os.path.join(STORAGE_DIR, "sweep_id.txt"), "w") as f:
                f.write(sweep_id)

    else:
        wandb.init(entity=ENTITY, project=PROJECT)
        config = wandb_config_merger()
        trial = ReactionTrainer(config)
        trial.run()
