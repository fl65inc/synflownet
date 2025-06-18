#!/usr/bin/env python3
"""
Test script to verify that the ReactionTrainer can initialize and log to Wandb
"""
import os
import sys
sys.path.append('/home/ubuntu/synflownet/src')

import wandb
from synflownet.config import Config, init_empty
from synflownet.tasks.reactions_task import ReactionTrainer

def test_reaction_trainer_wandb():
    """Test that ReactionTrainer can initialize and log to Wandb"""
    try:
        print("Testing ReactionTrainer with Wandb integration...")
        
        # Initialize wandb run
        run = wandb.init(
            project="synflownet-test",
            entity="asindhanai-montai-therapeutics",
            name="test-reaction-trainer",
            config={
                "test_run": True,
                "num_training_steps": 10,  # Very short test
            }
        )
        
        print(f"✓ Wandb run initialized: {run.name}")
        
        # Create minimal config
        config = init_empty(Config())
        config.log_dir = f"./test_logs/{wandb.run.name}-id-{wandb.run.id}"
        config.print_every = 1
        config.validate_every = 5
        config.num_final_gen_steps = 5
        config.num_training_steps = 10
        config.device = "cpu"  # Use CPU for testing
        config.num_workers = 0  # No multiprocessing for test
        config.overwrite_existing_exp = True
        config.reward = "qed"
        
        # Basic algorithm settings
        config.algo.sampling_tau = 0.99
        config.algo.max_len = 3
        config.algo.tb.backward_policy = "MaxLikelihood"
        config.algo.tb.do_parameterize_p_b = True
        config.algo.tb.do_sample_p_b = False
        config.replay.use = False
        
        # Model settings
        config.model.graph_transformer.fingerprint_type = "morgan_1024"
        config.model.graph_transformer.continuous_action_embs = True
        
        # Conditioning settings
        config.cond.temperature.sample_dist = "constant"
        config.cond.temperature.dist_params = [32.0]
        
        print("✓ Config created successfully")
        
        # Try to create trainer (this will test if all dependencies are available)
        try:
            trainer = ReactionTrainer(config)
            print("✓ ReactionTrainer created successfully")
            
            # Test that wandb logging works
            wandb.log({"test_metric": 1.0})
            print("✓ Wandb logging works")
            
            print("✓ All tests passed!")
            print(f"Check your wandb dashboard: https://wandb.ai/asindhanai-montai-therapeutics/synflownet-test")
            
            return True
            
        except Exception as e:
            print(f"✗ Error creating ReactionTrainer: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error in test: {e}")
        return False
    finally:
        if wandb.run:
            wandb.finish()

if __name__ == "__main__":
    success = test_reaction_trainer_wandb()
    if success:
        print("\n🎉 Your setup is ready! The ReactionTrainer should now log to Wandb.")
    else:
        print("\n💡 There are some issues with the setup. Check the errors above.")
