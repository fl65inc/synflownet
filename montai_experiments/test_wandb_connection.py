#!/usr/bin/env python3
"""
Simple script to test Wandb connectivity and configuration
"""

import wandb
import torch

def test_wandb():
    try:
        # Test basic wandb connection
        api = wandb.Api()
        print(f"✅ Wandb API connection successful")
        print(f"Default entity: {api.default_entity}")
        
        # Test wandb initialization
        run = wandb.init(
            project="synflownet-test", 
            entity="asindhanai",  # Your wandb username
            config={
                "test_param": 1.0,
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            },
            notes="Testing wandb integration for SynFlowNet"
        )
        
        print(f"✅ Wandb run initialized successfully")
        print(f"Run name: {run.name}")
        print(f"Run ID: {run.id}")
        print(f"Run URL: {run.url}")
        
        # Test logging
        for i in range(5):
            wandb.log({
                "test_metric": i * 0.1,
                "iteration": i
            })
        
        print(f"✅ Logged test metrics successfully")
        
        # Finish the run
        wandb.finish()
        print(f"✅ Wandb run completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Wandb connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Wandb connection...")
    success = test_wandb()
    if success:
        print("\n🎉 Wandb is properly configured! Your training should now appear in your Wandb account.")
    else:
        print("\n💡 Please run 'wandb login' and enter your API key from https://wandb.ai/authorize")
