#!/usr/bin/env python3
"""
Quick-start script for sampling molecules from SynFlowNet models.

This is a simplified version for users who just want to quickly sample
a few molecules without complex configuration.

Usage:
    python quick_sample.py path/to/checkpoint.pt
    python quick_sample.py path/to/checkpoint.pt --count 50 --temp 0.8
"""

import sys
import argparse
from pathlib import Path

# Add synflownet to path
sys.path.append(str(Path(__file__).parent / "src"))

from synflownet.tasks.reactions_task import ReactionTrainer
from synflownet.utils.synthesis_utils import clean_smiles
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def quick_sample(checkpoint_path: str, num_molecules: int = 10, temperature: float = 1.0):
    """
    Quickly sample molecules from a trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_molecules: Number of molecules to generate
        temperature: Sampling temperature (0.1=deterministic, 2.0=very random)
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Load model
    trainer = ReactionTrainer.load_from_checkpoint(checkpoint_path)
    model = trainer.model
    task = trainer.task
    ctx = trainer.ctx
    sampler = trainer.sampler
    
    model.eval()
    print(f"Model loaded! Generating {num_molecules} molecules...")
    
    # Set temperature
    original_temp = getattr(sampler, 'sample_temp', 1.0)
    sampler.sample_temp = temperature
    
    try:
        with torch.no_grad():
            # Sample conditioning info
            cond_info = task.sample_conditional_information(num_molecules, step=0)
            
            # Sample molecules
            trajectories = sampler.sample_from_model(
                model=model,
                n=num_molecules, 
                cond_info=cond_info["encoding"],
                random_action_prob=0.0
            )
            
    finally:
        # Restore original temperature
        sampler.sample_temp = original_temp
    
    # Process results
    molecules = []
    for i, traj in enumerate(trajectories):
        try:
            # Get final molecule
            final_graph = traj['result']
            mol = ctx.graph_to_obj(final_graph)
            
            if mol is None:
                continue
                
            smiles = Chem.MolToSmiles(mol)
            clean_smi = clean_smiles(smiles)
            
            if clean_smi is None:
                continue
            
            # Calculate basic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            qed = QED.qed(mol)
            
            # Calculate reward
            try:
                graph = ctx.obj_to_graph(mol)
                reward = task.compute_obj_properties([graph])
                reward_value = reward[0].item() if len(reward) > 0 else None
            except:
                reward_value = None
            
            molecules.append({
                'smiles': clean_smi,
                'molecular_weight': mw,
                'logp': logp,
                'qed': qed,
                'reward': reward_value,
                'valid': traj.get('is_valid', True)
            })
            
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    
    return molecules


def main():
    parser = argparse.ArgumentParser(description="Quick molecule sampling from SynFlowNet")
    parser.add_argument("checkpoint_path", help="Path to model checkpoint (.pt file)")
    parser.add_argument("--count", "-n", type=int, default=10, 
                       help="Number of molecules to generate (default: 10)")
    parser.add_argument("--temperature", "--temp", "-t", type=float, default=1.0,
                       help="Sampling temperature: 0.1=deterministic, 2.0=random (default: 1.0)")
    parser.add_argument("--output", "-o", help="Output file (optional)")
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    # Sample molecules
    try:
        molecules = quick_sample(args.checkpoint_path, args.count, args.temperature)
    except Exception as e:
        print(f"Error during sampling: {e}")
        sys.exit(1)
    
    if not molecules:
        print("No valid molecules were generated!")
        sys.exit(1)
    
    # Display results
    print(f"\n✓ Generated {len(molecules)} valid molecules:")
    print("=" * 80)
    
    for i, mol in enumerate(molecules, 1):
        reward_str = f"{mol['reward']:.3f}" if mol['reward'] is not None else "N/A"
        print(f"{i:2d}. {mol['smiles']}")
        print(f"    MW: {mol['molecular_weight']:.1f}  LogP: {mol['logp']:.2f}  "
              f"QED: {mol['qed']:.3f}  Reward: {reward_str}")
        print()
    
    # Summary statistics
    valid_count = sum(1 for mol in molecules if mol['valid'])
    avg_qed = sum(mol['qed'] for mol in molecules) / len(molecules)
    rewards = [mol['reward'] for mol in molecules if mol['reward'] is not None]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    
    print("=" * 80)
    print(f"Summary: {valid_count}/{len(molecules)} valid molecules")
    print(f"Average QED: {avg_qed:.3f}")
    if avg_reward is not None:
        print(f"Average Reward: {avg_reward:.3f}")
    
    # Save to file if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['smiles', 'molecular_weight', 'logp', 'qed', 'reward', 'valid'])
            writer.writeheader()
            writer.writerows(molecules)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
