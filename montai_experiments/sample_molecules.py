#!/usr/bin/env python3
"""
SynFlowNet Molecule Sampling Script

This script demonstrates how to use a trained SynFlowNet model to sample new molecules.
It provides various sampling strategies and configuration options.

Usage:
    python sample_molecules.py --checkpoint_path /path/to/model_checkpoint.pt
    
Example:
    python sample_molecules.py --checkpoint_path ./logs/my_training/model_state.pt \
                               --num_samples 100 \
                               --reward_threshold 0.7 \
                               --output_file sampled_molecules.csv
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Add the synflownet source to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from synflownet.tasks.reactions_task import ReactionTrainer
from synflownet.utils.synthesis_utils import clean_smiles
from synflownet.utils.synthesis_evals import calculate_molecular_diversity, to_fingerprint


class MoleculeSampler:
    """
    A class for sampling molecules from trained SynFlowNet models
    """
    
    def __init__(self, checkpoint_path: str):
        """
        Initialize the sampler with a trained model checkpoint
        
        Args:
            checkpoint_path: Path to the saved model checkpoint (.pt file)
        """
        self.checkpoint_path = checkpoint_path
        self.trainer = None
        self.model = None
        self.task = None
        self.ctx = None
        self.sampler = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and associated components"""
        print(f"Loading model from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Load the trainer from checkpoint
        self.trainer = ReactionTrainer.load_from_checkpoint(self.checkpoint_path)
        
        # Extract components
        self.model = self.trainer.model
        self.task = self.trainer.task
        self.ctx = self.trainer.ctx
        self.sampler = self.trainer.sampler
        
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")
        
        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {total_params:,} parameters")
    
    def sample_molecules(self, 
                        num_samples: int = 100,
                        conditional_info: Optional[Dict] = None,
                        random_action_prob: float = 0.0,
                        temperature: float = 1.0,
                        use_argmax: bool = False) -> List[Dict[str, Any]]:
        """
        Sample molecules from the trained model
        
        Args:
            num_samples: Number of molecules to sample
            conditional_info: Optional conditioning information (temperature, focus, etc.)
            random_action_prob: Probability of taking random actions (0.0 = greedy, 1.0 = random)
            temperature: Sampling temperature (lower = more deterministic)
            use_argmax: If True, use deterministic argmax sampling
            
        Returns:
            List of dictionaries containing sampled molecule information
        """
        print(f"Sampling {num_samples} molecules...")
        
        # Prepare conditional information
        if conditional_info is None:
            # Use default conditioning from the task
            cond_info = self.task.sample_conditional_information(num_samples, step=0)
            cond_encoding = cond_info["encoding"]
        else:
            # Custom conditioning (e.g., specific temperature, focus direction)
            cond_encoding = self._prepare_custom_conditioning(num_samples, conditional_info)
        
        # Set sampling temperature
        original_temp = getattr(self.sampler, 'sample_temp', 1.0)
        self.sampler.sample_temp = temperature
        
        try:
            with torch.no_grad():
                # Sample trajectories from the model
                trajectories = self.sampler.sample_from_model(
                    model=self.model,
                    n=num_samples,
                    cond_info=cond_encoding,
                    random_action_prob=random_action_prob,
                    use_argmax=use_argmax
                )
        finally:
            # Restore original temperature
            self.sampler.sample_temp = original_temp
        
        # Process trajectories into molecules
        molecules = []
        for i, traj in enumerate(trajectories):
            mol_data = self._process_trajectory(traj, i)
            if mol_data is not None:
                molecules.append(mol_data)
        
        print(f"Successfully generated {len(molecules)}/{num_samples} valid molecules")
        return molecules
    
    def _prepare_custom_conditioning(self, num_samples: int, conditional_info: Dict) -> torch.Tensor:
        """
        Prepare custom conditional information tensor
        
        Args:
            num_samples: Number of samples
            conditional_info: Dictionary with conditioning parameters
            
        Returns:
            Conditioning tensor
        """
        # This is a simplified version - you might need to adapt based on your specific task
        device = next(self.model.parameters()).device
        
        if 'temperature' in conditional_info:
            # Temperature conditioning
            temp = conditional_info['temperature']
            cond_tensor = torch.full((num_samples, 1), temp, device=device)
        else:
            # Default conditioning
            cond_tensor = torch.zeros((num_samples, self.task.num_cond_dim), device=device)
        
        return cond_tensor
    
    def _process_trajectory(self, trajectory: Dict, idx: int) -> Optional[Dict[str, Any]]:
        """
        Process a single trajectory into molecule data
        
        Args:
            trajectory: Trajectory dictionary from sampling
            idx: Trajectory index
            
        Returns:
            Dictionary with molecule data or None if invalid
        """
        try:
            # Get the final graph state
            final_graph = trajectory['result']
            
            # Convert graph to molecule
            mol = self.ctx.graph_to_obj(final_graph)
            
            if mol is None:
                return None
            
            # Get SMILES
            smiles = Chem.MolToSmiles(mol)
            clean_smi = clean_smiles(smiles)
            
            if clean_smi is None:
                return None
            
            # Calculate molecular properties
            properties = self._calculate_properties(mol)
            
            # Calculate reward if possible
            reward = self._calculate_reward(mol)
            
            mol_data = {
                'idx': idx,
                'smiles': clean_smi,
                'is_valid': trajectory.get('is_valid', True),
                'trajectory_length': len(trajectory.get('traj', [])),
                'reward': reward,
                'forward_logprob': trajectory.get('fwd_logprob', torch.tensor(0.0)).item(),
                'backward_logprob': trajectory.get('bck_logprob', torch.tensor(0.0)).item(),
                **properties
            }
            
            return mol_data
            
        except Exception as e:
            print(f"Error processing trajectory {idx}: {e}")
            return None
    
    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate molecular properties"""
        try:
            return {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'qed': QED.qed(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_rings': Descriptors.RingCount(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
            }
        except Exception as e:
            print(f"Error calculating properties: {e}")
            return {}
    
    def _calculate_reward(self, mol: Chem.Mol) -> Optional[float]:
        """Calculate reward using the task's reward function"""
        try:
            # Convert to graph for reward calculation
            graph = self.ctx.obj_to_graph(mol)
            
            # Calculate reward
            reward = self.task.compute_obj_properties([graph])
            if len(reward) > 0:
                return reward[0].item()
            return None
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return None
    
    def sample_with_filtering(self,
                            num_samples: int = 100,
                            reward_threshold: Optional[float] = None,
                            max_attempts: int = 1000,
                            **sampling_kwargs) -> List[Dict[str, Any]]:
        """
        Sample molecules with filtering criteria
        
        Args:
            num_samples: Target number of molecules
            reward_threshold: Minimum reward threshold
            max_attempts: Maximum sampling attempts
            **sampling_kwargs: Additional sampling parameters
            
        Returns:
            List of filtered molecules
        """
        print(f"Sampling with filtering (target: {num_samples}, max attempts: {max_attempts})")
        
        filtered_molecules = []
        total_attempts = 0
        
        while len(filtered_molecules) < num_samples and total_attempts < max_attempts:
            # Sample in batches
            batch_size = min(50, num_samples - len(filtered_molecules))
            molecules = self.sample_molecules(batch_size, **sampling_kwargs)
            
            for mol in molecules:
                total_attempts += 1
                
                # Apply filters
                if self._passes_filters(mol, reward_threshold):
                    filtered_molecules.append(mol)
                
                if len(filtered_molecules) >= num_samples:
                    break
        
        print(f"Found {len(filtered_molecules)} molecules passing filters after {total_attempts} attempts")
        return filtered_molecules[:num_samples]
    
    def _passes_filters(self, mol_data: Dict, reward_threshold: Optional[float]) -> bool:
        """Check if molecule passes filtering criteria"""
        # Validity check
        if not mol_data.get('is_valid', True):
            return False
        
        # Reward threshold check
        if reward_threshold is not None:
            reward = mol_data.get('reward')
            if reward is None or reward < reward_threshold:
                return False
        
        # Drug-likeness filters (Lipinski's Rule of Five)
        mw = mol_data.get('molecular_weight', 0)
        logp = mol_data.get('logp', 0)
        hbd = mol_data.get('num_hbd', 0)
        hba = mol_data.get('num_hba', 0)
        
        if mw > 500 or logp > 5 or hbd > 5 or hba > 10:
            return False
        
        return True
    
    def analyze_diversity(self, molecules: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze molecular diversity of sampled set"""
        smiles_list = [mol['smiles'] for mol in molecules]
        
        # Remove duplicates
        unique_smiles = list(set(smiles_list))
        diversity_ratio = len(unique_smiles) / len(smiles_list) if smiles_list else 0
        
        # Calculate molecular diversity using fingerprints
        try:
            mols = [Chem.MolFromSmiles(smi) for smi in unique_smiles]
            mols = [mol for mol in mols if mol is not None]
            
            if len(mols) > 1:
                diversity_score = calculate_molecular_diversity(mols)
            else:
                diversity_score = 0.0
        except Exception as e:
            print(f"Error calculating diversity: {e}")
            diversity_score = 0.0
        
        return {
            'total_molecules': len(molecules),
            'unique_molecules': len(unique_smiles),
            'diversity_ratio': diversity_ratio,
            'diversity_score': diversity_score
        }


def save_molecules(molecules: List[Dict[str, Any]], output_file: str):
    """Save molecules to CSV file"""
    if not molecules:
        print("No molecules to save")
        return
    
    df = pd.DataFrame(molecules)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(molecules)} molecules to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Sample molecules from trained SynFlowNet model")
    parser.add_argument("--checkpoint_path", required=True, 
                       help="Path to model checkpoint file")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of molecules to sample")
    parser.add_argument("--reward_threshold", type=float, default=None,
                       help="Minimum reward threshold for filtering")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--random_action_prob", type=float, default=0.0,
                       help="Probability of random actions")
    parser.add_argument("--use_argmax", action="store_true",
                       help="Use deterministic argmax sampling")
    parser.add_argument("--output_file", default=None,
                       help="Output CSV file path")
    parser.add_argument("--max_attempts", type=int, default=1000,
                       help="Maximum sampling attempts when filtering")
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = MoleculeSampler(args.checkpoint_path)
    
    # Sample molecules
    if args.reward_threshold is not None:
        molecules = sampler.sample_with_filtering(
            num_samples=args.num_samples,
            reward_threshold=args.reward_threshold,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            random_action_prob=args.random_action_prob,
            use_argmax=args.use_argmax
        )
    else:
        molecules = sampler.sample_molecules(
            num_samples=args.num_samples,
            temperature=args.temperature,
            random_action_prob=args.random_action_prob,
            use_argmax=args.use_argmax
        )
    
    # Analyze results
    if molecules:
        # Calculate statistics
        rewards = [mol.get('reward', 0) for mol in molecules if mol.get('reward') is not None]
        qeds = [mol.get('qed', 0) for mol in molecules]
        
        print(f"\nSampling Results:")
        print(f"Total molecules: {len(molecules)}")
        if rewards:
            print(f"Average reward: {sum(rewards)/len(rewards):.3f}")
            print(f"Max reward: {max(rewards):.3f}")
        if qeds:
            print(f"Average QED: {sum(qeds)/len(qeds):.3f}")
        
        # Diversity analysis
        diversity_stats = sampler.analyze_diversity(molecules)
        print(f"\nDiversity Analysis:")
        for key, value in diversity_stats.items():
            print(f"{key}: {value:.3f}")
        
        # Save results
        if args.output_file:
            output_path = args.output_file
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"sampled_molecules_{timestamp}.csv"
        
        save_molecules(molecules, output_path)
        
        # Print some example molecules
        print(f"\nExample molecules:")
        for i, mol in enumerate(molecules[:5]):
            print(f"{i+1}. {mol['smiles']} (reward: {mol.get('reward', 'N/A'):.3f})")
    
    else:
        print("No valid molecules were generated")


if __name__ == "__main__":
    main()
