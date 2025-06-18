#!/usr/bin/env python3
"""
Advanced SynFlowNet Molecule Sampling Script

This script provides advanced sampling capabilities including:
- Conditional generation with custom focus directions
- Multi-objective sampling
- Batch sampling with progress tracking
- Synthesis pathway analysis
- Molecule clustering and selection

Usage:
    python advanced_sampling.py --checkpoint_path /path/to/model.pt --mode conditional
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add synflownet to path
sys.path.append(str(Path(__file__).parent / "src"))

from synflownet.tasks.reactions_task import ReactionTrainer
from synflownet.utils.synthesis_utils import clean_smiles
from synflownet.utils.synthesis_evals import (
    calculate_molecular_diversity, 
    to_fingerprint,
    get_centroid_for_top_k_clusters
)


class AdvancedMoleculeSampler:
    """Advanced molecule sampling with conditional generation and analysis"""
    
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.trainer = ReactionTrainer.load_from_checkpoint(checkpoint_path)
        self.model = self.trainer.model
        self.task = self.trainer.task
        self.ctx = self.trainer.ctx
        self.sampler = self.trainer.sampler
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Task: {type(self.task).__name__}")
        print(f"Environment: {type(self.ctx).__name__}")
    
    def conditional_sampling(self, 
                           focus_directions: List[List[float]],
                           samples_per_condition: int = 50,
                           temperature_range: Tuple[float, float] = (0.1, 2.0),
                           **kwargs) -> List[Dict[str, Any]]:
        """
        Sample molecules with specific focus directions for multi-objective optimization
        
        Args:
            focus_directions: List of focus direction vectors for conditioning
            samples_per_condition: Number of samples per focus direction
            temperature_range: Range of temperatures to sample from
            
        Returns:
            List of sampled molecules with conditioning info
        """
        print(f"Conditional sampling with {len(focus_directions)} focus directions")
        
        all_molecules = []
        
        for i, focus_dir in enumerate(tqdm(focus_directions, desc="Focus directions")):
            print(f"\nSampling with focus direction {i+1}/{len(focus_directions)}: {focus_dir}")
            
            # Sample temperatures for this focus direction
            temperatures = np.random.uniform(
                temperature_range[0], 
                temperature_range[1], 
                samples_per_condition
            )
            
            for temp in temperatures:
                # Prepare conditional information
                cond_info = self._create_conditional_info(
                    focus_direction=focus_dir,
                    temperature=temp,
                    num_samples=1
                )
                
                # Sample molecule
                molecules = self.sample_molecules(
                    num_samples=1,
                    conditional_info=cond_info,
                    temperature=temp,
                    **kwargs
                )
                
                # Add conditioning metadata
                for mol in molecules:
                    mol['focus_direction'] = focus_dir
                    mol['focus_index'] = i
                    mol['sampling_temperature'] = temp
                
                all_molecules.extend(molecules)
        
        return all_molecules
    
    def _create_conditional_info(self, 
                               focus_direction: List[float] = None,
                               temperature: float = 1.0,
                               preference_weights: List[float] = None,
                               num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Create conditional information tensor for the model"""
        device = next(self.model.parameters()).device
        
        # Create base conditioning tensor
        cond_dim = self.task.num_cond_dim
        cond_tensor = torch.zeros((num_samples, cond_dim), device=device)
        
        # Add temperature conditioning
        if hasattr(self.task, 'temperature_conditional') and self.task.temperature_conditional:
            cond_tensor[:, 0] = temperature
        
        # Add focus direction conditioning (for multi-objective tasks)
        if focus_direction and hasattr(self.task, 'focus_conditional') and self.task.focus_conditional:
            focus_tensor = torch.tensor(focus_direction, device=device, dtype=torch.float32)
            if len(focus_tensor) <= cond_dim - 1:  # Leave space for temperature
                cond_tensor[:, 1:1+len(focus_tensor)] = focus_tensor
        
        # Add preference weights if provided
        if preference_weights and hasattr(self.task, 'preference_conditional'):
            pref_tensor = torch.tensor(preference_weights, device=device, dtype=torch.float32)
            start_idx = 1 + (len(focus_direction) if focus_direction else 0)
            if start_idx + len(pref_tensor) <= cond_dim:
                cond_tensor[:, start_idx:start_idx+len(pref_tensor)] = pref_tensor
        
        return {"encoding": cond_tensor}
    
    def sample_molecules(self, 
                        num_samples: int = 100,
                        conditional_info: Optional[Dict] = None,
                        temperature: float = 1.0,
                        random_action_prob: float = 0.0,
                        use_argmax: bool = False,
                        batch_size: int = 50) -> List[Dict[str, Any]]:
        """Sample molecules in batches with progress tracking"""
        
        all_molecules = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Sampling batches"):
                current_batch_size = min(batch_size, num_samples - len(all_molecules))
                
                # Prepare conditioning
                if conditional_info is None:
                    cond_info = self.task.sample_conditional_information(current_batch_size, step=0)
                    cond_encoding = cond_info["encoding"]
                else:
                    cond_encoding = conditional_info["encoding"]
                    if cond_encoding.size(0) == 1 and current_batch_size > 1:
                        cond_encoding = cond_encoding.repeat(current_batch_size, 1)
                
                # Set sampling temperature
                original_temp = getattr(self.sampler, 'sample_temp', 1.0)
                self.sampler.sample_temp = temperature
                
                try:
                    # Sample batch
                    trajectories = self.sampler.sample_from_model(
                        model=self.model,
                        n=current_batch_size,
                        cond_info=cond_encoding,
                        random_action_prob=random_action_prob,
                        use_argmax=use_argmax
                    )
                    
                    # Process trajectories
                    batch_molecules = []
                    for i, traj in enumerate(trajectories):
                        mol_data = self._process_trajectory(traj, len(all_molecules) + i)
                        if mol_data is not None:
                            batch_molecules.append(mol_data)
                    
                    all_molecules.extend(batch_molecules)
                    
                finally:
                    # Restore original temperature
                    self.sampler.sample_temp = original_temp
        
        return all_molecules
    
    def _process_trajectory(self, trajectory: Dict, idx: int) -> Optional[Dict[str, Any]]:
        """Process trajectory into molecule data with synthesis analysis"""
        try:
            final_graph = trajectory['result']
            mol = self.ctx.graph_to_obj(final_graph)
            
            if mol is None:
                return None
            
            smiles = Chem.MolToSmiles(mol)
            clean_smi = clean_smiles(smiles)
            
            if clean_smi is None:
                return None
            
            # Basic properties
            properties = self._calculate_properties(mol)
            reward = self._calculate_reward(mol)
            
            # Synthesis pathway analysis
            synthesis_info = self._analyze_synthesis_pathway(trajectory)
            
            mol_data = {
                'idx': idx,
                'smiles': clean_smi,
                'is_valid': trajectory.get('is_valid', True),
                'trajectory_length': len(trajectory.get('traj', [])),
                'reward': reward,
                'forward_logprob': trajectory.get('fwd_logprob', torch.tensor(0.0)).item(),
                'backward_logprob': trajectory.get('bck_logprob', torch.tensor(0.0)).item(),
                **properties,
                **synthesis_info
            }
            
            return mol_data
            
        except Exception as e:
            print(f"Error processing trajectory {idx}: {e}")
            return None
    
    def _calculate_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate comprehensive molecular properties"""
        try:
            properties = {
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'qed': QED.qed(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                'num_rings': Descriptors.RingCount(mol),
                'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_hbd': Descriptors.NumHDonors(mol),
                'num_hba': Descriptors.NumHAcceptors(mol),
                'formal_charge': Chem.GetFormalCharge(mol),
                'sa_score': self._calculate_sa_score(mol),
            }
            
            # Add fingerprint for similarity analysis
            fingerprint = to_fingerprint(mol, finger_type='morgan', n_bits=1024)
            properties['fingerprint'] = fingerprint.tolist()
            
            return properties
        except Exception as e:
            print(f"Error calculating properties: {e}")
            return {}
    
    def _calculate_sa_score(self, mol: Chem.Mol) -> float:
        """Calculate synthetic accessibility score"""
        try:
            from synflownet.utils.sascore import calculateScore
            return calculateScore(mol)
        except:
            return 0.0
    
    def _calculate_reward(self, mol: Chem.Mol) -> Optional[float]:
        """Calculate reward using task's objective function"""
        try:
            graph = self.ctx.obj_to_graph(mol)
            reward = self.task.compute_obj_properties([graph])
            if len(reward) > 0:
                return reward[0].item()
            return None
        except Exception as e:
            return None
    
    def _analyze_synthesis_pathway(self, trajectory: Dict) -> Dict[str, Any]:
        """Analyze the synthesis pathway from trajectory"""
        try:
            traj = trajectory.get('traj', [])
            building_blocks_used = trajectory.get('bbs', [])
            
            synthesis_info = {
                'num_synthesis_steps': len(traj),
                'num_building_blocks': len(building_blocks_used),
                'building_blocks_indices': building_blocks_used,
            }
            
            # Analyze reaction types used
            reaction_types = []
            for state, action in traj:
                if hasattr(action, 'action') and hasattr(action.action, 'name'):
                    reaction_types.append(action.action.name)
            
            synthesis_info['reaction_types'] = reaction_types
            synthesis_info['unique_reaction_types'] = len(set(reaction_types))
            
            return synthesis_info
        except Exception as e:
            return {'synthesis_error': str(e)}
    
    def cluster_and_select_diverse(self, 
                                 molecules: List[Dict[str, Any]], 
                                 num_clusters: int = 10,
                                 selection_strategy: str = 'centroid') -> List[Dict[str, Any]]:
        """
        Cluster molecules and select diverse representatives
        
        Args:
            molecules: List of molecule dictionaries
            num_clusters: Number of clusters
            selection_strategy: 'centroid', 'best_reward', or 'random'
            
        Returns:
            Selected diverse molecules
        """
        if len(molecules) <= num_clusters:
            return molecules
        
        print(f"Clustering {len(molecules)} molecules into {num_clusters} clusters")
        
        # Extract fingerprints
        fingerprints = []
        valid_molecules = []
        
        for mol in molecules:
            if 'fingerprint' in mol and mol['fingerprint']:
                fingerprints.append(mol['fingerprint'])
                valid_molecules.append(mol)
        
        if len(fingerprints) == 0:
            print("No fingerprints available for clustering")
            return molecules[:num_clusters]
        
        fingerprints = np.array(fingerprints)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(num_clusters, len(fingerprints)), random_state=42)
        cluster_labels = kmeans.fit_predict(fingerprints)
        
        # Select representatives from each cluster
        selected_molecules = []
        
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_molecules = [mol for i, mol in enumerate(valid_molecules) if cluster_mask[i]]
            
            if not cluster_molecules:
                continue
            
            if selection_strategy == 'centroid':
                # Find molecule closest to centroid
                cluster_fps = fingerprints[cluster_mask]
                centroid = cluster_fps.mean(axis=0)
                distances = np.linalg.norm(cluster_fps - centroid, axis=1)
                selected_idx = np.argmin(distances)
                selected_molecules.append(cluster_molecules[selected_idx])
                
            elif selection_strategy == 'best_reward':
                # Select molecule with highest reward
                best_mol = max(cluster_molecules, 
                             key=lambda x: x.get('reward', 0) if x.get('reward') is not None else 0)
                selected_molecules.append(best_mol)
                
            elif selection_strategy == 'random':
                # Random selection
                import random
                selected_molecules.append(random.choice(cluster_molecules))
        
        print(f"Selected {len(selected_molecules)} diverse molecules")
        return selected_molecules
    
    def optimize_molecules(self, 
                         target_properties: Dict[str, Tuple[float, float]],
                         num_samples: int = 100,
                         num_iterations: int = 5,
                         learning_rate: float = 0.1) -> List[Dict[str, Any]]:
        """
        Iteratively optimize molecules towards target properties
        
        Args:
            target_properties: Dict mapping property names to (min, max) ranges
            num_samples: Number of samples per iteration
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for conditioning adjustment
            
        Returns:
            Optimized molecules
        """
        print(f"Optimizing molecules for {num_iterations} iterations")
        
        best_molecules = []
        
        # Initial conditioning (random)
        current_conditioning = torch.randn(1, self.task.num_cond_dim) * 0.1
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Sample with current conditioning
            cond_info = {"encoding": current_conditioning.repeat(num_samples, 1)}
            molecules = self.sample_molecules(
                num_samples=num_samples,
                conditional_info=cond_info,
                temperature=1.0 - 0.1 * iteration  # Decrease temperature over time
            )
            
            # Evaluate molecules
            scores = []
            for mol in molecules:
                score = self._evaluate_target_properties(mol, target_properties)
                scores.append(score)
                mol['optimization_score'] = score
            
            # Keep best molecules
            molecules.sort(key=lambda x: x['optimization_score'], reverse=True)
            best_molecules.extend(molecules[:10])  # Keep top 10 per iteration
            
            # Update conditioning based on best molecules
            if iteration < num_iterations - 1:
                current_conditioning = self._update_conditioning(
                    current_conditioning, molecules[:5], learning_rate
                )
        
        # Return overall best molecules
        best_molecules.sort(key=lambda x: x['optimization_score'], reverse=True)
        return best_molecules[:num_samples // 2]
    
    def _evaluate_target_properties(self, 
                                  mol: Dict[str, Any], 
                                  target_properties: Dict[str, Tuple[float, float]]) -> float:
        """Evaluate how well molecule matches target properties"""
        score = 0.0
        num_properties = 0
        
        for prop_name, (min_val, max_val) in target_properties.items():
            if prop_name in mol and mol[prop_name] is not None:
                value = mol[prop_name]
                
                # Score based on how well value fits in target range
                if min_val <= value <= max_val:
                    score += 1.0
                else:
                    # Penalty based on distance from range
                    if value < min_val:
                        penalty = (min_val - value) / (max_val - min_val)
                    else:
                        penalty = (value - max_val) / (max_val - min_val)
                    score += max(0, 1.0 - penalty)
                
                num_properties += 1
        
        return score / max(1, num_properties)
    
    def _update_conditioning(self, 
                           current_conditioning: torch.Tensor,
                           best_molecules: List[Dict[str, Any]],
                           learning_rate: float) -> torch.Tensor:
        """Update conditioning based on best molecules (simplified version)"""
        # This is a simplified update - in practice you might want more sophisticated methods
        if best_molecules:
            # Small random perturbation towards success
            perturbation = torch.randn_like(current_conditioning) * learning_rate
            return current_conditioning + perturbation
        return current_conditioning


def main():
    parser = argparse.ArgumentParser(description="Advanced SynFlowNet molecule sampling")
    parser.add_argument("--checkpoint_path", required=True, help="Model checkpoint path")
    parser.add_argument("--mode", choices=['basic', 'conditional', 'optimize', 'diverse'], 
                       default='basic', help="Sampling mode")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--output_dir", default="./sampling_results", help="Output directory")
    parser.add_argument("--config_file", help="JSON config file for advanced parameters")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize sampler
    sampler = AdvancedMoleculeSampler(args.checkpoint_path)
    
    # Load configuration
    config = {}
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Sample based on mode
    if args.mode == 'basic':
        molecules = sampler.sample_molecules(num_samples=args.num_samples)
        
    elif args.mode == 'conditional':
        # Define focus directions for multi-objective sampling
        focus_directions = [
            [1.0, 0.0],  # Focus on first objective
            [0.0, 1.0],  # Focus on second objective
            [0.5, 0.5],  # Balanced
            [0.8, 0.2],  # Mostly first objective
            [0.2, 0.8],  # Mostly second objective
        ]
        
        molecules = sampler.conditional_sampling(
            focus_directions=focus_directions,
            samples_per_condition=args.num_samples // len(focus_directions)
        )
        
    elif args.mode == 'optimize':
        # Target drug-like properties
        target_properties = {
            'molecular_weight': (200, 500),
            'logp': (0, 5),
            'qed': (0.5, 1.0),
            'num_hbd': (0, 5),
            'num_hba': (0, 10)
        }
        
        molecules = sampler.optimize_molecules(
            target_properties=target_properties,
            num_samples=args.num_samples
        )
        
    elif args.mode == 'diverse':
        # Generate many molecules and select diverse subset
        all_molecules = sampler.sample_molecules(num_samples=args.num_samples * 3)
        molecules = sampler.cluster_and_select_diverse(
            all_molecules, 
            num_clusters=args.num_samples,
            selection_strategy='centroid'
        )
    
    # Save results
    if molecules:
        output_file = Path(args.output_dir) / f"molecules_{args.mode}.csv"
        df = pd.DataFrame(molecules)
        
        # Remove fingerprint column for CSV (too large)
        if 'fingerprint' in df.columns:
            fingerprints = df['fingerprint'].tolist()
            df = df.drop('fingerprint', axis=1)
            
            # Save fingerprints separately
            fp_file = Path(args.output_dir) / f"fingerprints_{args.mode}.pkl"
            with open(fp_file, 'wb') as f:
                pickle.dump(fingerprints, f)
        
        df.to_csv(output_file, index=False)
        print(f"Saved {len(molecules)} molecules to {output_file}")
        
        # Print summary statistics
        if 'reward' in df.columns:
            print(f"Average reward: {df['reward'].mean():.3f}")
            print(f"Max reward: {df['reward'].max():.3f}")
        
        if 'qed' in df.columns:
            print(f"Average QED: {df['qed'].mean():.3f}")
        
        print(f"Valid molecules: {df['is_valid'].sum()}/{len(df)}")


if __name__ == "__main__":
    main()
