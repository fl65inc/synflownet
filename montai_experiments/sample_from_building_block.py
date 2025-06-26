#!/usr/bin/env python3
"""
SynFlowNet Building Block-Specific Sampling Script

This script allows you to generate molecules starting from a specific building block.
It's useful for exploring the chemical space accessible from a particular starting material.

Usage:
    python sample_from_building_block.py --checkpoint_path /path/to/model.pt --building_block "CCO"
    
Examples:
    # Sample 1000 molecules starting from ethanol (CCO)
    python sample_from_building_block.py \
        --checkpoint_path ./logs/my_model/model_state.pt \
        --building_block "CCO" \
        --num_samples 1000 \
        --output_file ethanol_derivatives.csv
    
    # Sample from a specific building block index
    python sample_from_building_block.py \
        --checkpoint_path ./logs/my_model/model_state.pt \
        --building_block_index 426 \
        --num_samples 500
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Add the synflownet source to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from synflownet.tasks.reactions_task import ReactionTrainer
from synflownet.utils.synthesis_utils import clean_smiles
from synflownet.utils.synthesis_evals import calculate_molecular_diversity, to_fingerprint
from synflownet.envs.graph_building_env import GraphAction, GraphActionType


class BuildingBlockSampler:
    """
    A class for sampling molecules from a specific building block
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
        self.building_blocks = None
        
        self._load_model()
        self._load_building_blocks()
    
    def _load_model(self):
        """Load the trained model and associated components"""
        print(f"Loading model from {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        # Determine device to use
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set the worker device for sampling
        from synflownet.utils.misc import set_main_process_device
        set_main_process_device(self.device)
        
        # Load the trainer from checkpoint
        self.trainer = ReactionTrainer.load_from_checkpoint(self.checkpoint_path)
        
        # Extract components
        self.model = self.trainer.model
        self.task = self.trainer.task
        self.ctx = self.trainer.ctx
        self.sampler = self.trainer.sampler
        
        # Move model to the appropriate device
        self.model.to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        print("Model loaded successfully!")
        
        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {total_params:,} parameters")
    
    def _load_building_blocks(self):
        """Load the building blocks list from context"""
        self.building_blocks = self.ctx.building_blocks
        print(f"Loaded {len(self.building_blocks)} building blocks")
    
    def find_building_block_index(self, smiles: str) -> Optional[int]:
        """
        Find the index of a building block by its SMILES string
        
        Args:
            smiles: SMILES string of the building block
            
        Returns:
            Index of the building block or None if not found
        """
        # Canonicalize the input SMILES
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            canonical_smiles = Chem.MolToSmiles(mol)
        except:
            return None
        
        # Search for the building block
        for i, bb_smiles in enumerate(self.building_blocks):
            try:
                bb_mol = Chem.MolFromSmiles(bb_smiles)
                if bb_mol is not None:
                    bb_canonical = Chem.MolToSmiles(bb_mol)
                    if canonical_smiles == bb_canonical:
                        return i
            except:
                continue
        
        return None
    
    def list_similar_building_blocks(self, smiles: str, top_k: int = 10) -> List[Tuple[int, str, float]]:
        """
        Find building blocks similar to the given SMILES
        
        Args:
            smiles: Target SMILES string
            top_k: Number of similar building blocks to return
            
        Returns:
            List of (index, smiles, similarity) tuples
        """
        try:
            from rdkit.Chem import DataStructs
            from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
            
            target_mol = Chem.MolFromSmiles(smiles)
            if target_mol is None:
                return []
            
            target_fp = GetMorganFingerprintAsBitVect(target_mol, 2, nBits=2048)
            similarities = []
            
            for i, bb_smiles in enumerate(self.building_blocks):
                try:
                    bb_mol = Chem.MolFromSmiles(bb_smiles)
                    if bb_mol is not None:
                        bb_fp = GetMorganFingerprintAsBitVect(bb_mol, 2, nBits=2048)
                        similarity = DataStructs.TanimotoSimilarity(target_fp, bb_fp)
                        similarities.append((i, bb_smiles, similarity))
                except:
                    continue
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[2], reverse=True)
            return similarities[:top_k]
        
        except ImportError:
            print("RDKit not available for similarity calculation")
            return []
    
    def sample_from_building_block(self, 
                                 building_block_index: int,
                                 num_samples: int = 100,
                                 temperature: float = 1.0,
                                 random_action_prob: float = 0.0,
                                 max_attempts: int = None) -> List[Dict[str, Any]]:
        """
        Sample molecules starting from a specific building block
        
        Args:
            building_block_index: Index of the building block to start from
            num_samples: Number of molecules to sample
            temperature: Sampling temperature
            random_action_prob: Probability of random actions
            max_attempts: Maximum attempts (default: num_samples * 3)
            
        Returns:
            List of molecule dictionaries
        """
        if building_block_index < 0 or building_block_index >= len(self.building_blocks):
            raise ValueError(f"Building block index {building_block_index} out of range [0, {len(self.building_blocks)-1}]")
        
        bb_smiles = self.building_blocks[building_block_index]
        print(f"Sampling {num_samples} molecules starting from building block {building_block_index}: {bb_smiles}")
        
        if max_attempts is None:
            max_attempts = num_samples * 3
        
        molecules = []
        attempts = 0
        
        # Prepare conditioning (use default from task)
        cond_info = self.task.sample_conditional_information(1, train_it=0)
        cond_encoding = cond_info["encoding"]
        if hasattr(cond_encoding, 'to'):
            cond_encoding = cond_encoding.to(self.device)
        
        # Set sampling temperature
        original_temp = getattr(self.sampler, 'sample_temp', 1.0)
        self.sampler.sample_temp = temperature
        
        try:
            with torch.no_grad():
                while len(molecules) < num_samples and attempts < max_attempts:
                    attempts += 1
                    
                    # Sample a single trajectory starting from the specified building block
                    trajectory = self._sample_single_trajectory_from_bb(
                        building_block_index,
                        cond_encoding,
                        random_action_prob
                    )
                    
                    if trajectory is not None:
                        mol_data = self._process_trajectory(trajectory, len(molecules), building_block_index)
                        if mol_data is not None:
                            molecules.append(mol_data)
                            
                            # Progress update
                            if len(molecules) % 50 == 0:
                                print(f"Generated {len(molecules)}/{num_samples} molecules (attempts: {attempts})")
        
        finally:
            # Restore original temperature
            self.sampler.sample_temp = original_temp
        
        print(f"Successfully generated {len(molecules)}/{num_samples} valid molecules from {attempts} attempts")
        print(f"Success rate: {len(molecules)/attempts*100:.1f}%")
        
        return molecules
    
    def _sample_single_trajectory_from_bb(self, 
                                        building_block_index: int,
                                        cond_encoding: torch.Tensor,
                                        random_action_prob: float) -> Optional[Dict]:
        """
        Sample a single trajectory starting from a specific building block
        
        This manually starts the trajectory with the specified building block,
        then lets the model continue the synthesis.
        """
        try:
            # Start with empty graph
            graph = self.sampler.env.empty_graph()
            
            # Manually apply the first action: AddFirstReactant with our specific building block
            first_action = GraphAction(GraphActionType.AddFirstReactant, bb=building_block_index)
            
            # Step the environment with our forced first action
            mol = self.sampler.env.step(graph, first_action)
            current_graph = self.sampler.ctx.obj_to_graph(mol)
            
            # Initialize trajectory data
            trajectory_data = {
                'traj': [(graph, first_action)],
                'is_valid': True,
                'bbs': [building_block_index],
                'is_sink': [0]  # Not a terminal state yet
            }
            
            # Continue sampling from this state using the model
            # We'll modify the sampler's internal state and continue
            max_len = self.sampler.max_len - 1  # -1 because we already took first step
            
            for step in range(max_len):
                # Convert current graph to model input
                torch_graph = self.sampler.ctx.graph_to_Data(current_graph, traj_len=step+1)
                
                # Get model predictions
                fwd_cat, *_, _ = self.model(
                    self.sampler.ctx.collate([torch_graph]).to(self.device), 
                    cond_encoding
                )
                
                # Sample action
                actions = fwd_cat.sample(
                    nx_graphs=[current_graph], 
                    model=self.model,
                    random_action_prob=random_action_prob,
                    use_argmax=False
                )
                
                action_index = actions[0]
                graph_action = self.sampler.ctx.ActionIndex_to_GraphAction(torch_graph, action_index, fwd=True)
                
                # Add to trajectory
                trajectory_data['traj'].append((current_graph, graph_action))
                
                # Check if we're stopping
                if graph_action.action == GraphActionType.Stop:
                    trajectory_data['is_sink'].append(1)
                    trajectory_data['result'] = current_graph
                    break
                else:
                    trajectory_data['is_sink'].append(0)
                    
                    # Step the environment
                    try:
                        mol = self.sampler.env.step(current_graph, graph_action)
                        current_graph = self.sampler.ctx.obj_to_graph(mol)
                        
                        # Track building blocks used
                        if graph_action.action in [GraphActionType.AddFirstReactant, GraphActionType.ReactBi]:
                            if hasattr(graph_action, 'bb') and graph_action.bb is not None:
                                trajectory_data['bbs'].append(graph_action.bb)
                                
                    except Exception as e:
                        # Invalid action, mark trajectory as invalid
                        trajectory_data['is_valid'] = False
                        trajectory_data['result'] = current_graph
                        break
            else:
                # Reached max length without stopping
                trajectory_data['result'] = current_graph
            
            return trajectory_data
            
        except Exception as e:
            print(f"Error sampling trajectory: {e}")
            return None
    
    def _process_trajectory(self, trajectory: Dict, idx: int, starting_bb_index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single trajectory into molecule data
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
                'starting_building_block': self.building_blocks[starting_bb_index],
                'starting_bb_index': starting_bb_index,
                'is_valid': trajectory.get('is_valid', True),
                'trajectory_length': len(trajectory.get('traj', [])),
                'building_blocks_used': trajectory.get('bbs', []),
                'num_building_blocks': len(trajectory.get('bbs', [])),
                'reward': reward,
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
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
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
            # Calculate reward with required traj_lens argument
            # For single molecules, trajectory length is 1
            traj_lens = torch.tensor([1])
            obj_props, is_valid = self.task.compute_obj_properties([mol], traj_lens)
            
            if is_valid.any() and obj_props.numel() > 0:
                return obj_props[0].item()
            return None
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return None


def save_molecules(molecules: List[Dict[str, Any]], output_file: str):
    """Save molecules to CSV file"""
    if not molecules:
        print("No molecules to save")
        return
    
    df = pd.DataFrame(molecules)
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(molecules)} molecules to {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Valid molecules: {df['is_valid'].sum()}/{len(df)}")
    if 'reward' in df.columns and df['reward'].notna().any():
        print(f"Average reward: {df['reward'].mean():.3f} ± {df['reward'].std():.3f}")
    if 'qed' in df.columns:
        print(f"Average QED: {df['qed'].mean():.3f} ± {df['qed'].std():.3f}")
    if 'trajectory_length' in df.columns:
        print(f"Average trajectory length: {df['trajectory_length'].mean():.1f} ± {df['trajectory_length'].std():.1f}")
    
    # Show building blocks usage
    if 'num_building_blocks' in df.columns:
        print(f"Building blocks per molecule: {df['num_building_blocks'].mean():.1f} ± {df['num_building_blocks'].std():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Sample molecules from a specific building block")
    parser.add_argument("--checkpoint_path", required=True, 
                       help="Path to model checkpoint file")
    
    # Building block specification (mutually exclusive)
    bb_group = parser.add_mutually_exclusive_group(required=True)
    bb_group.add_argument("--building_block", type=str,
                         help="SMILES string of the building block to start from")
    bb_group.add_argument("--building_block_index", type=int,
                         help="Index of the building block to start from")
    
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of molecules to sample")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--random_action_prob", type=float, default=0.0,
                       help="Probability of random actions")
    parser.add_argument("--max_attempts", type=int, default=None,
                       help="Maximum sampling attempts")
    parser.add_argument("--output_file", default=None,
                       help="Output CSV file path")
    parser.add_argument("--list_similar", action="store_true",
                       help="List similar building blocks instead of sampling")
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = BuildingBlockSampler(args.checkpoint_path)
    
    # Determine building block index
    if args.building_block is not None:
        # Find building block by SMILES
        bb_index = sampler.find_building_block_index(args.building_block)
        if bb_index is None:
            print(f"Building block '{args.building_block}' not found in the database.")
            
            # Suggest similar building blocks
            similar = sampler.list_similar_building_blocks(args.building_block, top_k=10)
            if similar:
                print("\nSimilar building blocks found:")
                for i, (idx, smiles, similarity) in enumerate(similar):
                    print(f"{i+1:2d}. Index {idx:4d}: {smiles:30s} (similarity: {similarity:.3f})")
                print("\nUse --building_block_index with one of these indices.")
            return
        
        print(f"Found building block '{args.building_block}' at index {bb_index}")
        
        if args.list_similar:
            similar = sampler.list_similar_building_blocks(args.building_block, top_k=20)
            print(f"\nTop 20 building blocks similar to '{args.building_block}':")
            for i, (idx, smiles, similarity) in enumerate(similar):
                print(f"{i+1:2d}. Index {idx:4d}: {smiles:30s} (similarity: {similarity:.3f})")
            return
    else:
        bb_index = args.building_block_index
        if bb_index < 0 or bb_index >= len(sampler.building_blocks):
            print(f"Building block index {bb_index} out of range [0, {len(sampler.building_blocks)-1}]")
            return
        print(f"Using building block index {bb_index}: {sampler.building_blocks[bb_index]}")
    
    # Sample molecules
    molecules = sampler.sample_from_building_block(
        building_block_index=bb_index,
        num_samples=args.num_samples,
        temperature=args.temperature,
        random_action_prob=args.random_action_prob,
        max_attempts=args.max_attempts
    )
    
    # Save results
    if args.output_file:
        output_file = args.output_file
    else:
        # Generate default filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bb_smiles_safe = sampler.building_blocks[bb_index].replace('/', '_').replace('\\', '_')
        output_file = f"bb_{bb_index}_{bb_smiles_safe}_{timestamp}.csv"
    
    save_molecules(molecules, output_file)


if __name__ == "__main__":
    main()
