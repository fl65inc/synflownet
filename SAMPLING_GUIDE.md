# SynFlowNet Molecule Sampling Guide

This guide explains how to use trained SynFlowNet models to sample new molecules with various strategies and constraints.

## Overview

SynFlowNet generates molecules through a sequential synthesis process using chemical reactions and building blocks. After training, you can use the model to sample new molecules with different strategies.

## Basic Usage

### 1. Simple Molecule Sampling

```bash
python sample_molecules.py --checkpoint_path ./logs/my_model/model_state.pt --num_samples 100
```

This will:
- Load the trained model from the checkpoint
- Sample 100 new molecules
- Calculate molecular properties and rewards
- Save results to a CSV file

### 2. Filtered Sampling

```bash
python sample_molecules.py \
    --checkpoint_path ./logs/my_model/model_state.pt \
    --num_samples 50 \
    --reward_threshold 0.7 \
    --output_file high_reward_molecules.csv
```

This filters molecules by minimum reward threshold and drug-likeness criteria.

### 3. Temperature Control

```bash
python sample_molecules.py \
    --checkpoint_path ./logs/my_model/model_state.pt \
    --num_samples 100 \
    --temperature 0.5 \
    --use_argmax
```

- Lower temperature (0.1-0.5): More deterministic, focuses on high-probability molecules
- Higher temperature (1.5-2.0): More exploration, diverse but potentially lower quality
- `--use_argmax`: Completely deterministic sampling

## Advanced Sampling

### 1. Conditional Generation

```bash
python advanced_sampling.py \
    --checkpoint_path ./logs/my_model/model_state.pt \
    --mode conditional \
    --num_samples 100 \
    --config_file sampling_config.json
```

This uses focus directions to guide generation towards specific regions of chemical space.

### 2. Property Optimization

```bash
python advanced_sampling.py \
    --checkpoint_path ./logs/my_model/model_state.pt \
    --mode optimize \
    --num_samples 50
```

Iteratively optimizes molecules towards target drug-like properties.

### 3. Diverse Molecule Selection

```bash
python advanced_sampling.py \
    --checkpoint_path ./logs/my_model/model_state.pt \
    --mode diverse \
    --num_samples 50
```

Generates many molecules and selects a diverse subset using clustering.

## Understanding the Output

### CSV Columns

- `smiles`: SMILES string of the generated molecule
- `reward`: Model's reward score for the molecule
- `qed`: Drug-likeness score (0-1, higher is better)
- `molecular_weight`: Molecular weight in Daltons
- `logp`: Lipophilicity (partition coefficient)
- `tpsa`: Topological polar surface area
- `num_atoms`: Total number of atoms
- `num_rings`: Number of rings
- `is_valid`: Whether the molecule is chemically valid
- `trajectory_length`: Number of synthesis steps
- `forward_logprob`: Model's confidence in the synthesis path

### Synthesis Information

- `num_synthesis_steps`: Steps in the synthesis pathway
- `num_building_blocks`: Number of building blocks used
- `building_blocks_indices`: Indices of building blocks in the database
- `reaction_types`: Types of reactions used in synthesis

## Model Checkpoints

### Finding Checkpoints

Trained models save checkpoints in the log directory:
```
./logs/my_experiment/
├── model_state.pt          # Latest checkpoint
├── model_state_1000.pt     # Checkpoint at step 1000
├── model_state_2000.pt     # Checkpoint at step 2000
└── ...
```

### Checkpoint Contents

Each checkpoint contains:
- Model weights (`models_state_dict`)
- Optimizer state (`optimizer_state_dict`)
- Training configuration (`cfg`)
- Training step (`step`)

## Sampling Strategies

### 1. Greedy Sampling (Deterministic)
```python
# Most confident predictions
molecules = sampler.sample_molecules(
    num_samples=100,
    use_argmax=True,
    temperature=0.1
)
```

### 2. Stochastic Sampling (Exploration)
```python
# Balanced exploration
molecules = sampler.sample_molecules(
    num_samples=100,
    temperature=1.0,
    random_action_prob=0.1
)
```

### 3. High-Temperature Sampling (Diversity)
```python
# Maximum diversity
molecules = sampler.sample_molecules(
    num_samples=100,
    temperature=2.0,
    random_action_prob=0.2
)
```

## Conditional Generation

### Focus Directions

For multi-objective optimization, you can specify focus directions:

```python
focus_directions = [
    [1.0, 0.0],  # Maximize first objective
    [0.0, 1.0],  # Maximize second objective  
    [0.7, 0.3],  # Weighted combination
]
```

### Custom Conditioning

```python
# Target specific properties
conditional_info = {
    'temperature': 32.0,
    'focus_direction': [0.5, 0.5],
    'preference_weights': [0.8, 0.2]
}
```

## Filtering and Selection

### Drug-Likeness Filters

The sampler applies Lipinski's Rule of Five by default:
- Molecular weight ≤ 500 Da
- LogP ≤ 5
- Hydrogen bond donors ≤ 5  
- Hydrogen bond acceptors ≤ 10

### Custom Filters

```python
def custom_filter(mol_data):
    return (
        mol_data.get('qed', 0) > 0.6 and
        mol_data.get('molecular_weight', 0) < 400 and
        mol_data.get('num_rings', 0) >= 2
    )
```

## Analysis and Visualization

### Diversity Analysis

```python
diversity_stats = sampler.analyze_diversity(molecules)
print(f"Unique molecules: {diversity_stats['unique_molecules']}")
print(f"Diversity score: {diversity_stats['diversity_score']:.3f}")
```

### Property Distributions

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame(molecules)
df['qed'].hist(bins=20)
plt.xlabel('QED Score')
plt.ylabel('Count')
plt.title('QED Distribution of Generated Molecules')
plt.show()
```

### Clustering Analysis

```python
# Cluster by molecular fingerprints
diverse_molecules = sampler.cluster_and_select_diverse(
    molecules, 
    num_clusters=10,
    selection_strategy='best_reward'
)
```

## Troubleshooting

### Common Issues

1. **Low valid molecule rate**
   - Decrease temperature
   - Use argmax sampling
   - Check model training quality

2. **Low diversity**
   - Increase temperature
   - Add random action probability
   - Use different focus directions

3. **Poor rewards**
   - Check if model was trained on similar objectives
   - Try conditional generation with appropriate focus
   - Verify reward function is working correctly

### Memory Issues

For large-scale sampling:
- Reduce batch size
- Sample in multiple rounds
- Use lower precision (float16)

### GPU Usage

The sampling scripts will automatically use GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Example Workflows

### 1. Drug Discovery Pipeline

```bash
# 1. Generate diverse candidates
python advanced_sampling.py --mode diverse --num_samples 1000

# 2. Filter for drug-likeness  
python sample_molecules.py --reward_threshold 0.6 --num_samples 100

# 3. Optimize top candidates
python advanced_sampling.py --mode optimize --num_samples 50
```

### 2. Chemical Space Exploration

```bash
# 1. Sample with different temperatures
for temp in 0.5 1.0 1.5 2.0; do
    python sample_molecules.py --temperature $temp --num_samples 250
done

# 2. Conditional sampling
python advanced_sampling.py --mode conditional --num_samples 500

# 3. Analyze diversity and select representatives
python advanced_sampling.py --mode diverse --num_samples 100
```

### 3. Property-Targeted Design

```bash
# Configure target properties in sampling_config.json
python advanced_sampling.py \
    --mode optimize \
    --num_samples 200 \
    --config_file sampling_config.json
```

## Configuration Files

### sampling_config.json

Customize sampling parameters:

```json
{
  "sampling": {
    "temperature_range": [0.5, 1.5],
    "batch_size": 50
  },
  "filtering": {
    "min_qed": 0.5,
    "max_molecular_weight": 500
  },
  "optimization": {
    "target_properties": {
      "qed": [0.7, 1.0],
      "molecular_weight": [200, 400]
    }
  }
}
```

## Performance Tips

1. **Batch Processing**: Use larger batch sizes for GPU efficiency
2. **Parallel Sampling**: Run multiple sampling processes
3. **Checkpointing**: Save intermediate results for long runs
4. **Memory Management**: Clear cache between large sampling runs

## Integration with Other Tools

### RDKit Integration

```python
from rdkit import Chem
from rdkit.Chem import Draw

# Visualize molecules
mols = [Chem.MolFromSmiles(mol['smiles']) for mol in molecules[:4]]
img = Draw.MolsToGridImage(mols, molsPerRow=2)
img.save('generated_molecules.png')
```

### ChEMBL/PubChem Similarity

```python
# Check similarity to known drugs
from rdkit.Chem import DataStructs

def check_novelty(smiles, reference_db):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    
    max_similarity = 0
    for ref_smiles in reference_db:
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
        similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity < 0.4  # Novel if <40% similar
```

This guide should help you effectively use SynFlowNet for molecular generation. Adjust parameters based on your specific use case and computational resources.
