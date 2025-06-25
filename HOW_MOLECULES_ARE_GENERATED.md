# How Molecules Are Generated in SynFlowNet - Simplified Explanation

## The Big Picture 🎯

SynFlowNet generates molecules like a chemist would in a lab - by **combining building blocks using chemical reactions**, step by step. It's NOT like drawing atoms one by one, but more like LEGO blocks that snap together with specific rules.

## Key Components 🧩

### 1. **Building Blocks** 
- Pre-made small molecules (like LEGO pieces)
- Stored in `src/synflownet/data/building_blocks/`
- Examples: simple molecules like "CCO" (ethanol), "c1ccccc1" (benzene)
- Over 300,000 building blocks available from Enamine chemical supplier

### 2. **Reaction Templates**
- Rules for how building blocks can combine
- Written as SMIRKS patterns (chemical reaction notation)
- Examples: 
  - Unimolecular: One molecule → modified molecule
  - Bimolecular: Two molecules → combined molecule

### 3. **The Environment** (`ReactionTemplateEnv`)
- The "chemistry lab" where reactions happen
- Enforces which reactions are chemically valid
- Prevents impossible combinations

## How Generation Works Step-by-Step 🔄

### Step 1: Start Empty
```
Current molecule: (empty)
```

### Step 2: Add First Building Block
```
Action: AddFirstReactant(building_block_426)
Current molecule: "Cc1ccc(CC#N)c(C(=O)O)c1"
```

### Step 3: Apply Reaction 1
```
Action: ReactBi(reaction_72, building_block_3824)
- Take current molecule + building block 3824
- Apply bimolecular reaction template 72
- Get: intermediate molecule
```

### Step 4: Apply Reaction 2  
```
Action: ReactBi(reaction_34, building_block_2381)
- Take intermediate + building block 2381
- Apply bimolecular reaction template 34
- Get: "Cc1ccc(CC#N)c(C(=O)N2CCC(S(C)(=O)=O)CC2)c1"
```

### Step 5: Stop
```
Action: Stop
Final molecule: "Cc1ccc(CC#N)c(C(=O)N2CCC(S(C)(=O)=O)CC2)c1"
```

## The AI Model's Role 🤖

The neural network (GFlowNet) learns to:

1. **Choose which building block** to start with
2. **Choose which reaction** to apply next
3. **Choose which building block** to add (for bimolecular reactions)
4. **Decide when to stop**

It's trained to make choices that lead to molecules with high rewards (good drug-like properties).

## How Reinforcement Learning (GFlowNet) Works Here 🎮

### Traditional RL vs GFlowNet

**Traditional RL (like playing chess):**
- Goal: Find THE best move/strategy
- Explores to find optimal policy
- Focuses on maximizing expected reward

**GFlowNet (for molecule generation):**
- Goal: Sample diverse molecules proportional to their rewards
- Explores to find ALL good molecules, not just the best one
- Focuses on diverse sampling from the reward landscape

### The Learning Process 📚

1. **Forward Policy (πF)**: Learns to build molecules step by step
   - Input: Current molecule state
   - Output: Probability distribution over next actions
   - Example: "Given this partial molecule, what's the probability of adding building block X?"

2. **Backward Policy (πB)**: Learns to decompose molecules  
   - Input: Current molecule
   - Output: Probability of how it was built
   - Used for training stability and flow consistency

3. **Flow Training**: Ensures flow conservation
   - Flow in = Flow out for each state
   - Guarantees sampling proportional to rewards
   - Uses Trajectory Balance (TB) loss function

### Training Loop 🔄

```
1. Sample trajectory: Empty → Building block → Reactions → Final molecule
2. Calculate reward: How good is this molecule? (QED, docking score, etc.)
3. Update policies: Adjust probabilities to favor high-reward paths
4. Ensure flow balance: In-flow = Out-flow for consistency
5. Repeat thousands of times
```

### Key RL Components in SynFlowNet

**State Space**: All possible partial molecules during synthesis
**Action Space**: {AddFirstReactant, ReactUni, ReactBi, Stop}
**Reward Function**: Molecular properties (drug-likeness, binding affinity, etc.)
**Policy**: Neural network that outputs action probabilities
**Environment**: Chemistry rules (valid reactions, building blocks)

### Why GFlowNet for Molecules? 🧬

✅ **Diversity**: Samples many different good molecules, not just one "best" one
✅ **Proportional sampling**: Better molecules are sampled more often
✅ **Mode coverage**: Finds molecules in all high-reward regions
✅ **Amortized inference**: One model generates unlimited diverse molecules
✅ **Credit assignment**: Properly assigns reward to each synthesis step

### Training Details 🎯

**Reward Signal**: 
- Calculated on final molecule only
- Backpropagated through entire synthesis trajectory
- Examples: QED score, docking affinity, synthetic accessibility

**Exploration Strategy**:
- Learns to explore chemical space systematically
- Balances exploitation (known good molecules) vs exploration (novel regions)
- Uses flow-based exploration rather than ε-greedy

**Multi-objective**: 
- Can optimize multiple properties simultaneously
- Uses conditioning vectors to control focus
- Example: High QED + Low molecular weight + Good docking score

### Mathematical Framework 📊

The model learns a flow function F(s) such that:
```
P(molecule) ∝ R(molecule)
```

Where:
- P(molecule) = Probability of sampling this molecule
- R(molecule) = Reward/desirability of this molecule
- Flow ensures proper normalization across all possible molecules

This means if molecule A has 2x the reward of molecule B, the model will sample molecule A twice as often as molecule B.

## Code Flow 💻

### During Training 🏋️
```
1. Sample trajectory using current policy
2. Environment applies reactions → Build molecule step by step  
3. Calculate final reward → How good is this molecule?
4. Compute loss (Trajectory Balance) → Update policy weights
5. Repeat for thousands of molecules
```

### During Sampling/Generation 🎲
```
1. Trained model predicts action probabilities
2. Sample action → ReactionTemplateEnv.step()
3. Environment applies reaction → New molecule state
4. Check if valid/complete → Continue or stop
5. Log result → SQLite database
6. Repeat until desired number of molecules
```

## Key Files 📁

- **`synthesis_building_env.py`** - The main environment and chemistry engine
- **`trainer.py`** - Runs the generation process
- **`reactions_task.py`** - Defines rewards and objectives
- **`sqlite_log.py`** - Logs results to database

## Why This Approach? 🔬

✅ **Chemically valid**: Every step follows real chemistry rules
✅ **Synthesizable**: The recorded path shows how to make it in a lab
✅ **Efficient**: No need to check millions of invalid combinations
✅ **Interpretable**: Each step has chemical meaning

## Analogy 🏗️

Think of it like:
- **Building blocks** = LEGO pieces
- **Reactions** = Ways pieces can connect 
- **Environment** = The rules of what fits together
- **AI Model** = An architect deciding which pieces to use and how to connect them
- **Final molecule** = The completed building

The AI learns to be a good "molecular architect" by trying many combinations and getting rewarded for building useful molecules (like potential drugs).

## Real Example from Database 📊

From the 6,400 molecules generated:

```
Trajectory: [
  ('', AddFirstReactant(426)),
  ('Cc1ccc(CC#N)c(C(=O)O)c1', ReactBi(72, 3824)),  
  ('Intermediate...', ReactBi(34, 2381)),
  ('Final molecule', Stop)
]
```

This shows exactly how the AI "thought" step by step to build this specific molecule, making it completely reproducible in a real lab!

## Scaling Up: Generating More Molecules 🚀

### Can I Generate One Million Molecules? ✅ YES!

You can absolutely use your **existing trained model** to generate one million molecules without any retraining. Here's how:

### Option 1: Modify Configuration ⚙️

**Change the config parameters** in your experiment:

```yaml
# In your config file (e.g., config.yaml)
num_final_gen_steps: 15625    # Number of batches
algo:
  num_from_policy: 64         # Molecules per batch

# Total: 15625 × 64 = 1,000,000 molecules
```

**Or go even bigger:**
```yaml
num_final_gen_steps: 1562     # Fewer batches
algo:
  num_from_policy: 640        # Larger batch size

# Total: 1562 × 640 = 999,680 molecules (close to 1M)
```

### Option 2: Use Sampling Scripts 🐍

**Use the provided sampling scripts** (much easier):

```bash
# Generate 1 million molecules
python montai_experiments/advanced_sampling.py \
    --checkpoint_path ./logs/your_model/model_state.pt \
    --mode basic \
    --num_samples 1000000 \
    --output_dir ./million_molecules/

# Or use the simple sampler
python montai_experiments/sample_molecules.py \
    --checkpoint_path ./logs/your_model/model_state.pt \
    --num_samples 1000000 \
    --output_file million_molecules.csv
```

### Option 3: Batch Processing 📦

**Generate in chunks** to avoid memory issues:

```bash
# Generate 10 batches of 100k each
for i in {1..10}; do
    python montai_experiments/sample_molecules.py \
        --checkpoint_path ./logs/your_model/model_state.pt \
        --num_samples 100000 \
        --output_file molecules_batch_${i}.csv
done

# Then combine all files
cat molecules_batch_*.csv > million_molecules.csv
```

### Hardware Considerations 💻

**For 1 million molecules:**
- **RAM**: ~8-16 GB (depending on batch size)
- **Storage**: ~500 MB - 2 GB for results
- **Time**: 1-6 hours (depending on GPU/CPU)
- **GPU**: Recommended but not required

**Optimization tips:**
```bash
# Use larger batch sizes for GPU efficiency
--batch_size 128   # Default is usually 50-64

# Use multiple workers if you have many CPUs
--num_workers 4

# Save memory by processing in chunks
--max_attempts 10000  # For filtered sampling
```

### What You Get 📊

**Output for 1M molecules:**
- **CSV file**: All molecular properties, rewards, synthesis info
- **SMILES file**: Just the chemical structures  
- **SQLite DB**: Complete trajectories and metadata
- **Analysis**: Diversity, property distributions, top molecules

### No Retraining Needed! 🎯

**Why you don't need to retrain:**
✅ **Model is generative**: Trained to sample from learned distribution
✅ **Unlimited sampling**: Can generate as many molecules as you want
✅ **Consistent quality**: Same reward distribution as training
✅ **Diverse output**: GFlowNet ensures variety even with many samples

### Advanced Scaling Options 🔧

**1. Distributed Generation:**
```bash
# Run on multiple machines/GPUs simultaneously
# Each generates 100k, combine results
```

**2. Conditional Generation:**
```bash
# Generate molecules with specific properties
python advanced_sampling.py --mode conditional --num_samples 1000000
```

**3. Filtered Generation:**
```bash
# Only keep high-quality molecules
python sample_molecules.py \
    --num_samples 1000000 \
    --reward_threshold 0.8 \
    --max_attempts 10000000
```

### Practical Example 💡

**Your current setup generated 6,400 molecules:**
- Config: `num_final_gen_steps: 100`, `num_from_policy: 64`
- Total: 100 × 64 = 6,400

**To get 1 million, just change to:**
- Config: `num_final_gen_steps: 15625`, `num_from_policy: 64`  
- Total: 15,625 × 64 = 1,000,000

**Or use the sampling scripts** which are more flexible and don't require config changes!

### Bottom Line 🎉

Your trained model is like a **molecular generator machine** - once trained, you can use it to generate unlimited molecules without any additional training. Just point it at your checkpoint and specify how many you want!
