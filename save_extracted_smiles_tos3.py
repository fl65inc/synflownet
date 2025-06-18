from py65.utils.s3_wizard import archive_to_s3 
import pandas as pd

df = pd.read_csv("/home/ubuntu/synflownet/logs/debug_run_reactions_task_2025-06-17_20-05-58/final/extracted_smiles_with_rewards.csv", index_col=0)

print(df)

out = archive_to_s3(df, archive_name="synflow_mols_debug_run_reactions_task_2025-06-17_20-05-58", pref_ext='csv')
print(out)