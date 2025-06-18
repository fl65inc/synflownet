#!/usr/bin/env python3
"""
Extract all SMILES from the generated_objs_0.db database.
"""

import sqlite3
import pandas as pd
import os

def extract_smiles_from_db(db_path, output_format='txt'):
    """
    Extract all SMILES from the SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database
        output_format (str): Output format - 'txt', 'csv', or 'both'
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    try:
        # Query all SMILES with their rewards
        query = "SELECT smi, r FROM results ORDER BY r DESC"
        df = pd.read_sql_query(query, conn)
        
        print(f"Extracted {len(df)} SMILES from database")
        print(f"Reward range: {df['r'].min():.4f} - {df['r'].max():.4f}")
        
        # Create output directory
        output_dir = os.path.dirname(db_path)
        
        if output_format in ['txt', 'both']:
            # Save as text file (SMILES only)
            txt_path = os.path.join(output_dir, 'extracted_smiles.txt')
            with open(txt_path, 'w') as f:
                for smiles in df['smi']:
                    f.write(f"{smiles}\n")
            print(f"SMILES saved to: {txt_path}")
        
        if output_format in ['csv', 'both']:
            # Save as CSV file (SMILES with rewards and ranking)
            csv_path = os.path.join(output_dir, 'extracted_smiles_with_rewards.csv')
            df['rank'] = range(1, len(df) + 1)
            df.columns = ['SMILES', 'Reward', 'Rank']
            df.to_csv(csv_path, index=False)
            print(f"SMILES with rewards saved to: {csv_path}")
        
        # Print top 10 molecules
        print("\nTop 10 molecules by reward:")
        print("-" * 80)
        top_10 = df.head(10)
        for i in range(len(top_10)):
            smiles = top_10.iloc[i]['smi']
            reward = top_10.iloc[i]['r']
            print(f"{i+1:2d}. {smiles} (reward: {reward:.4f})")
        
        # Basic statistics
        print(f"\nStatistics:")
        print(f"Total molecules: {len(df)}")
        print(f"Unique molecules: {df['smi'].nunique()}")
        print(f"Duplicates: {len(df) - df['smi'].nunique()}")
        
    except Exception as e:
        print(f"Error extracting data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Path to the database
    db_path = '/home/ubuntu/synflownet/logs/debug_run_reactions_task_2025-06-17_20-05-58/final/generated_objs_0.db'
    
    # Extract SMILES in both formats
    extract_smiles_from_db(db_path, output_format='both')
    
    print("\nExtraction complete!")
