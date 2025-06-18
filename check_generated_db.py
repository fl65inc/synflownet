import sqlite3
import pandas as pd

# Connect to the database
db_path = '/home/ubuntu/synflownet/logs/debug_run_reactions_task_2025-06-17_20-05-58/final/generated_objs_0.db'
conn = sqlite3.connect(db_path)

# Get table names
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print('Tables in the database:')
for table in tables:
    print(f'  - {table[0]}')

# Get schema for each table
for table in tables:
    table_name = table[0]
    print(f'\nSchema for table {table_name}:')
    cursor.execute(f'PRAGMA table_info({table_name});')
    columns = cursor.fetchall()
    for col in columns:
        print(f'  {col[1]} ({col[2]})')
    
    # Get row count
    cursor.execute(f'SELECT COUNT(*) FROM {table_name};')
    count = cursor.fetchone()[0]
    print(f'  Row count: {count}')
    
    # Show first few rows if table has data
    if count > 0:
        print(f'  Sample data:')
        cursor.execute(f'SELECT * FROM {table_name} LIMIT 3;')
        rows = cursor.fetchall()
        for i, row in enumerate(rows):
            print(f'    Row {i+1}: {row[:5]}...' if len(row) > 5 else f'    Row {i+1}: {row}')

conn.close()