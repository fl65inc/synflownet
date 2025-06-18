from rdkit import Chem
with Chem.SDMolSupplier('/home/ubuntu/synflownet/src/synflownet/data/building_blocks/Enamine_Building_Blocks_Stock_312882cmpd_20250604.sdf') as suppl:
    # save the smiles to a text file
    with open('/home/ubuntu/synflownet/src/synflownet/data/building_blocks/enamine_bbs.txt', 'w') as f:
        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                f.write(smiles + '\n')
print("Done!")