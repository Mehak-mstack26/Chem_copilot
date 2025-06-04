import os
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D 

class ChemVisualizer: 

    def __init__(self):
        pass

    def detect_input_type(self, smiles_input: str) -> str: 
        """Detects if the input is a reaction or molecule SMILES."""
        if '>>' in smiles_input:
            return 'reaction'
        else:
            return 'molecule'

    def visualize_reaction(self, rxn_smiles: str, output_file: str) -> str:
        """
        Visualizes a reaction SMILES and saves it to the specified output_file.
        Returns the output_file path on success, or an error message string on failure.
        """
        try:
            print(f"[ChemVisualizer Class] Visualizing reaction: {rxn_smiles} to {output_file}")
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            if rxn is None or rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
                error_msg = f"Error: RDKit could not parse the reaction SMILES '{rxn_smiles}' or found no reactants/products."
                print(f"[ChemVisualizer ERROR] {error_msg}")
                return error_msg

            num_components = rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates()
            width_per_component = 200 
            min_width = 400
            width = max(min_width, width_per_component * num_components)

            max_mols_on_one_side = max(rxn.GetNumReactantTemplates(), rxn.GetNumProductTemplates(), 1) 
            height_per_row = 200 
            min_height = 250
            height = max(min_height, height_per_row * max_mols_on_one_side)


            drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
            dopts = drawer.drawOptions()
            dopts.dummiesAreAttachments = True
            dopts.includeAtomTags = False 
            dopts.bondLineWidth = 1.5 
            dopts.padding = 0.1
            drawer.DrawReaction(rxn)
            drawer.FinishDrawing()

            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText()) 
            print(f"[ChemVisualizer DEBUG] Saved reaction image to {output_file}")
            return output_file
        except Chem.rdchem.AtomValenceException as ave:
            error_msg = f"Error visualizing reaction (AtomValenceException for '{rxn_smiles}'): {str(ave)}"
            print(f"[ChemVisualizer ERROR] {error_msg}")
            return error_msg
        except Chem.rdchem.KekulizeException as ke:
            error_msg = f"Error visualizing reaction (KekulizeException for '{rxn_smiles}'): {str(ke)}"
            print(f"[ChemVisualizer ERROR] {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"General error visualizing reaction ('{rxn_smiles}'): {type(e).__name__}: {str(e)}"
            print(f"[ChemVisualizer ERROR] {error_msg}\n{traceback.format_exc()}")
            return error_msg

    def visualize_molecule(self, smiles: str, output_file: str) -> str:
        """
        Visualizes a molecule SMILES and saves it to the specified output_file.
        Returns the output_file path on success, or an error message string on failure.
        """
        try:
            print(f"[ChemVisualizer Class] Visualizing molecule: {smiles} to {output_file}")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                error_msg = f"Error: Failed to parse molecule SMILES: '{smiles}'"
                print(f"[ChemVisualizer ERROR] {error_msg}")
                return error_msg

            AllChem.Compute2DCoords(mol)
            
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 300) 
            dopts = drawer.drawOptions()
            dopts.bondLineWidth = 1.5 
            dopts.padding = 0.1

            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
                
            print(f"[ChemVisualizer DEBUG] Saved molecule image to {output_file}")
            return output_file 
        except Exception as e:
            error_msg = f"Error visualizing molecule ('{smiles}'): {type(e).__name__}: {str(e)}"
            print(f"[ChemVisualizer ERROR] {error_msg}\n{traceback.format_exc()}")
            return error_msg