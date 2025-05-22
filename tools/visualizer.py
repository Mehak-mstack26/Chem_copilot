import os
import traceback
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from langchain.tools import BaseTool
import time # Make sure this is imported
import re   # Make sure this is imported

class ChemVisualizer(BaseTool):
    name: str = "ChemVisualizer"
    description: str = "Visualizes chemical molecules and reactions from SMILES strings and returns the path to the image file."

    def _run(self, smiles_input: str) -> str:
        print(f"[VISUALIZER DEBUG] Received smiles_input (repr): {repr(smiles_input)}")
        input_type = self.detect_input_type(smiles_input)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Sanitize more aggressively for filenames, especially for complex SMILES
        sanitized_smiles = re.sub(r'[^a-zA-Z0-9_]', '_', smiles_input) # Allow only alphanumeric and underscore
        sanitized_smiles = sanitized_smiles[:50] # Truncate

        project_root_parts = os.path.abspath(__file__).split(os.sep)
        # Assuming 'Chem_copilot' is the project root folder name
        # and 'tools' is inside it.
        try:
            project_root_index = project_root_parts.index('Chem_copilot')
            project_root = os.sep.join(project_root_parts[:project_root_index + 1])
        except ValueError:
            # Fallback if 'Chem_copilot' not in path (e.g. running from different structure)
            # This assumes chemvisualizer.py is in a 'tools' subdir of the project root.
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            print(f"[VISUALIZER WARNING] Could not find 'Chem_copilot' in path, falling back to project_root: {project_root}")


        static_viz_dir = os.path.join(project_root, "static", "visualizations")
        os.makedirs(static_viz_dir, exist_ok=True)
        
        filename = f"{input_type}_{sanitized_smiles}_{timestamp}.png"
        output_filepath = os.path.join(static_viz_dir, filename)

        if input_type == 'reaction':
            result = self.visualize_reaction(smiles_input, output_file=output_filepath)
        else: # 'molecule' or any other fallback
            result = self.visualize_molecule(smiles_input, output_file=output_filepath)
        
        if result == output_filepath:
            # Return a path relative to the 'static' folder base for web serving
            return os.path.join("static", "visualizations", filename)
        else:
            return result # This will be an error message

    def detect_input_type(self, smiles_input):
        if '>>' in smiles_input:
            return 'reaction'
        else:
            return 'molecule'

    def visualize_reaction(self, rxn_smiles, output_file):
        try:
            print(f"[VISUALIZER DEBUG] Visualizing reaction: {rxn_smiles}")
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            if rxn is None or rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
                error_msg = f"Error: RDKit could not parse the reaction SMILES or found no reactants/products: '{rxn_smiles}'"
                print(f"[VISUALIZER ERROR] {error_msg}")
                return error_msg

            num_components = rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates()
            width = max(600, 150 * num_components)
            height = max(250, 100 + 50 * max(rxn.GetNumReactantTemplates(), rxn.GetNumProductTemplates()))

            # AllChem.Compute2DCoordsForReaction(rxn) # This can sometimes cause issues with complex reactions
                                                  # or if RDKit struggles with layout. Drawing often implies 2D coord gen.
                                                  # Try drawing directly first.
            drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
            dopts = drawer.drawOptions()
            dopts.dummiesAreAttachments = True
            dopts.includeAtomTags = False
            
            # --- POTENTIAL FIX ---
            dopts.bondLineWidth = 2 # Changed from 1.5 to an integer
            # ---------------------
            
            dopts.padding = 0.1 # Keep as float, this is usually fine

            print("[VISUALIZER DEBUG] Drawing reaction...")
            drawer.DrawReaction(rxn) # This implicitly generates 2D coords if not present
            drawer.FinishDrawing()
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
            print(f"[VISUALIZER DEBUG] Saved reaction image to {output_file}")
            return output_file
        except Chem.rdchem.AtomValenceException as ave:
            error_msg = f"Error visualizing reaction (AtomValenceException for '{rxn_smiles}'): {str(ave)}"
            print(f"[VISUALIZER ERROR] {error_msg}")
            return error_msg
        except Chem.rdchem.KekulizeException as ke:
             error_msg = f"Error visualizing reaction (KekulizeException for '{rxn_smiles}'): {str(ke)}"
             print(f"[VISUALIZER ERROR] {error_msg}")
             return error_msg
        except Exception as e:
            error_msg = f"Error visualizing reaction ('{rxn_smiles}'): {type(e).__name__}: {str(e)}"
            print(f"[VISUALIZER ERROR] {error_msg}\n{traceback.format_exc()}")
            return error_msg

    def visualize_molecule(self, smiles, output_file):
        try:
            print(f"[VISUALIZER DEBUG] Visualizing molecule: {smiles}")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Error: Failed to parse SMILES: {smiles}"

            AllChem.Compute2DCoords(mol) # Good to have explicit 2D coord generation for molecules
            
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
            dopts = drawer.drawOptions()
            
            # --- POTENTIAL FIX ---
            dopts.bondLineWidth = 2 
            dopts.padding = 0.1 
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
                
            print(f"[VISUALIZER DEBUG] Saved molecule image to {output_file}")
            return output_file
        except Exception as e:
            error_msg = f"Visualization error for molecule '{smiles}': {type(e).__name__}: {str(e)}"
            print(f"[VISUALIZER ERROR] {error_msg}\n{traceback.format_exc()}")
            return error_msg