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
        sanitized_smiles = re.sub(r'[^\w\-]+', '_', smiles_input)[:50]

        # Determine project root more reliably to locate the 'static' folder
        # Assumes 'tools' is a direct child of the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_viz_dir = os.path.join(project_root, "static", "visualizations")

        os.makedirs(static_viz_dir, exist_ok=True)
        
        filename = f"{input_type}_{sanitized_smiles}_{timestamp}.png"
        output_filepath = os.path.join(static_viz_dir, filename)

        if input_type == 'reaction':
            result = self.visualize_reaction(smiles_input, output_file=output_filepath)
        else:
            result = self.visualize_molecule(smiles_input, output_file=output_filepath)
        
        if result == output_filepath:
            # Return a relative path that Streamlit can use if 'static' is served
            # e.g., "static/visualizations/image.png"
            # This often works if Streamlit runs from the project root.
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
            print(f"[VISUALIZER DEBUG] Visualizing reaction: {rxn_smiles}") # For debugging
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            if rxn is None or rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
                # Adding more detail to the error message if parsing fails
                error_msg = f"Error: RDKit could not parse the reaction SMILES or found no reactants/products: '{rxn_smiles}'"
                print(f"[VISUALIZER ERROR] {error_msg}")
                return error_msg # Return the specific error

            num_components = rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates()
            width = max(600, 150 * num_components)
            height = max(250, 100 + 50 * max(rxn.GetNumReactantTemplates(), rxn.GetNumProductTemplates()))

            AllChem.Compute2DCoordsForReaction(rxn)
            drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
            dopts = drawer.drawOptions()
            dopts.dummiesAreAttachments = True
            # CORRECTED/REMOVED: dopts.atom = False
            # If you want to hide atom numbers, RDKit usually doesn't show them by default unless explicitly asked.
            # To hide atom symbols (e.g., C, O, N), you'd need more complex settings or custom drawing.
            # For default behavior (showing symbols, not numbers), these are often not needed:
            # dopts.atomLabels = False # Example: if you wanted to hide symbols
            dopts.includeAtomTags = False
            dopts.bondLineWidth = 1.5
            dopts.padding = 0.1

            print("[VISUALIZER DEBUG] Drawing reaction...") # For debugging
            drawer.DrawReaction(rxn)
            drawer.FinishDrawing()
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
            print(f"[VISUALIZER DEBUG] Saved reaction image to {output_file}") # For debugging
            return output_file
        except Chem.rdchem.AtomValenceException as ave: # Catch specific RDKit errors
            error_msg = f"Error visualizing reaction (AtomValenceException for '{rxn_smiles}'): {str(ave)}"
            print(f"[VISUALIZER ERROR] {error_msg}")
            return error_msg
        except Chem.rdchem.KekulizeException as ke:
             error_msg = f"Error visualizing reaction (KekulizeException for '{rxn_smiles}'): {str(ke)}"
             print(f"[VISUALIZER ERROR] {error_msg}")
             return error_msg
        except Exception as e: # General exception
            error_msg = f"Error visualizing reaction ('{rxn_smiles}'): {type(e).__name__}: {str(e)}"
            print(f"[VISUALIZER ERROR] {error_msg}\n{traceback.format_exc()}")
            return error_msg

    # ... visualize_molecule (ensure no similar invalid dopts here either) ...
    def visualize_molecule(self, smiles, output_file):
        try:
            print(f"[VISUALIZER DEBUG] Visualizing molecule: {smiles}")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return f"Error: Failed to parse SMILES: {smiles}"

            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Create drawer with appropriate size
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
            
            # Set drawing options
            dopts = drawer.drawOptions()
            dopts.bondLineWidth = 1.5
            dopts.padding = 0.1
            
            # Draw molecule directly - no highlighting
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
                
            print(f"[VISUALIZER DEBUG] Saved molecule image to {output_file}")
            return output_file
        except Exception as e:
            error_msg = f"Visualization error: {str(e)}"
            print(f"[VISUALIZER ERROR] {error_msg}")
            return error_msg
