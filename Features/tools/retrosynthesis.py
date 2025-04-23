# from langchain.tools import BaseTool
# import requests
# import re
# import os
# import subprocess
# import urllib.parse
# from tools.name2smiles import NameToSMILES  # Update this with the correct import path

# # Initialize NameToSMILES tool for chemical name conversion
# name_to_smiles_tool = NameToSMILES()

# def get_smiles_for_compound(compound_name):
#     """Get SMILES notation for a compound name using the NameToSMILES tool"""
#     try:
#         result = name_to_smiles_tool._run(compound_name)
#         if "SMILES:" in result:
#             # Extract SMILES from the result
#             smiles = result.split("SMILES:")[1].split("\n")[0].strip()
#             return smiles
#         return None
#     except Exception:
#         return None

# def run_retrosynthesis(compound_name):
#     """
#     Run the RetroSynthesis Agent for a given compound and process the results
#     to ensure all reactions have proper SMILES
#     """
#     try:
#         # Get the path to the RetroSynthesisAgent directory
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         parent_dir = os.path.dirname(current_dir)  # Get out of 'tools'
#         project_root = os.path.dirname(parent_dir)  # Get out of 'Features'
#         retro_agent_dir = os.path.join(project_root, "RetroSynthesisAgent")
        
#         # Save current directory
#         original_dir = os.getcwd()
        
#         # Change to RetroSynthesisAgent directory
#         os.chdir(retro_agent_dir)
        
#         # Run the shell script
#         result = subprocess.run(
#             ["/bin/sh", "runRetroSynAgent.sh", compound_name],
#             capture_output=True,
#             text=True
#         )
        
#         # Change back to original directory
#         os.chdir(original_dir)
        
#         # Parse the output
#         output = result.stdout
        
#         # Extract the recommended pathway section
#         recommended_match = re.search(r"Recommended Reaction Pathway:(.*?)Reasons:", output, re.DOTALL)
#         if not recommended_match:
#             return {"status": "error", "message": "Could not find recommended pathway in output"}
        
#         pathway_text = recommended_match.group(1).strip()
#         indices_match = re.search(r"Recommended Reaction Pathway: (.*?)$", output, re.MULTILINE)
#         indices = indices_match.group(1).strip() if indices_match else ""
#         recommended_indices = re.findall(r"idx(\d+)", indices)
        
#         # Extract all reactions
#         reactions = []
#         reaction_blocks = re.findall(r"Reaction idx: (\d+)\s+Reactants: (.*?)\s+Products: (.*?)\s+Reaction SMILES: (.*?)\s+Conditions: (.*?)\s+Source: (.*?)(?:\s+SourceLink: (.*?))?(?=\s+\s+Reaction|\Z)", output, re.DOTALL)
        
#         for block in reaction_blocks:
#             idx, reactants_str, products_str, smiles, conditions, source = block[:6]
#             source_link = block[6] if len(block) > 6 else None
            
#             # Parse reactants and products
#             reactants = [r.strip() for r in reactants_str.split(",")]
#             products = [p.strip() for p in products_str.split(",")]
            
#             # Clean up the reaction SMILES - remove placeholders if present
#             original_smiles = smiles.strip()
#             cleaned_smiles = original_smiles.replace("[reactant_SMILES]", "").replace("[product_SMILES]", "")
#             cleaned_smiles = re.sub(r'\[[a-zA-Z_]+_SMILES\]', "", cleaned_smiles)
            
#             # Check if SMILES is valid or needs generation
#             valid_smiles = cleaned_smiles if ">>" in cleaned_smiles and not re.search(r'\[.*?\]', cleaned_smiles) else None
            
#             # If no valid SMILES, generate from reactants and products
#             if not valid_smiles:
#                 # First try to get SMILES for each reactant and product
#                 reactant_smiles = []
#                 for reactant in reactants:
#                     smiles = get_smiles_for_compound(reactant)
#                     if smiles:
#                         reactant_smiles.append(smiles)
#                     else:
#                         # Keep original name if conversion fails
#                         reactant_smiles.append(reactant)
                
#                 product_smiles = []
#                 for product in products:
#                     smiles = get_smiles_for_compound(product)
#                     if smiles:
#                         product_smiles.append(smiles)
#                     else:
#                         # Keep original name if conversion fails
#                         product_smiles.append(product)
                
#                 # Create reaction SMILES from reactants and products
#                 reactants_str = '.'.join(reactant_smiles)
#                 products_str = '.'.join(product_smiles)
#                 generated_smiles = f"{reactants_str}>>{products_str}"
                
#                 # Use the generated SMILES if it looks reasonable
#                 if ">>" in generated_smiles:
#                     valid_smiles = generated_smiles
            
#             reactions.append({
#                 "idx": idx,
#                 "reactants": reactants,
#                 "products": products,
#                 "reaction_smiles": original_smiles,
#                 "cleaned_reaction_smiles": valid_smiles,
#                 "conditions": conditions.strip(),
#                 "source": source.strip(),
#                 "source_link": source_link.strip() if source_link else None
#             })
        
#         # Extract reasoning
#         reasoning_match = re.search(r"Reasons:\s+(.*?)(?:={10,}|\Z)", output, re.DOTALL)
#         reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
#         return {
#             "status": "success",
#             "data": {
#                 "reactions": reactions,
#                 "reasoning": reasoning,
#                 "recommended_indices": recommended_indices
#             }
#         }
    
#     except Exception as e:
#         import traceback
#         return {
#             "status": "error", 
#             "message": str(e),
#             "traceback": traceback.format_exc()
#         }




from langchain.tools import BaseTool
import requests
import re
import os
import subprocess
import urllib.parse
from tools.name2smiles import NameToSMILES  # Update this with the correct import path

# Initialize NameToSMILES tool for chemical name conversion
name_to_smiles_tool = NameToSMILES()

def get_smiles_for_compound(compound_name):
    """Get SMILES notation for a compound name using the NameToSMILES tool"""
    try:
        result = name_to_smiles_tool._run(compound_name)
        if "SMILES:" in result:
            # Extract SMILES from the result
            smiles = result.split("SMILES:")[1].split("\n")[0].strip()
            return smiles
        return None
    except Exception:
        return None

def run_retrosynthesis(compound_name):
    """
    Run the RetroSynthesis Agent for a given compound and process the results
    to ensure all reactions have proper SMILES
    """
    try:
        # Get the path to the RetroSynthesisAgent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)  # Get out of 'tools'
        project_root = os.path.dirname(parent_dir)  # Get out of 'Features'
        retro_agent_dir = os.path.join(project_root, "RetroSynthesisAgent")
        
        # Save current directory
        original_dir = os.getcwd()
        
        # Change to RetroSynthesisAgent directory
        os.chdir(retro_agent_dir)
        
        # Run the shell script
        result = subprocess.run(
            ["/bin/sh", "runRetroSynAgent.sh", compound_name],
            capture_output=True,
            text=True
        )
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Parse the output
        output = result.stdout
        
        # Extract the recommended pathway section
        recommended_match = re.search(r"Recommended Reaction Pathway:(.*?)Reasons:", output, re.DOTALL)
        if not recommended_match:
            return {"status": "error", "message": "Could not find recommended pathway in output"}
        
        pathway_text = recommended_match.group(1).strip()
        indices_match = re.search(r"Recommended Reaction Pathway: (.*?)$", output, re.MULTILINE)
        indices = indices_match.group(1).strip() if indices_match else ""
        recommended_indices = re.findall(r"idx(\d+)", indices)
        
        # Extract all reactions
        reactions = []
        reaction_blocks = re.findall(r"Reaction idx: (\d+)\s+Reactants: (.*?)\s+Products: (.*?)\s+Reaction SMILES: (.*?)\s+Conditions: (.*?)\s+Source: (.*?)(?:\s+SourceLink: (.*?))?(?=\s+\s+Reaction|\Z)", output, re.DOTALL)
        
        for block in reaction_blocks:
            idx, reactants_str, products_str, smiles, conditions, source = block[:6]
            source_link = block[6] if len(block) > 6 else None
            
            # Parse reactants and products
            reactants = [r.strip() for r in reactants_str.split(",")]
            products = [p.strip() for p in products_str.split(",")]
            
            # Clean up the reaction SMILES - remove placeholders if present
            original_smiles = smiles.strip()
            cleaned_smiles = original_smiles.replace("[reactant_SMILES]", "").replace("[product_SMILES]", "")
            cleaned_smiles = re.sub(r'\[[a-zA-Z_]+_SMILES\]', "", cleaned_smiles)
            
            # Check if SMILES is valid or needs generation
            valid_smiles = cleaned_smiles if ">>" in cleaned_smiles and not re.search(r'\[.*?\]', cleaned_smiles) else None
            
            # If no valid SMILES, generate from reactants and products
            if not valid_smiles:
                # First try to get SMILES for each reactant and product
                reactant_smiles = []
                for reactant in reactants:
                    smiles = get_smiles_for_compound(reactant)
                    if smiles:
                        reactant_smiles.append(smiles)
                    else:
                        # Keep original name if conversion fails
                        reactant_smiles.append(reactant)
                
                product_smiles = []
                for product in products:
                    smiles = get_smiles_for_compound(product)
                    if smiles:
                        product_smiles.append(smiles)
                    else:
                        # Keep original name if conversion fails
                        product_smiles.append(product)
                
                # Create reaction SMILES from reactants and products
                reactants_str = '.'.join(reactant_smiles)
                products_str = '.'.join(product_smiles)
                generated_smiles = f"{reactants_str}>>{products_str}"
                
                # Use the generated SMILES if it looks reasonable
                if ">>" in generated_smiles:
                    valid_smiles = generated_smiles
            
            reactions.append({
                "idx": idx,
                "reactants": reactants,
                "products": products,
                "reaction_smiles": original_smiles,
                "cleaned_reaction_smiles": valid_smiles,
                "conditions": conditions.strip(),
                "source": source.strip(),
                "source_link": source_link.strip() if source_link else None
            })
        
        # Extract reasoning
        reasoning_match = re.search(r"Reasons:\s+(.*?)(?:={10,}|\Z)", output, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {
            "status": "success",
            "data": {
                "reactions": reactions,
                "reasoning": reasoning,
                "recommended_indices": recommended_indices
            }
        }
    
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }