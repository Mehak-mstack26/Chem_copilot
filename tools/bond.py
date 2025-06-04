from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Any, Dict, List, Optional, Union # Keep standard typing
import re
from rxnmapper import RXNMapper 

class BondChangeAnalyzer:
    _mapper: RXNMapper 

    def __init__(self):
        """Initialize the BondChangeAnalyzer with RXNMapper"""
        self._mapper = RXNMapper()

    def _map_reaction(self, rxn_smiles: str) -> str:
        """Add atom mapping to an unmapped reaction SMILES using RXNMapper"""
        try:
            atom_map_pattern = r':[0-9]+'
            if re.search(atom_map_pattern, rxn_smiles):
                print(f"[BondChangeAnalyzer._map_reaction] Reaction already mapped: {rxn_smiles}")
                return rxn_smiles

            original_rxn_for_log = rxn_smiles
            if " " in rxn_smiles:
                rxn_smiles = rxn_smiles.replace(" ", "")
            
            # This + to . conversion should be cautious.
            # A more robust way is to parse reactants/products individually and then join.
            # For now, keeping your existing logic.
            if "+" in rxn_smiles and not re.search(r'\[[^\]]*\+[^\]]*\]', rxn_smiles): # Slightly improved + check
                parts = rxn_smiles.split(">>")
                if len(parts) == 2:
                    reactants, products = parts
                    # More robust replacement of + by . for reactants/products separation
                    # Split by '.', then filter out empty strings, then rejoin with '.'
                    # This is still a simplification; proper parsing is better.
                    reactants = ".".join(filter(None, re.split(r'\.(?![^\[]*\])', reactants.replace("+", "."))))
                    products = ".".join(filter(None, re.split(r'\.(?![^\[]*\])', products.replace("+", "."))))
                    rxn_smiles = f"{reactants}>>{products}"
            
            print(f"[BondChangeAnalyzer._map_reaction] Attempting to map (RXNMapper): {original_rxn_for_log} -> {rxn_smiles}")
            results = self._mapper.get_attention_guided_atom_maps([rxn_smiles]) # RXNMapper expects a list
            
            if results and 'mapped_rxn' in results[0] and results[0]['mapped_rxn']:
                print(f"[BondChangeAnalyzer._map_reaction] RXNMapper success: {results[0]['mapped_rxn']}")
                return results[0]['mapped_rxn']
            else:
                print(f"[BondChangeAnalyzer._map_reaction] RXNMapper did not return a mapped reaction for: {rxn_smiles}. Results: {results}")
                # Fallback if RXNMapper fails or returns empty
                return self._rdkit_map_reaction(original_rxn_for_log) # Pass original for RDKit fallback

        except Exception as e:
            print(f"[BondChangeAnalyzer._map_reaction] RXNMapper error for '{rxn_smiles}': {str(e)}. Trying RDKit fallback.")
            return self._rdkit_map_reaction(rxn_smiles) # Fallback with potentially modified SMILES

    def _rdkit_map_reaction(self, rxn_smiles: str) -> str:
        """Fallback mapping using RDKit's ReactionMapAtoms."""
        try:
            print(f"[BondChangeAnalyzer._rdkit_map_reaction] Attempting RDKit mapping for: {rxn_smiles}")
            # RDKit's ReactionFromSmarts is sensitive to format.
            # Ensure reactants and products are dot-separated if multiple.
            # The previous + to . conversion might handle this.
            rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
            if not rxn:
                print(f"[BondChangeAnalyzer._rdkit_map_reaction] RDKit ReactionFromSmarts failed to parse: {rxn_smiles}")
                return ""
            
            map_success_code = AllChem.ReactionMapAtoms(rxn)
            
            mapped_smiles_check = AllChem.ReactionToSmiles(rxn)
            if not re.search(r':[0-9]+', mapped_smiles_check): 
                 print(f"[BondChangeAnalyzer._rdkit_map_reaction] RDKit ReactionMapAtoms did not produce mapped atoms. Code: {map_success_code}")
                 return ""

            print(f"[BondChangeAnalyzer._rdkit_map_reaction] RDKit mapping successful. Code: {map_success_code}")
            return mapped_smiles_check
        except Exception as inner_e:
            print(f"[BondChangeAnalyzer._rdkit_map_reaction] RDKit mapping error for '{rxn_smiles}': {str(inner_e)}")
            return ""

    def _get_bond_changes(self, mapped_rxn: str) -> Dict[str, Any]: 
        """Extract bonds broken, formed, and changed based on the mapped reaction"""
        try:
            mapped_reactants_smi, mapped_products_smi = mapped_rxn.split(">>")
            
            reactant_mol = Chem.MolFromSmiles(mapped_reactants_smi)
            product_mol = Chem.MolFromSmiles(mapped_products_smi)
            
            if not reactant_mol:
                return {"error": f"Could not parse mapped reactants: {mapped_reactants_smi}", "bonds_broken": [], "bonds_formed": [], "bonds_changed": []}
            if not product_mol:
                return {"error": f"Could not parse mapped products: {mapped_products_smi}", "bonds_broken": [], "bonds_formed": [], "bonds_changed": []}

            reactant_atoms_map_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reactant_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
            product_atoms_map_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in product_mol.GetAtoms() if atom.GetAtomMapNum() > 0}

            reactant_bonds = {}
            for bond in reactant_mol.GetBonds():
                map_nums = sorted([bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()])
                if map_nums[0] > 0 and map_nums[1] > 0: 
                    reactant_bonds[tuple(map_nums)] = str(bond.GetBondType())
            
            product_bonds = {}
            for bond in product_mol.GetBonds():
                map_nums = sorted([bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()])
                if map_nums[0] > 0 and map_nums[1] > 0:
                    product_bonds[tuple(map_nums)] = str(bond.GetBondType())
            
            bonds_broken = []
            bonds_changed = []
            
            for bond_key_maps, r_bond_type in reactant_bonds.items():
                map1, map2 = bond_key_maps
                
                atom1_symbol = reactant_mol.GetAtomWithIdx(reactant_atoms_map_to_idx[map1]).GetSymbol()
                atom2_symbol = reactant_mol.GetAtomWithIdx(reactant_atoms_map_to_idx[map2]).GetSymbol()

                if bond_key_maps not in product_bonds:
                    bonds_broken.append(f"{atom1_symbol}-{atom2_symbol} ({r_bond_type}) between atoms mapped {map1}-{map2}")
                elif product_bonds[bond_key_maps] != r_bond_type:
                    p_bond_type = product_bonds[bond_key_maps]
                    bonds_changed.append(f"{atom1_symbol}-{atom2_symbol}: {r_bond_type} to {p_bond_type} between atoms mapped {map1}-{map2}")
            
            bonds_formed = []
            for bond_key_maps, p_bond_type in product_bonds.items():
                if bond_key_maps not in reactant_bonds:
                    map1, map2 = bond_key_maps
                    atom1_symbol = product_mol.GetAtomWithIdx(product_atoms_map_to_idx[map1]).GetSymbol()
                    atom2_symbol = product_mol.GetAtomWithIdx(product_atoms_map_to_idx[map2]).GetSymbol()
                    bonds_formed.append(f"{atom1_symbol}-{atom2_symbol} ({p_bond_type}) between atoms mapped {map1}-{map2}")
            
            return {
                "bonds_broken": bonds_broken,
                "bonds_formed": bonds_formed,
                "bonds_changed": bonds_changed
            }
        except Exception as e:

            return {"error": f"Error extracting bond changes from '{mapped_rxn}': {str(e)}",
                   "bonds_broken": [], "bonds_formed": [], "bonds_changed": []}

    def _run(self, rxn_smiles: str) -> Dict[str, Any]: 
        """Run the tool on a reaction SMILES string (mapped or unmapped)"""
        print(f"[BondChangeAnalyzer Class] _run called with: {rxn_smiles}")
        try:
            if ">>" not in rxn_smiles:
                return {"error": "Input is not a valid reaction SMILES (missing '>>')."}
            
            parts = rxn_smiles.split(">>")
            if len(parts) == 2:
                cleaned_rxn_smiles = parts[0].strip() + ">>" + parts[1].strip()
            else: 
                cleaned_rxn_smiles = rxn_smiles.strip()

            mapped_rxn = self._map_reaction(cleaned_rxn_smiles) 
            
            if not mapped_rxn:
                return {"error": "Failed to map the reaction. Please check the reaction SMILES format and validity."}
            
            bond_changes_result = self._get_bond_changes(mapped_rxn) 
            
            if "error" in bond_changes_result: 
                return bond_changes_result 
                
            result = {
                "mapped_reaction": mapped_rxn,
                "bonds_broken": bond_changes_result["bonds_broken"],
                "bonds_formed": bond_changes_result["bonds_formed"],
                "bonds_changed": bond_changes_result["bonds_changed"]
            }
            
            if mapped_rxn != cleaned_rxn_smiles and not re.search(r':[0-9]+', cleaned_rxn_smiles):
                result["note"] = "The reaction was automatically mapped for analysis."
                
            return result
        except Exception as e:
            return {"error": f"Critical error in BondChangeAnalyzer tool processing '{rxn_smiles}': {str(e)}"}

