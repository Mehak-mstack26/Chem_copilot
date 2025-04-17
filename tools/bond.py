from langchain.tools import BaseTool
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Any, Dict, List, Optional, Union
import re
from pydantic import PrivateAttr
from rxnmapper import RXNMapper

class BondChangeAnalyzer(BaseTool):
    name: str = "BondChangeAnalyzer"
    description: str = "Identifies bonds broken, formed, and changed in any chemical reaction, whether mapped or unmapped."

    _mapper: RXNMapper = PrivateAttr()
    
    def __init__(self):
        """Initialize the BondChangeAnalyzer with RXNMapper"""
        super().__init__()
        self._mapper = RXNMapper()  # Use _mapper instead of rxn_mapper to match the PrivateAttr name
    
    def _map_reaction(self, rxn_smiles: str) -> str:
        """Add atom mapping to an unmapped reaction SMILES using RXNMapper"""
        try:
            # Check if the reaction is already mapped
            atom_map_pattern = r':[0-9]+'
            if re.search(atom_map_pattern, rxn_smiles):
                return rxn_smiles  # Already mapped
            
            # Convert to standard reaction SMILES format if needed
            if " " in rxn_smiles:
                # Remove spaces only
                rxn_smiles = rxn_smiles.replace(" ", "")
            
            # Convert standalone + characters to . but preserve + inside brackets
            # This regex looks for + not inside square brackets
            if "+" in rxn_smiles and not re.search(r'\[\w+\+\]', rxn_smiles):
                parts = rxn_smiles.split(">>")
                if len(parts) == 2:
                    reactants, products = parts
                    # Only replace + that are not inside square brackets
                    reactants = re.sub(r'(?<!\[[\w])\+(?![\w]\])', '.', reactants)
                    products = re.sub(r'(?<!\[[\w])\+(?![\w]\])', '.', products)
                    rxn_smiles = f"{reactants}>>{products}"
            
            # Map the reaction using RXNMapper
            results = self._mapper.get_attention_guided_atom_maps([rxn_smiles])
            if results and 'mapped_rxn' in results[0]:
                return results[0]['mapped_rxn']
            return ""
        except Exception as e:
            print(f"RXNMapper error: {str(e)}")
            # Fallback to RDKit mapping if RXNMapper fails
            try:
                rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
                if not rxn:
                    return ""
                
                success = AllChem.ReactionMapAtoms(rxn)
                if not success:
                    return ""
                
                mapped_smiles = AllChem.ReactionToSmiles(rxn)
                return mapped_smiles
            except Exception as inner_e:
                print(f"RDKit mapping error: {str(inner_e)}")
                return ""
    
    def _get_bond_changes(self, mapped_rxn: str) -> Dict[str, List[str]]:
        """Extract bonds broken, formed, and changed based on the mapped reaction"""
        try:
            # Parse mapped reaction
            mapped_reactants, mapped_products = mapped_rxn.split(">>")
            
            # Get molecules with atom mapping
            reactant_mol = Chem.MolFromSmiles(mapped_reactants)
            product_mol = Chem.MolFromSmiles(mapped_products)
            
            if not reactant_mol or not product_mol:
                return {"bonds_broken": [], "bonds_formed": [], "bonds_changed": []}
            
            # Extract atom maps
            reactant_atoms = {}
            for atom in reactant_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    reactant_atoms[map_num] = atom.GetIdx()
            
            product_atoms = {}
            for atom in product_mol.GetAtoms():
                map_num = atom.GetAtomMapNum()
                if map_num > 0:
                    product_atoms[map_num] = atom.GetIdx()
            
            # Extract bonds in reactants and products
            reactant_bonds = {}
            for bond in reactant_mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                begin_map = begin_atom.GetAtomMapNum()
                end_map = end_atom.GetAtomMapNum()
                if begin_map > 0 and end_map > 0:
                    # Store as tuple of atom maps and bond type
                    bond_key = (min(begin_map, end_map), max(begin_map, end_map))
                    bond_type = bond.GetBondType()
                    reactant_bonds[bond_key] = str(bond_type)
            
            product_bonds = {}
            for bond in product_mol.GetBonds():
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                begin_map = begin_atom.GetAtomMapNum()
                end_map = end_atom.GetAtomMapNum()
                if begin_map > 0 and end_map > 0:
                    bond_key = (min(begin_map, end_map), max(begin_map, end_map))
                    bond_type = bond.GetBondType()
                    product_bonds[bond_key] = str(bond_type)
            
            # Find differences
            bonds_broken = []
            bonds_changed = []
            
            for bond_key, bond_type in reactant_bonds.items():
                map1, map2 = bond_key
                
                # Bond exists in reactants but not in products - broken
                if bond_key not in product_bonds:
                    # Get atom symbols
                    atom1 = reactant_mol.GetAtomWithIdx(reactant_atoms[map1]).GetSymbol()
                    atom2 = reactant_mol.GetAtomWithIdx(reactant_atoms[map2]).GetSymbol()
                    bonds_broken.append(f"{atom1}-{atom2} ({bond_type})")
                # Bond exists in both but changed type
                elif product_bonds[bond_key] != bond_type:
                    atom1 = reactant_mol.GetAtomWithIdx(reactant_atoms[map1]).GetSymbol()
                    atom2 = reactant_mol.GetAtomWithIdx(reactant_atoms[map2]).GetSymbol()
                    bonds_changed.append(f"{atom1}-{atom2}: {bond_type} → {product_bonds[bond_key]}")
            
            bonds_formed = []
            for bond_key, bond_type in product_bonds.items():
                map1, map2 = bond_key
                
                # Bond exists in products but not in reactants - formed
                if bond_key not in reactant_bonds:
                    # Get atom symbols
                    atom1 = product_mol.GetAtomWithIdx(product_atoms[map1]).GetSymbol()
                    atom2 = product_mol.GetAtomWithIdx(product_atoms[map2]).GetSymbol()
                    bonds_formed.append(f"{atom1}-{atom2} ({bond_type})")
            
            return {
                "bonds_broken": bonds_broken,
                "bonds_formed": bonds_formed,
                "bonds_changed": bonds_changed
            }
        except Exception as e:
            return {"error": f"Error extracting bond changes: {str(e)}",
                   "bonds_broken": [], "bonds_formed": [], "bonds_changed": []}
    
    def _run(self, rxn_smiles: str) -> Dict[str, Any]:
        """Run the tool on a reaction SMILES string (mapped or unmapped)"""
        try:
            if ">>" not in rxn_smiles:
                return {"error": "Not a valid reaction SMILES"}
            
            # Check if reaction is already mapped, if not, map it
            mapped_rxn = self._map_reaction(rxn_smiles)
            
            if not mapped_rxn:
                return {"error": "Failed to map the reaction. Please check the reaction SMILES."}
            
            bond_changes = self._get_bond_changes(mapped_rxn)
            
            if "error" in bond_changes:
                return bond_changes
                
            result = {
                "mapped_reaction": mapped_rxn,
                "bonds_broken": bond_changes["bonds_broken"],
                "bonds_formed": bond_changes["bonds_formed"],
                "bonds_changed": bond_changes["bonds_changed"]
            }
            
            # If the original reaction wasn't mapped, include a note
            if mapped_rxn != rxn_smiles:
                result["note"] = "The reaction was automatically mapped for analysis using RXNMapper."
                
            return result
        except Exception as e:
            return {"error": f"Error in BondChangeAnalyzer tool: {str(e)}"}