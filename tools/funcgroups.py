from rdkit import Chem
import re

class FuncGroups:
    """Tool to identify functional groups in a molecule or reaction SMILES using SMARTS patterns."""
    
    def __init__(self):
        """Initialize with SMARTS patterns for functional groups."""
        self.smarts_patterns = self.load_functional_groups()  # Fixed method call

    def load_functional_groups(self, smarts_file="open_babel.txt"):  # Added self parameter
        func_groups = {}
        with open(smarts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    name, smarts = line.split(":", 1)
                    func_groups[name.strip()] = Chem.MolFromSmarts(smarts.strip())
                except Exception as e:
                    print(f"Error parsing line: {line} — {e}")
        return func_groups
    
    def _identify_functional_groups(self, mol):
        """Identify functional groups in a molecule."""
        if not mol:
            return []
        
        groups = []
        for name, pattern in self.smarts_patterns.items():
            if mol.HasSubstructMatch(pattern):
                groups.append(name)
        
        return groups
    
    def analyze_molecule(self, smiles):
        """Analyze functional groups in a molecule SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {"error": f"Invalid SMILES: {smiles}"}
            
            groups = self._identify_functional_groups(mol)
            return {
                "smiles": smiles,
                "functional_groups": groups
            }
        except Exception as e:
            return {"error": f"Error analyzing molecule: {str(e)}"}
    
    def analyze_reaction(self, reaction_smiles):
        """Analyze functional groups in reactants and products of a reaction SMILES."""
        try:
            # Split reaction into reactants and products
            if ">>" in reaction_smiles:
                reactants_str, products_str = reaction_smiles.split(">>")
            else:
                return {"error": "Invalid reaction SMILES. Must contain '>>'."}
            
            # Split reactants and products (handling periods in SMILES)
            reactants = reactants_str.split(".")
            products = products_str.split(".")
            
            # Analyze each reactant
            reactant_results = []
            for r in reactants:
                if r.strip():  # Skip empty strings
                    result = self.analyze_molecule(r)
                    reactant_results.append(result)
            
            # Analyze each product
            product_results = []
            for p in products:
                if p.strip():  # Skip empty strings
                    result = self.analyze_molecule(p)
                    product_results.append(result)
            
            return {
                "reaction_smiles": reaction_smiles,
                "reactants": reactant_results,
                "products": product_results,
                "transformation_summary": self._summarize_transformation(reactant_results, product_results)
            }
        except Exception as e:
            return {"error": f"Error analyzing reaction: {str(e)}"}
    
    def _summarize_transformation(self, reactant_results, product_results):
        """Summarize the functional group transformations in the reaction."""
        # Collect all functional groups from reactants and products
        reactant_groups = set()
        for r in reactant_results:
            if "functional_groups" in r:
                reactant_groups.update(r["functional_groups"])
        
        product_groups = set()
        for p in product_results:
            if "functional_groups" in p:
                product_groups.update(p["functional_groups"])
        
        # Identify changes
        disappeared = reactant_groups - product_groups
        appeared = product_groups - reactant_groups
        remained = reactant_groups & product_groups
        
        summary = {
            "disappeared_groups": list(disappeared),
            "appeared_groups": list(appeared),
            "unchanged_groups": list(remained)
        }
        
        return summary
    
    def _run(self, query):
        """Run the tool with the given query."""
        # Check if query is a reaction SMILES
        if ">>" in query:
            return self.analyze_reaction(query)
        else:
            # Try to analyze as a molecule SMILES
            return self.analyze_molecule(query)
    
    def __call__(self, query):
        """Make the class callable for direct invocation."""
        return self._run(query)