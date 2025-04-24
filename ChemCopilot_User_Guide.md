# ChemCopilot User Guide

This guide provides detailed instructions on how to use ChemCopilot for retrosynthetic analysis and chemical reaction analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Retrosynthesis Analysis](#retrosynthesis-analysis)
3. [Reaction Analysis](#reaction-analysis)
4. [Advanced Features](#advanced-features)
5. [Command Line Interface](#command-line-interface)
6. [API Usage](#api-usage)
7. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### Launching ChemCopilot

1. Start the RetroSynthesisAgent API:
   ```bash
   cd RetroSynthesisAgent
   uvicorn api:app --reload --port 8000
   ```

2. Start the web interface:
   ```bash
   cd Features
   streamlit run app.py
   ```

3. Open your web browser and navigate to http://localhost:8501

### Web Interface Overview

The ChemCopilot web interface consists of:

- **Header**: Application title and description
- **Sidebar**: Tools information and example queries
- **Main Content Area**: Input fields and results display
- **Footer**: Application information

## Retrosynthesis Analysis

Retrosynthesis analysis helps you find synthesis pathways for target molecules.

### Basic Retrosynthesis Search

1. Enter the target compound name in the "Enter compound name" field
   - Example: "flubendiamide" or "2-amino-5-chloro-3-methyl benzoic acid"
   
2. Click the "Search Retrosynthesis" button

3. Wait for the system to:
   - Download relevant scientific literature
   - Extract reactions from the papers
   - Build a retrosynthetic tree
   - Identify possible synthesis pathways
   - Recommend the optimal pathway

4. Review the results:
   - **Recommended Synthesis Route**: The optimal pathway
   - **Reactions**: Individual reaction steps with details

### Understanding Retrosynthesis Results

Each reaction in the results includes:

- **Reaction Index**: Unique identifier for the reaction
- **Reactants**: Starting materials for the reaction
- **Products**: Compounds produced by the reaction
- **Conditions**: Reaction conditions (temperature, pressure, catalyst, solvent, etc.)
- **Source**: Reference to the scientific literature

Reactions marked as "Recommended" form the optimal synthesis pathway.

### Analyzing Individual Reactions

To analyze a specific reaction from the retrosynthesis results:

1. Click the "Analyze Reaction" button next to the reaction of interest
2. The system will convert the reaction to SMILES notation and perform detailed analysis
3. Review the analysis results, including:
   - Reaction type
   - Bond changes
   - Functional group transformations
   - Mechanism
   - Industrial relevance

## Reaction Analysis

The reaction analysis feature provides detailed information about chemical reactions.

### Analyzing a Reaction

1. Enter a reaction query in the input field
   - Format: "Give full information about this rxn [REACTION_SMILES]"
   - Example: "Give full information about this rxn CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
   
2. Click the "Analyze" button

3. Review the analysis results:
   - **Reaction Type**: Classification of the reaction
   - **Bond Changes**: Bonds formed and broken
   - **Functional Group Transformations**: Changes in functional groups
   - **Mechanism**: Proposed reaction mechanism
   - **Industrial Relevance**: Applications in industry
   - **Visualization**: Graphical representation of the reaction

### Using Chemical Names Instead of SMILES

If you don't have SMILES notation:

1. Enter reactants and products by name
   - Example: "Analyze the reaction between ethanol and acetic acid to form ethyl acetate and water"
   
2. The system will convert chemical names to SMILES notation and perform the analysis

### Asking Follow-up Questions

After analyzing a reaction, you can ask follow-up questions:

1. Enter your question in the "Ask About This Reaction" field
   - Example: "What is the mechanism of this reaction?"
   - Example: "How does this reaction relate to industrial processes?"
   
2. Click the "Ask Question" button

3. Review the answer provided by the system

## Advanced Features

### Chemical Name to SMILES Conversion

To convert a chemical name to SMILES notation:

1. Enter a query like "Convert [CHEMICAL_NAME] to SMILES"
   - Example: "Convert ethanol to SMILES"
   
2. The system will return the SMILES notation for the compound

### SMILES to Chemical Name Conversion

To convert SMILES notation to a chemical name:

1. Enter a query like "What is the name of [SMILES]"
   - Example: "What is the name of CCO"
   
2. The system will return the chemical name for the SMILES notation

### Functional Group Analysis

To analyze functional groups in a molecule:

1. Enter a query like "What functional groups are in [CHEMICAL_NAME or SMILES]"
   - Example: "What functional groups are in aspirin"
   - Example: "What functional groups are in CC(=O)OC1=CC=CC=C1C(=O)O"
   
2. The system will identify and list all functional groups in the molecule

### Bond Change Analysis

To analyze bond changes in a reaction:

1. Enter a query like "What bonds change in this reaction: [REACTION_SMILES]"
   - Example: "What bonds change in this reaction: CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
   
2. The system will identify bonds formed and broken during the reaction

## Command Line Interface

For advanced users, ChemCopilot can be run from the command line.

### Running Retrosynthesis Analysis

```bash
cd RetroSynthesisAgent
python main.py --material "Polyimide" --num_results 10 --alignment True --expansion True --filtration False
```

Parameters:
- `--material`: Target molecule name (required)
- `--num_results`: Number of PDF papers to download (required)
- `--alignment`: Whether to align entities (True/False, default: False)
- `--expansion`: Whether to expand the tree with additional literature (True/False, default: False)
- `--filtration`: Whether to filter reactions (True/False, default: False)

### Visualizing the Retrosynthesis Tree

```bash
cd RetroSynthesisAgent
python vistree.py
```

This will start a web server for visualizing the retrosynthesis tree. Open your browser and navigate to http://localhost:8000 to view the tree.

## API Usage

ChemCopilot provides a REST API for integration with other applications.

### Retrosynthesis API

**Endpoint**: `/retro-synthesis/`
**Method**: POST
**Payload**:
```json
{
  "material": "flubendiamide",
  "num_results": 10,
  "alignment": true,
  "expansion": true,
  "filtration": false
}
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "reactions": [
      {
        "idx": "1",
        "reactants": ["compound A", "compound B"],
        "products": ["target compound"],
        "conditions": "temperature: 25°C, solvent: water",
        "source": "Journal of Chemistry, 2020"
      },
      ...
    ],
    "recommended_indices": ["1", "3", "5"],
    "reasoning": "This pathway is recommended because..."
  }
}
```

### Example API Call with Python

```python
import requests

url = "http://localhost:8000/retro-synthesis/"
payload = {
  "material": "flubendiamide",
  "num_results": 10,
  "alignment": True,
  "expansion": True,
  "filtration": False
}

response = requests.post(url, json=payload)
data = response.json()

if data["status"] == "success":
    print(f"Found {len(data['data']['reactions'])} reactions")
    print(f"Recommended pathway: {data['data']['recommended_indices']}")
    print(f"Reasoning: {data['data']['reasoning']}")
else:
    print(f"Error: {data.get('message', 'Unknown error')}")
```

## Tips and Best Practices

### Optimizing Retrosynthesis Results

1. **Use specific compound names**: More specific names yield better results
   - Good: "2,4-dinitrophenylhydrazine"
   - Less good: "DNPH" or "dinitrophenylhydrazine"

2. **Adjust parameters based on complexity**:
   - For simple molecules: `--num_results 5 --expansion False`
   - For complex molecules: `--num_results 10 --expansion True`

3. **Use entity alignment for consistent results**:
   - Always set `--alignment True` unless you have a specific reason not to

### Improving Reaction Analysis

1. **Provide complete SMILES notation** including all reactants, products, and spectator ions
   - Good: "CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"
   - Less good: "CCCl.CCO>>CCOCC"

2. **Ask specific questions** for more detailed analysis
   - Good: "What is the mechanism of the Williamson ether synthesis reaction CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]?"
   - Less good: "Tell me about this reaction CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]"

3. **Use follow-up questions** to explore specific aspects of a reaction

### Handling Large Molecules

For large or complex molecules:

1. Break down the synthesis into smaller parts
2. Analyze each part separately
3. Combine the results to form a complete synthesis pathway

### Troubleshooting Common Issues

1. **No results found**: Try alternative names for the target compound or increase `--num_results`

2. **Incorrect reaction analysis**: Verify SMILES notation is correct and complete

3. **Slow performance**: Reduce `--num_results` or disable `--expansion` for faster results

4. **PDF download failures**: Check internet connection and Sci-Hub access
