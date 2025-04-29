# ChemCopilot User Guide

This guide provides detailed instructions on how to use ChemCopilot for retrosynthetic analysis and chemical reaction analysis.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Retrosynthesis Analysis](#retrosynthesis-analysis)
3. [Reaction Analysis](#reaction-analysis)
4. [Available Tools](#available-tools)
5. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### Launching ChemCopilot

1. Start the RetroSynthesis Agent service:

```bash
   # Follow the instructions from the RetroSynthesis Agent repository
```

2. Start the ChemCopilot web interface:

```bash
    conda activate retrosyn
    cd Features
    python -m streamlit run app.py
```

3. Open your web browser and navigate to http://localhost:8501

### Web Interface Overview

The ChemCopilot web interface consists of:

- Input area for compound names or reaction SMILES
- Results display for retrosynthesis pathways
- Reaction analysis view
- Follow-up question section

## Retrosynthesis Analysis

Retrosynthesis analysis helps you find synthesis pathways for target molecules.

### Performing a Retrosynthesis Search

1. Enter the target compound name in the designated field
   - Example: "flubendiamide"
   
2. Click the "Search Retrosynthesis" button

3. Wait for the system to generate the synthesis pathways
    - Download relevant scientific literature
    - Extract reactions from the papers
    - Build a retrosynthetic tree
    - Identify possible synthesis pathways
    - Recommend the optimal pathway

4. Review the results:
    - Synthesis pathway options
    - Individual reaction steps with details
    - SMILES notation for each reaction

### Understanding Retrosynthesis Results

Each reaction in the results includes:

- **Reaction Index**: Unique identifier for the reaction
- **Reactants**: Starting materials for the reaction
- **Products**: Compounds produced by the reaction
- **Conditions**: Reaction conditions (temperature, pressure, catalyst, solvent, etc.)
- **SMILES Notation**: Chemical structure representation
- **Source**: Reference to the scientific literature

## Reaction Analysis

The reaction analysis feature provides detailed information about chemical reactions.

### Analyzing a Reaction

1. Select a reaction from the retrosynthesis results by clicking on it.
   
2. Review the comprehensive analysis including:
    - Reaction Visualization: Graphical representation of the reaction
    - Functional Group Changes: Transformations of functional groups
    - Bond Transformations: Bonds broken, formed, and changed
    - Reaction Mechanism: Description of the mechanism
    - Applications: Industrial and research applications

### Asking Follow-up Questions

After analyzing a reaction, you can ask follow-up questions:

1. Enter your question in the follow-up question box
   - Example: "What are alternative conditions for this reaction?"
   - Example: "Explain the stereochemistry of this reaction"
   
2. Click the "Ask" button

3. Review the tailored answer provided by the system

## Available Tools

ChemCopilot integrates several specialized chemistry tools:

### RetroSynthesis
- Searches for synthesis pathways for a target compound
- Returns reaction steps, conditions, and SMILES notation

### FuncGroups
- Identifies functional groups in molecules or reactions
- Analyzes functional group transformations

### NameToSMILES
- Converts chemical names (IUPAC or common) to SMILES notation

### SMILES2Name
- Converts SMILES strings to chemical names
- Provides both IUPAC and common names when available

### Bond Change Analysis
- Identifies bonds broken, formed, and changed in reactions
- Explains bond transformation patterns

### ChemVisualizer
- Generates visual representations of molecules and reactions
- Creates clear reaction diagrams showing transformations

### ReactionClassifier
- Classifies reaction types based on reaction SMILES
- Provides educational information about reaction mechanisms

## Tips and Best Practices

### For Best Retrosynthesis Results

1. **Use specific compound names**: More specific names yield better results
   - Good: "2,4-dinitrophenylhydrazine"
   - Less good: "DNPH" or "dinitrophenylhydrazine"

2. **Be patient with complex molecules**: The system may take longer to generate pathways for complex structures

### For Effective Reaction Analysis

1. **Review all analysis sections**: Each section provides unique insights into the reaction
2. **Use follow-up questions for details**: Ask about specific aspects of the reaction to get targeted information
3. **Compare different reactions**: Select different reactions from the retrosynthesis results to compare approaches

### Troubleshooting Common Issues

1. **No retrosynthesis results**: Make sure the RetroSynthesis Agent service is running
2. **Visualization issues**: Ensure you have appropriate permissions to write to the filesystem
3. **OpenAI API errors**: Check your .env file contains a valid API key

