# ChemCopilot 🧪

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Features and Components](#features-and-components)
4. [Installation and Setup](#installation-and-setup)
5. [Technical Details](#technical-details)
6. [Workflow Examples](#workflow-examples)
7. [Troubleshooting](#troubleshooting)
8. [Additional Documentation](#additional-documentation)

## Introduction
ChemCopilot is an advanced chemistry assistant that combines retrosynthesis pathway prediction with detailed reaction analysis. It helps chemists understand complex chemical reactions, predict synthesis routes, and analyze reaction mechanisms by integrating RetroSynthesis Agent with comprehensive reaction analysis tools.

## System Architecture

### High-Level Design (HLD)
```ascii
+------------------+     +----------------------+     +------------------+
|    User Input    | --> | RetroSynthesis Agent | --> | Analysis Engine  |
| (IUPAC/Common)   |     | (Pathway Generation) |     | (Tools Pipeline) |
+------------------+     +----------------------+     +------------------+
                                                            |
                                                            v
                                                  +------------------+
                                                  |  Visualization   |
                                                  |     Engine       |
                                                  +------------------+
```

### Low-Level Design (LLD)
```ascii
+------------------------+
|      User Input        |
+------------------------+
          |
          v
+------------------------+     +-------------------------+
|   Name2SMILES Tool     | --> |  RetroSynthesis Agent   |
+------------------------+     |  - PDF Download         |
                              |  - Reaction Extraction   |
                              |  - Tree Construction     |
                              +-------------------------+
                                          |
                                          v
+------------------------+     +-------------------------+
|   Analysis Pipeline    |     |    Reaction Data        |
| - Functional Groups    | <-- | - Reactants/Products    |
| - Bond Changes         |     | - Conditions            |
| - Mechanism Analysis   |     | - SMILES               |
+------------------------+     +-------------------------+
          |
          v
+------------------------+
|    Output Generation   |
| - Interactive UI       |
| - Visualizations       |
| - Detailed Analysis    |
+------------------------+
```

## Features and Components

### 1. RetroSynthesis Agent
- Source: [RetroSynthesis Agent](https://github.com/anujmst/RetroSynthesisAgent)
- Core Functionalities:
  - Literature-based pathway prediction
  - Reaction extraction from papers
  - Tree-based synthesis planning
  - Feasibility analysis

### 2. ChemCopilot Core
- Source: [Chem_copilot](https://github.com/Mehak-mstack26/Chem_copilot)
- Analysis Tools:
  - Name2SMILES: Chemical name conversion
  - FuncGroups: Functional group identification
  - BondAnalyzer: Reaction mechanism analysis
  - Visualizer: Structure rendering

## Technical Details

### Core Components

#### 1. Input Processing Layer
- Name validation
- SMILES conversion
- Input sanitization
- Error handling

#### 2. RetroSynthesis Integration
- API communication
- Data transformation
- Response parsing
- Error recovery

#### 3. Analysis Pipeline
- Sequential tool execution
- Data aggregation
- Result validation
- Performance optimization

#### 4. Visualization Engine
- Interactive UI components
- Real-time updates
- Data rendering
- Export capabilities

## Installation and Setup

1. Environment Setup:
```bash
conda create -n chemcopilot python=3.11
conda activate chemcopilot
```

2. Clone Repositories:
```bash
git clone https://github.com/YourUsername/ChemCopilot.git
cd ChemCopilot
git clone https://github.com/anujmst/RetroSynthesisAgent.git
```

3. RetroSynthesis Agent Setup:
- Follow setup instructions in RetroSynthesis Agent repository
- Configure environment variables
- Test installation

4. ChemCopilot Setup:
```bash
pip install -r requirements.txt
```

## Workflow Examples

### Example 1: Simple Molecule Analysis
```python
# Input: Aspirin
1. Name to SMILES conversion
2. Retrosynthesis pathway generation
3. Reaction analysis for each step
4. Visualization and output
```

### Example 2: Complex Synthesis
```python
# Input: Complex pharmaceutical
1. Multiple pathway generation
2. Feasibility analysis
3. Detailed mechanism study
4. Interactive exploration
```

## Troubleshooting

### Common Issues
1. RetroSynthesis Agent Connection
2. SMILES Conversion Errors
3. Visualization Problems
4. Performance Issues

### Solutions
- Detailed troubleshooting steps
- Configuration checks
- Error messages and fixes

## Additional Documentation
- [Features Documentation](Features/README.md)
- [API Documentation](docs/API.md)
- [Tool Documentation](docs/Tools.md)

## Requirements
- Python 3.11
- Dependencies in requirements.txt
- RetroSynthesis Agent dependencies
