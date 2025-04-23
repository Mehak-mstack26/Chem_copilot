# ChemCopilot 🧪

ChemCopilot is an advanced chemistry assistant that combines retrosynthesis pathway prediction with detailed reaction analysis. It helps chemists understand complex chemical reactions, predict synthesis routes, and analyze reaction mechanisms.

## Features and Components

### 1. RetroSynthesis Agent
- Predicts retrosynthesis pathways for target molecules
- Provides multiple synthesis routes with detailed reaction steps
- Source: [RetroSynthesis Agent](https://github.com/anujmst/RetroSynthesisAgent)

### 2. ChemCopilot Core
- Comprehensive reaction analysis tools
- Chemical name to SMILES conversion
- Functional group identification
- Bond change analysis
- Source: [Chem_copilot](https://github.com/Mehak-mstack26/Chem_copilot)

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/YourUsername/ChemCopilot.git
cd ChemCopilot
```

2. Add RetroSynthesis Agent as a subfolder:
```bash
git clone https://github.com/anujmst/RetroSynthesisAgent.git
```

3. Follow RetroSynthesis Agent setup instructions in their repository to configure the environment and dependencies.

4. Install ChemCopilot dependencies:
```bash
pip install -r requirements.txt
```

Note: ChemCopilot requires Python 3.11

## Architecture and Workflow

1. **Input Processing**
   - User provides target molecule (IUPAC or common name)
   - System validates and processes the input

2. **Retrosynthesis Prediction**
   - RetroSynthesis Agent analyzes the target molecule
   - Generates 3 possible synthesis pathways
   - Each pathway includes:
     - Reaction steps
     - Reactants and products
     - Reaction conditions
     - Reaction SMILES
     - Literature sources

3. **Reaction Analysis**
   - Extracts reaction SMILES from each pathway
   - For each reaction:
     - Identifies functional groups
     - Analyzes bond changes
     - Provides mechanistic insights
     - Generates comprehensive reaction information

4. **Output Generation**
   - Presents synthesis pathways with detailed analysis
   - Offers interactive exploration of reactions
   - Provides visualization options

For detailed information about the Features module, please refer to [Features/README.md](Features/README.md).
