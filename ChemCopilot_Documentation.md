# ChemCopilot: Comprehensive Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
   - [RetroSynthesisAgent](#retrosynthesisagent)
   - [Features Module](#features-module)
4. [Installation Guide](#installation-guide)
5. [Usage Guide](#usage-guide)
   - [Command Line Interface](#command-line-interface)
   - [Web Interface](#web-interface)
   - [API](#api)
6. [Code Flow and Processes](#code-flow-and-processes)
7. [Visual Representations](#visual-representations)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

## Introduction

ChemCopilot is an advanced computational chemistry tool designed to assist chemists, researchers, and students in analyzing chemical reactions and planning synthetic routes. The system combines retrosynthetic analysis capabilities with chemical reaction analysis tools to provide a comprehensive platform for chemical synthesis planning.

The platform consists of two main modules:
1. **RetroSynthesisAgent**: A powerful tool for retrosynthetic analysis that can suggest synthesis pathways for target molecules
2. **Features**: A collection of chemistry tools for analyzing reactions, converting between chemical notations, and visualizing chemical structures

## System Architecture

ChemCopilot follows a modular architecture with several interconnected components:

```
ChemCopilot
├── RetroSynthesisAgent/       # Retrosynthetic analysis module
│   ├── main.py                # Main entry point for retrosynthesis
│   ├── api.py                 # FastAPI interface for retrosynthesis
│   ├── vistree.py             # Tree visualization web interface
│   └── RetroSynAgent/         # Core retrosynthesis components
│       ├── GPTAPI.py          # OpenAI API integration
│       ├── pdfDownloader.py   # Scientific literature retrieval
│       ├── pdfProcessor.py    # PDF text extraction and processing
│       ├── treeBuilder.py     # Synthesis tree construction
│       ├── entityAlignment.py # Chemical entity standardization
│       ├── treeExpansion.py   # Tree expansion with additional literature
│       ├── reactionsFiltration.py # Filtering reactions by criteria
│       └── prompts.py         # LLM prompts for various tasks
│
├── Features/                  # Chemistry tools and web interface
│   ├── app.py                 # Streamlit web interface
│   └── tools/                 # Chemistry tool implementations
│       ├── name2smiles.py     # Convert chemical names to SMILES
│       ├── smiles2name.py     # Convert SMILES to chemical names
│       ├── funcgroups.py      # Functional group analysis
│       ├── bond.py            # Bond change analysis
│       ├── retrosynthesis.py  # Interface to RetroSynthesisAgent
│       ├── visualizer.py      # Chemical structure visualization
│       └── websearch.py       # Web search for chemical information
```

## Core Components

### RetroSynthesisAgent

The RetroSynthesisAgent is the heart of ChemCopilot's synthesis planning capabilities. It performs retrosynthetic analysis to suggest possible synthesis routes for a target molecule.

#### Key Components:

1. **PDF Downloader (`pdfDownloader.py`)**: 
   - Downloads scientific literature related to the target molecule from sources like Sci-Hub
   - Uses Google Scholar to find relevant papers
   - Manages parallel downloads with threading

2. **PDF Processor (`pdfProcessor.py`)**: 
   - Extracts text from downloaded PDFs
   - Uses LLMs to identify chemical reactions from the text
   - Formats reactions in a standardized way

3. **Tree Builder (`treeBuilder.py`)**: 
   - Constructs a retrosynthetic tree from extracted reactions
   - Identifies common laboratory chemicals as leaf nodes
   - Manages tree traversal and path finding

4. **Entity Alignment (`entityAlignment.py`)**: 
   - Standardizes chemical names across different papers
   - Resolves synonyms and alternative names for the same compound

5. **Tree Expansion (`treeExpansion.py`)**: 
   - Expands the synthesis tree with additional literature
   - Adds alternative synthesis routes

6. **Reactions Filtration (`reactionsFiltration.py`)**: 
   - Filters reactions based on practical criteria
   - Removes reactions with extreme conditions or toxic reagents

7. **GPT API Integration (`GPTAPI.py`)**: 
   - Interfaces with OpenAI's API
   - Processes text and images from PDFs
   - Extracts structured reaction data

8. **Tree Visualization (`vistree.py`)**: 
   - Web interface for visualizing synthesis trees
   - Uses FastAPI and JavaScript for interactive visualization

### Features Module

The Features module provides a user-friendly web interface and various chemistry tools:

1. **Web Interface (`app.py`)**: 
   - Streamlit-based web application
   - Integrates all chemistry tools
   - Provides reaction analysis and retrosynthesis interfaces

2. **Chemistry Tools**:
   - **name2smiles.py**: Converts chemical names to SMILES notation
   - **smiles2name.py**: Converts SMILES notation to chemical names
   - **funcgroups.py**: Analyzes functional groups in molecules
   - **bond.py**: Analyzes bond changes in chemical reactions
   - **visualizer.py**: Generates visual representations of molecules and reactions
   - **retrosynthesis.py**: Interface to the RetroSynthesisAgent

## Installation Guide

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Sci-Hub access (for literature retrieval)

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/chem_copilot.git
   cd chem_copilot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root with the following variables:
   ```
   API_KEY=your_openai_api_key
   BASE_URL=https://api.openai.com/v1  # Optional if using OpenAI directly
   HEADERS={"User-Agent": "Mozilla/5.0..."}  # For Sci-Hub access
   COOKIES={"session": "..."}  # For Sci-Hub access
   ```

## Usage Guide

### Command Line Interface

The RetroSynthesisAgent can be run from the command line:

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

### Web Interface

The web interface provides a user-friendly way to interact with ChemCopilot:

1. **Start the RetroSynthesisAgent API**:
   ```bash
   cd RetroSynthesisAgent
   uvicorn api:app --reload
   ```

2. **Start the Streamlit web interface**:
   ```bash
   cd Features
   streamlit run app.py
   ```

3. **Access the web interface** at http://localhost:8501

### API

The RetroSynthesisAgent provides a REST API:

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
    "reactions": [...],
    "recommended_indices": [...],
    "reasoning": "..."
  }
}
```

## Code Flow and Processes

### RetroSynthesis Process Flow

1. **Literature Retrieval**:
   - User specifies a target molecule
   - System searches Google Scholar for relevant papers
   - PDFs are downloaded from Sci-Hub

2. **Reaction Extraction**:
   - PDFs are processed to extract text
   - LLM (GPT-4) extracts chemical reactions from the text
   - Reactions are standardized and formatted

3. **Tree Construction**:
   - A retrosynthetic tree is built with the target molecule as the root
   - Each node represents a chemical compound
   - Each edge represents a reaction
   - Leaf nodes are common laboratory chemicals

4. **Entity Alignment** (optional):
   - Chemical names are standardized across different papers
   - Synonyms and alternative names are resolved

5. **Tree Expansion** (optional):
   - Additional literature is searched for alternative synthesis routes
   - The tree is expanded with new reactions

6. **Reaction Filtration** (optional):
   - Reactions are filtered based on practical criteria
   - Reactions with extreme conditions or toxic reagents are removed

7. **Pathway Recommendation**:
   - All possible synthesis pathways are identified
   - The most promising pathway is recommended based on criteria like:
     - Commercial availability of starting materials
     - Mildness of reaction conditions
     - Overall yield
     - Number of steps

8. **Visualization**:
   - The synthesis tree is visualized for user exploration

### Reaction Analysis Process Flow

1. **Input Processing**:
   - User provides a reaction SMILES or chemical names
   - System converts names to SMILES if necessary

2. **Analysis**:
   - Reaction is analyzed for:
     - Reaction type
     - Bond changes
     - Functional group transformations
     - Mechanism
     - Industrial relevance

3. **Visualization**:
   - Reaction is visualized showing reactants, products, and bond changes

## Visual Representations

### Retrosynthesis Tree Visualization

The system generates interactive visualizations of retrosynthesis trees:

```
                      Target Molecule
                            |
                 -------------------------
                 |           |           |
            Reaction 1   Reaction 2   Reaction 3
                 |           |           |
          ------------   ---------   ------------
          |    |     |   |   |   |   |    |     |
      Compound Compound Compound Compound Compound
```

The visualization allows users to:
- Explore different synthesis pathways
- Compare trees before and after expansion
- Identify recommended pathways
- View reaction details by clicking on nodes

### Reaction Visualization

For individual reactions, the system generates visualizations showing:
- Reactant and product structures
- Bond changes (formed/broken)
- Atom mapping

## Troubleshooting

### Common Issues and Solutions

1. **PDF Download Failures**:
   - Check Sci-Hub access and credentials
   - Verify internet connection
   - Try different search terms

2. **Entity Alignment Issues**:
   - Check for unusual chemical naming conventions
   - Manually standardize key compound names

3. **Tree Construction Failures**:
   - Verify that reactions are properly formatted
   - Check for disconnected reactions (products not used as reactants)

4. **API Connection Issues**:
   - Verify OpenAI API key
   - Check API rate limits
   - Ensure proper network connectivity

## References

- OpenAI API Documentation: https://platform.openai.com/docs/
- RDKit Documentation: https://www.rdkit.org/docs/
- Streamlit Documentation: https://docs.streamlit.io/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- PyMuPDF Documentation: https://pymupdf.readthedocs.io/
