# ChemCopilot Technical Reference

This technical reference provides detailed information about the internal workings of ChemCopilot, including code structure, algorithms, data flows, and integration points.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Components Overview](#components-overview)
3. [RetroSynthesisAgent Module](#retrosynthesisagent-module)
4. [Features Module](#features-module)
5. [Data Structures](#data-structures)
6. [Algorithms](#algorithms)
7. [API Reference](#api-reference)
8. [Configuration Options](#configuration-options)
9. [Integration Points](#integration-points)
10. [Performance Considerations](#performance-considerations)

## System Architecture

ChemCopilot follows a modular architecture with two primary components:

1. **RetroSynthesisAgent**: Handles retrosynthetic analysis
2. **Features Repository**: Provides chemistry tools and web interface

These components communicate through well-defined interfaces, allowing them to be used independently or together.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────┐
│           Streamlit UI              │
│           (app.py)                  │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────┐      ┌─────────────────────┐
│  Retrosynthesis     │─────▶│ Features Repository │
│  Agent              │      │                     │
└─────────┬───────────┘      └─────────┬───────────┘
          │                            │
          ▼                            ▼
┌─────────────────────┐      ┌─────────────────────┐
│ External Services   │      │   Features Tools    │
│ - OpenAI API        │      │ - NameToSMILES     │
│ - CACTUS           │      │ - SMILES2Name      │
│ - RetroSynthesis DB │      │ - FuncGroups       │
└─────────────────────┘      │ - BondChangeAnalyzer│
                             │ - ChemVisualizer   │
                             │ - ReactionClassifier│
                             └─────────────────────┘
```

## Components Overview

ChemCopilot consists of two main components:

### RetroSynthesisAgent
The RetroSynthesisAgent is responsible for generating synthetic pathways for target compounds. It:

- Downloads relevant scientific literature
- Extracts chemical reactions from papers
- Constructs retrosynthetic trees
- Recommends optimal synthesis pathways

### Features Repository
The Features Repository provides chemistry analysis tools and a web interface. It:

- Offers a Streamlit-based user interface
- Provides various chemistry tools (functional group analysis, name-to-SMILES conversion, etc.)
- Integrates with the RetroSynthesisAgent
- Visualizes reaction pathways and molecular structures

## RetroSynthesisAgent Module

### Main Entry Point (`main.py`)

The `main.py` file serves as the entry point for the RetroSynthesisAgent. It orchestrates the entire retrosynthesis process:

1. Parses command-line arguments
2. Downloads relevant PDFs
3. Processes PDFs to extract reactions
4. Constructs the retrosynthetic tree
5. Performs entity alignment (optional)
6. Expands the tree with additional literature (optional)
7. Filters reactions (optional)
8. Recommends optimal synthesis pathways

Key functions:
- `parse_arguments()`: Processes command-line arguments
- `main()`: Main execution flow
- `parse_reaction_data()`: Parses reaction data from text
- `recommendReactions()`: Uses LLM to recommend optimal pathways

### PDF Downloader (`pdfDownloader.py`)

The `PDFDownloader` class handles downloading scientific literature:

- Uses Google Scholar to find relevant papers
- Downloads PDFs from Sci-Hub
- Manages parallel downloads with threading
- Handles retries and error cases

Key methods:
- `get_scholar_titles()`: Searches Google Scholar for relevant papers
- `title_href()`: Gets download links from Sci-Hub
- `get_download_pdf()`: Downloads PDFs
- `download_pdfs()`: Manages parallel downloads
- `main()`: Orchestrates the download process

### PDF Processor (`pdfProcessor.py`)

The `PDFProcessor` class extracts and processes text from PDFs:

- Converts PDFs to text
- Uses LLM to extract chemical reactions
- Formats reactions in a standardized way
- Manages batch processing

Key methods:
- `pdf_to_long_string()`: Extracts text from PDFs
- `process_pdfs_txt()`: Processes PDFs to extract reactions
- `remove_references_section()`: Cleans up extracted text

### Tree Builder (`treeBuilder.py`)

The `Tree` and `Node` classes handle the construction and management of the retrosynthetic tree:

- `Node`: Represents a chemical compound in the tree
- `Tree`: Manages the overall tree structure
- `CommonSubstanceDB`: Identifies common laboratory chemicals

Key methods:
- `construct_tree()`: Builds the retrosynthetic tree
- `expand()`: Expands a node in the tree
- `find_all_paths()`: Finds all possible synthesis pathways
- `get_node_count()`: Counts nodes in the tree
- `get_reactions_in_tree()`: Gets all reactions in the tree

### Entity Alignment (`entityAlignment.py`)

The `EntityAlignment` class standardizes chemical names:

- Resolves synonyms and alternative names
- Ensures consistent naming throughout the tree
- Aligns the root node with the target compound

Key methods:
- `alignRootNode()`: Aligns the root node with the target compound
- `entityAlignment_1()`: First pass of entity alignment
- `entityAlignment_2()`: Second pass of entity alignment

### Tree Expansion (`treeExpansion.py`)

The `TreeExpansion` class expands the synthesis tree:

- Searches for additional literature
- Adds alternative synthesis routes
- Updates the tree with new reactions

Key methods:
- `treeExpansion()`: Expands the tree with additional literature
- `update_dict()`: Updates the reaction dictionary

### Reactions Filtration (`reactionsFiltration.py`)

The `ReactionsFiltration` class filters reactions:

- Removes reactions with extreme conditions
- Filters out reactions with toxic reagents
- Ensures practical synthesis pathways

Key methods:
- `filterReactions()`: Filters reactions based on conditions
- `filterPathways()`: Filters pathways based on validity
- `getFullReactionPathways()`: Gets all reaction pathways

### GPT API Integration (`GPTAPI.py`)

The `GPTAPI` class interfaces with OpenAI's API:

- Handles text-based queries
- Processes images from PDFs
- Manages API requests and responses

Key methods:
- `answer_wo_vision()`: Gets answers without vision capabilities
- `answer_w_vision_img_list_txt()`: Gets answers with vision capabilities

### Prompts (`prompts.py`)

Contains prompt templates for various LLM tasks:

- Reaction extraction
- Entity alignment
- Pathway recommendation
- Reaction evaluation

## Features Module

### Web Interface (`app.py`)

The Streamlit-based web application:

- Provides user interface for all features
- Integrates retrosynthesis and reaction analysis
- Handles user input and displays results

Key sections:
- Retrosynthesis search
- Reaction analysis
- Follow-up questions
- Visualization display

### Chemistry Tools

#### Name to SMILES (`name2smiles.py`)

Converts chemical names to SMILES notation:

- Uses LLM and chemical databases
- Handles complex chemical nomenclature
- Returns standardized SMILES

#### SMILES to Name (`smiles2name.py`)

Converts SMILES notation to chemical names:

- Generates IUPAC and common names
- Handles complex molecular structures
- Provides human-readable names

#### Functional Groups (`funcgroups.py`)

Analyzes functional groups in molecules:

- Identifies common functional groups
- Categorizes by type and properties
- Provides detailed descriptions

#### Bond Analysis (`bond.py`)

Analyzes bond changes in chemical reactions:

- Identifies bonds formed and broken
- Maps atoms between reactants and products
- Visualizes bond changes

#### Visualizer (`visualizer.py`)

Generates visual representations:

- Creates 2D molecular structures
- Visualizes reactions with atom mapping
- Highlights bond changes

#### ReactionClassifier (`reactionClassifier.py`)

Classifies reaction types:

- Identifies reaction mechanisms
- Categorizes by reaction class
- Provides educational information about reactions

#### Query Processing (`test.py`)

Contains the enhanced_query function that coordinates the analysis pipeline:

- Processes user queries about reactions
- Calls appropriate chemistry tools based on query intent
- Generates comprehensive reaction analyses
- Handles visualization requests

#### Retrosynthesis Interface (`retrosynthesis.py`)

Interfaces with the RetroSynthesisAgent:

- Processes and displays results
- Handles error cases

## Data Structures

### Reaction Dictionary

```python
{
    "idx": "1",
    "reactants": ["compound A", "compound B"],
    "products": ["target compound"],
    "conditions": {
        "temperature": "25°C",
        "solvent": "water",
        "catalyst": "acid"
    },
    "source": "Journal of Chemistry, 2020",
    "source_link": "https://doi.org/..."
}
```

### Node Structure

```python
class Node:
    def __init__(self, substance, reactions, product_dict,
                 fathers_set=None, father=None, reaction_index=None,
                 reaction_line=None, cache_func=None, unexpandable_substances=None):
        self.reaction_index = reaction_index  # Index of the reaction that produced this node
        self.substance = substance  # Chemical name
        self.children = []  # Child nodes (reactants)
        self.fathers_set = fathers_set  # Set of ancestor nodes
        self.father = father  # Parent node
        self.reaction_line = reaction_line  # Path of reactions from root
        self.is_leaf = False  # Whether this is a leaf node
        self.cache_func = cache_func  # Function to check if substance is common
        self.reactions = reactions  # All available reactions
        self.product_dict = product_dict  # Dictionary mapping products to reactions
        self.unexpandable_substances = unexpandable_substances  # Set of substances that cannot be expanded
```

### Tree Structure

```python
class Tree:
    def __init__(self, target_substance, result_dict=None, reactions_txt=None, reactions=None):
        self.reactions = reactions  # Dictionary of all reactions
        self.product_dict = self.get_product_dict(self.reactions)  # Maps products to reactions
        self.target_substance = target_substance  # Target molecule
        self.reaction_infos = set()  # Set of reaction information
        self.all_path = []  # All possible synthesis pathways
        self.db = CommonSubstanceDB()  # Database of common substances
        self.unexpandable_substances = set()  # Substances that cannot be expanded
        self.root = Node(target_substance, self.reactions, self.product_dict,
                         cache_func=self.db.is_common_chemical_cached,
                         unexpandable_substances=self.unexpandable_substances)
```

## Algorithms

### Retrosynthetic Tree Construction

The retrosynthetic tree is constructed using a recursive algorithm:

1. Start with the target molecule as the root node
2. For each node:
   a. Check if it's a common laboratory chemical (leaf node)
   b. If not, find reactions that produce this compound
   c. For each reaction, create child nodes for the reactants
   d. Recursively expand each child node
   e. Remove cycles (where a compound appears in its own synthesis path)
3. Continue until all branches end in common chemicals or cannot be expanded further

```python
def expand(self) -> bool:
    # Check if this is a common chemical (leaf node)
    if self.cache_func(self.substance):
        self.is_leaf = True
        return True
    
    # Find reactions that produce this compound
    reactions_idxs = self.product_dict.get(self.substance, [])
    
    # If no reactions found, this compound cannot be expanded
    if len(reactions_idxs) == 0:
        self.unexpandable_substances.add(self.substance)
        return False
    
    # For each reaction, create child nodes for the reactants
    for reaction_idx in reactions_idxs:
        reactants_list = self.reactions[reaction_idx]['reactants']
        
        # Add each reactant as a child node
        for reactant in reactants_list:
            child = self.add_child(reactant, reaction_idx)
            
            # Check for cycles
            if child.substance in child.fathers_set:
                self.remove_child_by_reaction(reaction_idx)
                break
            
            # Recursively expand the child node
            is_valid = child.expand()
            
            # If the child cannot be expanded, mark it as invalid
            if not is_valid:
                child.is_leaf = False
                continue
    
    # If no valid children, this node cannot be expanded
    if len(self.children) == 0:
        return False
    
    # Otherwise, this node can be expanded
    return True
```

### Path Finding Algorithm

To find all possible synthesis pathways:

1. Start at the root node
2. For each child node:
   a. Recursively find all paths from the child to leaf nodes
   b. Combine the reaction index with each path
3. Deduplicate paths and remove supersets

```python
def search_reaction_pathways(self, node):
    # If it's a leaf node, return an empty path
    if node.is_leaf:
        return [[]]
    
    # Store paths for each reaction
    reaction_paths = {}
    
    for child in node.children:
        # Recursively get paths from child nodes
        paths = self.search_reaction_pathways(child)
        reaction_idx = child.reaction_index
        
        # Store or combine paths for this reaction
        if reaction_idx not in reaction_paths or reaction_paths[reaction_idx] == [[]]:
            reaction_paths[reaction_idx] = paths
        elif paths:
            # Combine existing paths with new paths
            combined_paths = []
            for prev_path in reaction_paths[reaction_idx]:
                for curr_path in paths:
                    combined_paths.append(prev_path + curr_path)
            reaction_paths[reaction_idx] = combined_paths
    
    # Aggregate all paths
    pathways = []
    for reaction_idx, paths in reaction_paths.items():
        for path in paths:
            pathways.append([reaction_idx] + path)
    
    return pathways
```

### Entity Alignment Algorithm

The entity alignment process uses a two-pass approach:

1. First pass: Align entities based on exact matches and simple variations
2. Second pass: Use LLM to identify and standardize synonyms

```python
def entityAlignment_1(self, reactions_dict):
    # First pass: Align based on exact matches and simple variations
    standardized_dict = {}
    for idx, reaction in reactions_dict.items():
        # Process reactants and products
        standardized_reactants = []
        for reactant in reaction['reactants']:
            # Apply standardization rules
            standardized_reactant = self.standardize_name(reactant)
            standardized_reactants.append(standardized_reactant)
        
        # Similar process for products
        # ...
        
        # Update the reaction with standardized names
        standardized_dict[idx] = {
            'reactants': tuple(standardized_reactants),
            'products': tuple(standardized_products),
            'conditions': reaction['conditions'],
            'source': reaction['source']
        }
    
    return standardized_dict

def entityAlignment_2(self, reactions_dict):
    # Second pass: Use LLM to identify and standardize synonyms
    # Extract all unique substances
    all_substances = set()
    for reaction in reactions_dict.values():
        all_substances.update(reaction['reactants'])
        all_substances.update(reaction['products'])
    
    # Use LLM to identify synonyms
    synonyms = self.identify_synonyms(all_substances)
    
    # Apply synonym standardization
    standardized_dict = {}
    for idx, reaction in reactions_dict.items():
        # Apply synonym standardization to reactants and products
        # ...
        
        # Update the reaction with standardized names
        standardized_dict[idx] = {
            'reactants': tuple(standardized_reactants),
            'products': tuple(standardized_products),
            'conditions': reaction['conditions'],
            'source': reaction['source']
        }
    
    return standardized_dict
```

## API Reference

### RetroSynthesisAgent API

#### Endpoint: `/retro-synthesis/`

**Method**: POST

**Request Body**:
```json
{
  "material": "string",
  "num_results": "integer",
  "alignment": "boolean",
  "expansion": "boolean",
  "filtration": "boolean"
}
```

**Response Body**:
```json
{
  "status": "string",
  "data": {
    "reactions": [
      {
        "idx": "string",
        "reactants": ["string"],
        "products": ["string"],
        "conditions": "string",
        "source": "string",
        "source_link": "string"
      }
    ],
    "recommended_indices": ["string"],
    "reasoning": "string"
  }
}
```

**Status Codes**:
- 200: Success
- 400: Bad Request
- 500: Internal Server Error

#### Endpoint: `/`

**Method**: GET

**Response Body**:
```json
{
  "message": "RetroSynthesisAgent API is running."
}
```

### Tree Visualization API

#### Endpoint: `/api/double`

**Method**: GET

**Response Body**:
```json
{
  "bigTree": {
    "name": "string",
    "children": [
      {
        "name": "string",
        "children": [],
        "is_leaf": "boolean"
      }
    ],
    "is_leaf": "boolean"
  },
  "smallTree": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  }
}
```

#### Endpoint: `/api/three`

**Method**: GET

**Response Body**:
```json
{
  "main": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  },
  "son": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  },
  "path1": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  }
}
```

#### Endpoint: `/api/quad`

**Method**: GET

**Response Body**:
```json
{
  "main": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  },
  "son": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  },
  "path1": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  },
  "path2": {
    "name": "string",
    "children": [],
    "is_leaf": "boolean"
  }
}
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| API_KEY | OpenAI API key | None |
| BASE_URL | OpenAI API base URL | https://api.openai.com/v1 |
| HEADERS | Headers for Sci-Hub access | None |
| COOKIES | Cookies for Sci-Hub access | None |

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| --material | Target molecule name | Required |
| --num_results | Number of PDF papers to download | Required |
| --alignment | Whether to align entities | False |
| --expansion | Whether to expand the tree | False |
| --filtration | Whether to filter reactions | False |

### Folder Structure

| Folder | Description |
|--------|-------------|
| pdf_pi | Downloaded PDFs |
| res_pi | Extracted reactions |
| tree_pi | Saved trees |

## Integration Points

### Integrating with External Chemical Databases

ChemCopilot can be integrated with external chemical databases:

1. Modify `CommonSubstanceDB` in `treeBuilder.py` to query additional databases
2. Add API keys and endpoints to the `.env` file
3. Implement database-specific query functions

### Extending with Custom Chemistry Tools

To add a new chemistry tool:

1. Create a new Python file in the `Features/tools` directory
2. Implement the tool as a class with a `_run` method
3. Add the tool to the `app.py` file
4. Update the sidebar in `app.py` to include the new tool

### Integrating with Laboratory Automation Systems

ChemCopilot can be integrated with laboratory automation systems:

1. Use the RetroSynthesisAgent API to get synthesis pathways
2. Convert the pathways to machine-readable instructions
3. Send the instructions to the laboratory automation system

## Performance Considerations

### Optimizing PDF Processing

- Use batch processing to handle large numbers of PDFs
- Implement caching to avoid reprocessing the same PDFs
- Use parallel processing for PDF text extraction

### Optimizing Tree Construction

- Implement memoization for common subproblems
- Use pruning to remove unlikely synthesis pathways early
- Implement lazy evaluation for tree expansion

### Optimizing API Requests

- Implement request batching to reduce API calls
- Use caching to avoid redundant API calls
- Implement rate limiting to avoid API throttling

### Memory Management

- Use generators for large data structures
- Implement pagination for large result sets
- Use streaming responses for large API responses


