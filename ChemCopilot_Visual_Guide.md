# ChemCopilot Visual Guide

This document provides visual explanations of ChemCopilot's key concepts and workflows to help users understand how the system works.

## Table of Contents

1. [System Overview](#system-overview)
2. [Retrosynthesis Process](#retrosynthesis-process)
3. [Retrosynthetic Tree Structure](#retrosynthetic-tree-structure)
4. [Reaction Analysis Workflow](#reaction-analysis-workflow)
5. [Web Interface Guide](#web-interface-guide)

## System Overview

ChemCopilot consists of two main components that work together to provide comprehensive chemical synthesis planning and analysis capabilities.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ChemCopilot System                          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 ▼                                   ▼
┌────────────────────────────────┐      
│    RetroSynthesisAgent         │    │    Features Repository         │
│                                │    │                                │
│  • Literature retrieval        │    │  • Web interface (Streamlit)   │
│  • Reaction extraction         │    │  • Name to SMILES conversion   │
│  • Tree construction           │    │  • SMILES to name conversion   │
│  • Entity alignment            │    │  • Functional group analysis   │
│  • Tree expansion              │    │  • Bond change analysis        │
│  • Reaction filtration         │    │  • Reaction visualization      │
│  • Pathway recommendation      │    │  • Reaction classification     │
└────────────────────────────────┘    
              │                                       │
              ▼                                       ▼
┌────────────────────────────────┐    
│      External Services         │    │       Features Tools           │
│                                │    │                                │
│  • OpenAI API                  │    │  • NameToSMILES                │
│  • CACTUS                      │    │  • SMILES2Name                 │
│  • RetroSynthesis DB           │    │  • FuncGroups                  │
│                                │    │  • BondChangeAnalyzer          │
│                                │    │  • ChemVisualizer              │
│                                │    │  • ReactionClassifier          │
└────────────────────────────────┘    
```

## Retrosynthesis Process

The retrosynthesis process in ChemCopilot follows a systematic workflow to identify synthesis pathways for target molecules.

```
┌─────────────────┐
│ Target Molecule │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Literature     │────►│  PDF Download   │
│   Search        │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  Reaction       │◄────┤  PDF Text       │
│  Extraction     │     │  Processing     │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Retrosynthetic  │
│  Tree Building  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│    Entity       │────►│     Tree        │
│   Alignment     │     │   Expansion     │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│   Reaction      │────►│    Pathway      │
│   Filtration    │     │ Recommendation  │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Recommended    │
                        │    Pathway      │
                        └─────────────────┘
```

### Step-by-Step Process Description

1. **Target Molecule Identification**:
   - User specifies the target molecule to synthesize

2. **Literature Search**:
   - System searches Google Scholar for relevant papers
   - Identifies papers discussing synthesis of the target molecule

3. **PDF Download**:
   - Downloads identified papers from Sci-Hub
   - Stores PDFs for processing

4. **PDF Text Processing**:
   - Extracts text from downloaded PDFs
   - Removes references and irrelevant sections

5. **Reaction Extraction**:
   - Uses LLM to identify chemical reactions in the text
   - Formats reactions in a standardized way

6. **Retrosynthetic Tree Building**:
   - Constructs a tree with the target molecule as the root
   - Adds reactions and reactants as nodes
   - Identifies common laboratory chemicals as leaf nodes

7. **Entity Alignment** (optional):
   - Standardizes chemical names across different papers
   - Resolves synonyms and alternative names

8. **Tree Expansion** (optional):
   - Searches for additional literature
   - Adds alternative synthesis routes

9. **Reaction Filtration** (optional):
   - Filters reactions based on practical criteria
   - Removes reactions with extreme conditions or toxic reagents

10. **Pathway Recommendation**:
    - Identifies all possible synthesis pathways
    - Recommends the optimal pathway based on criteria like:
      - Commercial availability of starting materials
      - Mildness of reaction conditions
      - Overall yield
      - Number of steps

## Retrosynthetic Tree Structure

The retrosynthetic tree represents the hierarchical relationship between chemical compounds and reactions in a synthesis pathway.

```
                      ┌───────────────┐
                      │Target Molecule│
                      └───────┬───────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
     ┌────────▼─────────┐           ┌────────▼─────────┐
     │    Reaction 1    │           │    Reaction 2    │
     └────────┬─────────┘           └────────┬─────────┘
              │                               │
    ┌─────────┴──────────┐          ┌────────┴─────────┐
    │                    │          │                  │
┌───▼───┐           ┌───▼───┐  ┌───▼───┐         ┌───▼───┐
│Reactant│          │Reactant│ │Reactant│        │Reactant│
│   A    │          │   B    │ │   C    │        │   D    │
└───┬───┘           └───┬───┘  └───┬───┘         └───┬───┘
    │                   │          │                 │
┌───▼───┐           ┌───▼───┐  ┌───▼───┐         ┌───▼───┐
│Reaction│          │Reaction│ │Reaction│        │  Leaf │
│   3    │          │   4    │ │   5    │        │ Node  │
└───┬───┘           └───┬───┘  └───┬───┘         └───────┘
    │                   │          │
┌───▼───┐           ┌───▼───┐  ┌───▼───┐
│  Leaf │           │  Leaf │  │  Leaf │
│ Node  │           │ Node  │  │ Node  │
└───────┘           └───────┘  └───────┘
```

### Tree Components

- **Root Node**: The target molecule to be synthesized
- **Internal Nodes**: Intermediate compounds in the synthesis pathway
- **Edges**: Reactions that transform one compound to another
- **Leaf Nodes**: Common laboratory chemicals that serve as starting materials

### Path Representation

A synthesis pathway is represented as a path from the root node to leaf nodes:

```
Target Molecule → Reaction 1 → Reactant A → Reaction 3 → Leaf Node
```

This path indicates that:
1. Reactant A is produced by Reaction 1 from the Target Molecule
2. Reactant A is synthesized from a common laboratory chemical via Reaction 3

## Reaction Analysis Workflow

The reaction analysis process in ChemCopilot follows a systematic workflow to provide detailed information about chemical reactions.

```
┌─────────────────┐
│ Reaction Input  │
│ (SMILES/Names)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Name to SMILES  │
│   Conversion    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reaction Type  │
│ Identification  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Bond Change   │
│    Analysis     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Functional Group │
│ Transformation  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Mechanism     │
│   Proposal      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Reaction      │────►│   Reaction      │
│  Visualization  │     │    Analysis     │
└─────────────────┘     └─────────────────┘
```

### Step-by-Step Process Description

1. **Reaction Input**:
   - User provides a reaction in SMILES notation or using chemical names

2. **Name to SMILES Conversion** (if needed):
   - Converts chemical names to SMILES notation
   - Standardizes the reaction representation

3. **Reaction Type Identification**:
   - Identifies the type of reaction (e.g., substitution, elimination, addition)
   - Classifies the reaction according to standard reaction types

4. **Bond Change Analysis**:
   - Identifies bonds formed and broken during the reaction
   - Maps atoms between reactants and products

5. **Functional Group Transformation**:
   - Identifies changes in functional groups
   - Tracks functional group transformations

6. **Mechanism Proposal**:
   - Proposes a plausible reaction mechanism
   - Identifies intermediates and transition states

7. **Reaction Visualization**:
   - Generates a visual representation of the reaction
   - Highlights bond changes and atom mapping

8. **Reaction Analysis**:
   - Provides a comprehensive analysis of the reaction
   - Includes industrial relevance, applications, and alternatives

## Web Interface Guide

The ChemCopilot web interface provides a user-friendly way to interact with the system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ChemCopilot Header                          │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────────┐  ┌─────────────────────────────────────────────┐
│                   │  │                                             │
│                   │  │                                             │
│                   │  │                                             │
│                   │  │                                             │
│     Sidebar       │  │             Main Content Area               │
│                   │  │                                             │
│  • Tools Info     │  │  ┌─────────────────────────────────────┐    │
│  • Example Queries│  │  │       Input Field                   │    │
│                   │  │  └─────────────────────────────────────┘    │
│                   │  │                                             │
│                   │  │  ┌─────────────────────────────────────┐    │
│                   │  │  │       Results Display               │    │
│                   │  │  └─────────────────────────────────────┘    │
│                   │  │                                             │
└───────────────────┘  └─────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Footer                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Retrosynthesis Search Interface

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Retrosynthesis Search                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Enter compound name: [________________________________] [Search]    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Recommended Synthesis Route                                         │
│                                                                     │
│ The recommended pathway uses reactions 1, 3, and 5 because...       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Reactions                                                           │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Step 1 (Recommended): Reaction 1                                │ │
│ │ Reaction: Compound A + Compound B → Target Compound             │ │
│ │ Conditions: Temperature: 25°C, Solvent: water                   │ │
│ │ Source: Journal of Chemistry, 2020                              │ │
│ │ [Analyze Reaction 1]                                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ Step 2: Reaction 2                                              │ │
│ │ Reaction: Compound C + Compound D → Compound A                  │ │
│ │ Conditions: Temperature: 50°C, Catalyst: acid                   │ │
│ │ Source: Journal of Organic Chemistry, 2019                      │ │
│ │ [Analyze Reaction 2]                                            │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Reaction Analysis Interface

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Reaction Analysis                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Selected Reaction                                                   │
│                                                                     │
│ CCCl.CC[O-].[Na+]>>CCOCC.[Na+].[Cl-]                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Reaction Visualization                                              │
│                                                                     │
│ [Image of reaction with atom mapping and bond changes]              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Analysis Results                                                    │
│                                                                     │
│ Reaction Type: Williamson Ether Synthesis                           │
│                                                                     │
│ Bond Changes:                                                       │
│ - C-Cl bond broken                                                  │
│ - C-O bond formed                                                   │
│                                                                     │
│ Mechanism:                                                          │
│ This reaction proceeds via an SN2 mechanism where...                │
│                                                                     │
│ Industrial Relevance:                                               │
│ This reaction is commonly used in the pharmaceutical industry...    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Ask About This Reaction                                             │
│                                                                     │
│ [_______________________________________________________] [Ask]     │
└─────────────────────────────────────────────────────────────────────┘
```

### Chemcopilot Workflow 

The overall workflow of ChemCopilot follows these steps:

```
          ┌───────────────┐
          │     Start     │
          └───────┬───────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           User Input                │
│    (IUPAC or Common Name)           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Retrosynthesis Analysis        │
│        (Multiple Pathways)          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│     Display Reaction Pathways       │
│       (With Analysis Button)        │
└─────────────────┬───────────────────┘
                  │
                  ▼
          ┌───────┴───────┐
          │   Analyze?    │
          └───────┬───────┘
       No         │         Yes
        ┌─────────┴────────┐
        │                  │
        ▼                  ▼
┌───────────────┐  ┌───────────────────┐
│ Exit/New      │  │ Full Reaction     │
│ Search        │  │ Analysis          │
└───────┬───────┘  └─────────┬─────────┘
        │                    │
        │                    ▼
        │          ┌───────────────────┐
        └──────────┤ Follow-up         │
                   │ Questions         │
                   └───────────────────┘
```

These visual representations should help users understand the structure and flow of ChemCopilot's interfaces and processes.
