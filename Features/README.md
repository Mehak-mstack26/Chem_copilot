# ChemCopilot Features Module

This module contains the core analysis tools and web interface for ChemCopilot.

## Components

### 1. Web Interface (`app.py`)
- Streamlit-based user interface
- Interactive reaction analysis
- Visualization tools
- Query handling

### 2. Analysis Tools (`tools/`)
- `name2smiles.py`: Chemical name to SMILES conversion
- `smiles2name.py`: SMILES to chemical name conversion
- `funcgroups.py`: Functional group analysis
- `bond.py`: Bond change analysis
- `visualizer.py`: Chemical structure visualization
- `asckos.py`: Reaction classification
- `retrosynthesis.py`: Interface with RetroSynthesis Agent
- `wrapper.py`: Tool integration handler

### 3. Core Logic (`test.py`)
- Main processing pipeline
- Tool orchestration
- Response generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:
```bash
streamlit run app.py
```

## Requirements
- Python 3.11
- Dependencies listed in requirements.txt
- Valid API keys for chemical databases
