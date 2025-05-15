# ChemCopilot - Chemistry Assistant

ChemCopilot is an advanced chemistry assistant designed for analyzing chemical reactions and performing retrosynthesis analysis. This tool provides chemists with comprehensive information about reactions, including bond changes, functional group transformations, visualization, and more.

## 📋 Prerequisites

- Python 3.11
- Conda (recommended for environment management)
- OpenAI API key
- Git

## ⚙️ Installation

### Step 1: Clone the ChemCopilot Repository
```bash
git clone https://github.com/yourusername/chemcopilot.git
cd chemcopilot
```

### Step 2: Create and Activate a Conda Environment
```bash
conda create -n retrosyn python=3.11
conda activate retrosyn
```

### Step 3: Configure the Features Repository
Create a `.env` file in the main directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

1. In a new terminal, start the ChemCopilot web interface:
```bash
conda activate retrosyn
python -m streamlit run app.py
```

2. Open your browser and navigate to http://localhost:8501

## 📊 Tools

ChemCopilot integrates several specialized chemistry tools:

### FuncGroups
- Identifies functional groups in molecules or reactions
- Analyzes functional group transformations

### NameToSMILES
- Converts chemical names (IUPAC or common) to SMILES notation

### SMILES2Name
- Converts SMILES strings to chemical names
- Provides both IUPAC and common names when available

### BondChangeAnalyzer
- Identifies bonds broken, formed, and changed in reactions
- Explains bond transformation patterns

### ChemVisualizer
- Generates visual representations of molecules and reactions
- Creates clear reaction diagrams showing transformations

### ReactionClassifier
- Classifies reaction types based on reaction SMILES
- Provides educational information about reaction mechanisms

## 🔄 Data Flow

1. User Input → Query Processing
2. Compound Name → RetroSynthesis API → Pathway Results
3. Reaction SMILES → Chemistry Tools → Analysis Results
4. Results → Visualization → User Interface

### Reaction Analysis
1. Select a reaction from the retrosynthesis results
2. View comprehensive analysis including:
   - Reaction visualization
   - Functional group changes
   - Bond transformations
   - Reaction mechanism
   - Applications

### Follow-up Questions
1. After viewing a reaction analysis, use the question box
2. Ask specific questions about the reaction
3. Get tailored answers about mechanisms, applications, etc.

## 📝 Important Notes

- Both components require Python 3.11
- A valid OpenAI API key is required for the GPT-4o powered analysis tools

<!--
Note: To properly display the diagrams, please save the following images:
1. Save the system architecture diagram as "images/system_architecture.png"
2. Save the workflow diagram as "images/workflow.png"
-->
