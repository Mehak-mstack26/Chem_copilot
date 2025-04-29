# ChemCopilot Installation Guide

This guide provides detailed instructions for installing and setting up ChemCopilot on your system.

## Prerequisites

Before installing ChemCopilot, ensure you have the following prerequisites:

- **Python**: Version 3.11
- **Git**: For cloning the repository
- **Virtual Environment**: Conda (recommended for environment management)
- **OpenAI API Key**: Required for LLM-based analysis
- **Sufficient Disk Space**: At least 2GB for the application and dependencies

## Step 1: Clone the ChemCopilot Repository

```bash
git clone https://github.com/yourusername/chem_copilot.git
cd chem_copilot
```

## Step 2: Create and Activate a Conda Environment

```bash
conda create -n retrosyn python=3.11
conda activate retrosyn
```

## Step 3: Clone the RetroSynthesis Agent Repository

```bash
# Inside the chemcopilot directory
git clone https://github.com/anujmst/RetroSynthesisAgent.git
```


## Step 4: Set Up RetroSynthesis Agent

Follow the instructions at: https://github.com/anujmst/RetroSynthesisAgent

## Step 5: Configure the Features Repository

Create a .env file in the Features directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Application

1. Start the RetroSynthesis Agent service (follow the instructions from the RetroSynthesis Agent repository)

2. In a new terminal, start the ChemCopilot web interface:

```bash
conda activate retrosyn
cd chemcopilot
python -m streamlit run app.py
```

3. Open your browser and navigate to http://localhost:8501

## Troubleshooting

- If the retrosynthesis feature doesn't work, ensure the RetroSynthesis Agent service is running
- Check the .env file if experiencing authentication errors with the OpenAI API
- For visualization issues, ensure you have appropriate permissions to write to the filesystem

## System Requirements

- Python 3.11 (required for both components)
- Sufficient RAM and disk space for running machine learning models
- Internet connection for API access
- Operating system: Windows, macOS, or Linux

## Next Steps

After successful installation, refer to the main README for:

- Usage instructions and examples
- Available tools and features
- Component architecture
- Workflow information
