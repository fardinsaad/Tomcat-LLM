## Setup
This repository contains:
- `.env`: Contains the API key.
- `llm.py`: Loads the API key.
- `env_setup`: Loads the dataset for different grid configurations.
- `agent.py`: Contains the Tomcat's framework and prompting techniques.
- `test_agent.ipynb`: Contains code for running the experiments Tomcat with PaP and Tomcat with Fs-CoT.

## 1. Configure the Environment

Before running the code, **ensure you set up your `.env` file** with ***your GPT-4o OpenAI API key***:

Make sure to replace `'Your API Key'` with your **actual OpenAI API key**.

## 2. Running the Code

Navigate to the `test_agent.ipynb` notebook and follow these steps:

### **Environment Setup:**
- Run the **first two cells** under the **"Environment Setup"** header to import the necessary Python files.
>**Note:** Use python kernel 3.12 or above for running the cells.

### **Experiments:**
- Run the **two cells** under the **"Experiments"** section to execute the experiments. The first cell corresponds to PaP and the second cell is for Fs-CoT.
  
  The results of the experiments will automatically be saved in:
  - `ToM_PaP-dataset.csv` (for experiments ***using PaP***)
  - `ToM_FsCoT-dataset.csv` (for experiments ***using Fs-CoT***)

### **Dataset Information:**
- The `dataset/problems` folder contains grid configurations. The code will automatically import the dataset required for the experiments.

## 3. Running Individual Commands

- For running individual commands, use the cell under the **"Individual Instruction"** header. 
- You can load different grid configurations and modify the instructions to test various cases and behaviors.

---

>**Note:** Make sure all necessary dependencies are installed and configured as per the project requirements. For instance, numpy, panda, openai, backoff, and python-dotenv.