# Bug Prediction Model

A machine learning model that predicts bug types in Python code using PyTorch and NLTK.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize NLTK data (this will be done automatically when running the model, but you can run it separately):
```bash
python setup.py
```

## Running the Model

You have two options to run the model:

### 1. Using Jupyter Notebook (Recommended for Analysis)
```bash
jupyter notebook bug_prediction.ipynb
```
This will open an interactive notebook where you can:
- View data analysis and visualizations
- Train the model with different parameters
- Test predictions with custom examples

### 2. Using Command Line
```bash
python main.py
```
This will:
- Load the dataset
- Train the model
- Run a test prediction

## Model Architecture

- Uses BERT tokenizer for text encoding
- BiLSTM network for sequence processing
- Combines code snippets and bug descriptions for prediction
- NLTK for text preprocessing
- Supports both CPU and GPU training

## Dataset

The model uses a dataset (`python_bug_dataset.csv`) containing:
- Python code snippets
- Bug descriptions
- Bug types
- Severity levels

## Files

- `main.py`: Core model implementation
- `setup.py`: NLTK data initialization
- `requirements.txt`: Project dependencies
- `bug_prediction.ipynb`: Interactive analysis notebook
- `python_bug_dataset.csv`: Training data
