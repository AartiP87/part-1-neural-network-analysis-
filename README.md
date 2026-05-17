# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview
This project builds and analyzes a feed-forward neural network to predict **customer churn** — whether a customer is likely to leave a service. It covers the full machine learning pipeline from data exploration to hyperparameter tuning.

## Dataset Link

Use the relevant Part 4 reference files from the shared folder:

https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

## Dataset
- **File:** `customer_churn_nn.csv`
- **Rows:** 2,000 customers
- **Features:** 16 columns (4 categorical, 11 numerical, 1 ID)
- **Target:** `churn` (1 = churned, 0 = retained)
- **Class balance:** ~98.5% retained, ~1.5% churned (imbalanced dataset)

## Project Structure
```
part-1-neural-network-analysis/
│
├── README.md                    ← This file
├── notebook.ipynb               ← Main Jupyter notebook (all 6 tasks)
├── requirements.txt             ← Python dependencies
├── customer_churn_nn.csv        ← Dataset
└── results/
    ├── target_distribution.png       ← Task 1: Churn class distribution charts
    ├── training_curves.png           ← Task 4: Loss and accuracy over epochs
    ├── evaluation_outputs.png        ← Task 4: Confusion matrix
    ├── model_comparison_table.csv    ← Task 5: Experiment results (Excel-friendly)
    └── model_comparison_table.png    ← Task 5: Experiment results (image)
```

## Tasks Completed
| Task | Description |
|------|-------------|
| Task 1 | Dataset understanding — shape, types, missing values, statistics, target distribution |
| Task 2 | Data preprocessing — encoding, scaling, train/test split |
| Task 3 | Neural network model building — Input → Dense(ReLU)+Dropout → Sigmoid output |
| Task 4 | Training and evaluation — accuracy, loss, confusion matrix, classification report |
| Task 5 | Hyperparameter experiments — 4 configurations compared |
| Task 6 | Final reflection — weights/biases, activation functions, learning rate, overfitting |

## Model Architecture
```
Input Layer  → (number of features)
Hidden Layer 1 → Dense(32, activation=ReLU) + Dropout(0.2)
Hidden Layer 2 → Dense(16, activation=ReLU) + Dropout(0.2)
Output Layer → Dense(1, activation=Sigmoid)

Loss      : Binary Cross-Entropy
Optimizer : Adam (lr=0.001)
```

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook notebook.ipynb
```

## Key Findings
- The dataset is highly imbalanced (1.5% churn rate), requiring class weighting during training
- Dropout layers help prevent overfitting
- A learning rate of 0.001 with Adam optimizer gave the most stable training
- See `results/model_comparison_table.csv` for full experiment comparison
