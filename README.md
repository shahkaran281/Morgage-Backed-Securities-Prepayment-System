# Mortgage-Backed Securities Prepayment Prediction System

This project aims to predict the likelihood of prepayment for mortgage-backed securities (MBS) using machine learning techniques. The system processes mortgage data to evaluate the risk of loan prepayment, which is crucial for investors managing MBS portfolios.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Model Evaluation](#model-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Model Persistence](#model-persistence)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project leverages machine learning techniques to predict prepayment risk in mortgage-backed securities (MBS). Using data such as borrower credit scores, loan-to-value ratios, and loan age, the system predicts whether a loan is likely to be prepaid. The project involves:
- Data preprocessing using Python and Pandas.
- Building and training models using **Logistic Regression**, **Decision Tree**, and **Random Forest** algorithms.
- Evaluating the models' performance based on accuracy and other metrics.

## Dataset

The dataset used in this project is sourced from **Freddie Mac**, containing data on mortgages with around **500,137 rows** and **27 columns**. It includes various features like:
- **Loan balance**
- **Credit scores**
- **Loan age**
- **Prepayment history**

The dataset is cleaned and processed to remove missing values, handle categorical variables, and prepare it for machine learning models.

## Technologies

- **Python**: For data manipulation, machine learning, and visualization.
- **Scikit-learn**: For implementing machine learning models (Logistic Regression, Decision Tree, Random Forest).
- **Pickle**: For saving and loading trained models.
- **Matplotlib/Seaborn**: For visualizing the data and results.

## Data Preprocessing

1. **Handling Missing Data**: The dataset is cleaned by dropping rows and columns with excessive missing values.
2. **Feature Engineering**:
   - Categorical variables like `FIRST_TIME_HOMEBUYER_FLAG`, `OCCUPANCY_STATUS`, and others are encoded into numeric formats.
   - The target variable `PREPAID` is label-encoded into a binary format (0 or 1).
3. **Feature Selection**: A subset of 17 key features is selected for model training based on their correlation with the prepayment target.

## Modeling

The following machine learning models are used to predict mortgage prepayment risk:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

The models are trained using the preprocessed dataset and then evaluated based on their performance on a test set.

## Model Evaluation

The models are evaluated using:
- **Accuracy**: The proportion of correct predictions.
- **Classification Report**: Includes precision, recall, F1-score for each class.
- **Confusion Matrix**: A matrix that shows the performance of the models.

Each model's performance is evaluated, with **Logistic Regression** yielding the highest accuracy.

## Installation

To set up this project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/shahkaran281/Morgage-Backed-Securities-Prepayment-System.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Morgage-Backed-Securities-Prepayment-System
   ```

3. Run the Jupyter notebook (`Mortgage Backed Securities Prepayment System.ipynb`) to execute the steps:
   ```bash
   jupyter notebook
   ```

## Usage

1. **Data Processing**: Run the data preprocessing steps in the Jupyter notebook to clean and prepare the data.
2. **Model Training**: Train the machine learning models (Logistic Regression, Decision Tree, Random Forest) using the training dataset.
3. **Model Evaluation**: Evaluate each model using accuracy and classification reports.
4. **Model Persistence**: Save the trained models using Pickle for future use.

## Model Persistence

Once the models are trained, they can be saved using **Pickle** for reuse without retraining:

```python
import pickle
filename = 'finalized_model.sav'
pickle.dump(log, open(filename, 'wb'))
```

This will save the trained Logistic Regression model, which can be loaded later for making predictions on new data.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request. 

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
```
