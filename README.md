# Spam Email Classifier

## Project Overview

This project implements a basic email spam classification model using Python. It utilizes natural language processing (NLP) techniques to classify emails into "spam" or "ham" (non-spam). The model is trained using logistic regression and evaluates its performance with metrics such as accuracy, confusion matrix, and classification report.

## Features

- **Data Handling**: Reads and processes email data from specified directories.
- **Text Vectorization**: Converts email content into numerical features using `CountVectorizer`.
- **Model Training**: Trains a logistic regression model to classify emails.
- **Model Evaluation**: Assesses the model's performance using accuracy, confusion matrix, and classification report.
- **Feature Importance**: Identifies and displays the top features contributing to the classification.

## Installation

Ensure you have the necessary libraries installed. You can install them using `pip`:

```bash
pip install pandas scikit-learn
```

## Data Structure

- **Training Data**:
  - The data should be organized in the following directory structure:
    ```
    training_data/
    ├── ham/
    │   └── *.txt
    └── spam/
        └── *.txt
    ```
  - Each directory contains text files where each file represents an email.

## Usage

1. **Prepare Your Data**:
   - Place your email text files into the `training_data/ham` and `training_data/spam` directories.

2. **Run the Script**:

    ```bash
    python main.py
    ```

3. **View Results**:
   - The script will print the following:
     - **Accuracy** of the classification model.
     - **Confusion Matrix** showing the performance of the classifier.
     - **Classification Report** detailing precision, recall, and F1-score.
     - **Top 10 Positive Features** and **Top 10 Negative Features** indicating which words have the most impact on the classification.

## Code Overview

- **`read_spam()`** and **`read_ham()`**:
  - Functions to read emails from the respective directories and categorize them.

- **`read_category(category, directory)`**:
  - Helper function to read emails from a given directory and categorize them.

- **`preprocessor(e)`**:
  - Placeholder for text preprocessing. Modify if needed.

- **Vectorization**:
  - Converts email content into numerical format using `CountVectorizer`.

- **Model Training**:
  - Trains a logistic regression model with the vectorized text data.

- **Model Evaluation**:
  - Evaluates and prints the model's performance metrics.

- **Feature Importance**:
  - Identifies and displays the most significant features for classification.

## File Structure

- `main.py`: Main script for email spam classification.

## Contribution

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.
