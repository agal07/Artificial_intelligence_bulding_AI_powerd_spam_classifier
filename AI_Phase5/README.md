# Building a Smarter AI- Powered by Spam Classifier README

This Jupyter Notebook provides a machine learning model to build an AI spam classifier. The code includes data preprocessing, exploratory data analysis, data cleaning, and model building. Below are the steps to run this code:

## Prerequisites
1. You need to have Python installed on your system.
2. Install Jupyter Notebook and the required libraries using pip:
   ```bash
   pip install jupyter numpy pandas seaborn matplotlib scikit-learn scipy
   ```

## Steps to Run the Code
1. Clone or download the Jupyter Notebook file and the dataset ('spam.csv') to your local machine.
2. Open a terminal and navigate to the directory containing the Jupyter Notebook and the dataset.
3. Start a Jupyter Notebook session:
   ```bash
   jupyter notebook
   ```
4. In the Jupyter Notebook dashboard, open the 'AI_PHASE5.ipynb' file.
5. Run the code cells in the notebook sequentially by clicking on each cell and pressing Shift + Enter.
6. You can interact with the code, view visualizations, and see model performance metrics as the code executes.
7. The final model, a Random Forest Classifier, is saved as 'model.pkl' and can be used for spam classification

## Dataset Used
In this project, we used a spam dataset obtained from Kaggle. The dataset contains a many spam words which will be filtered in the spam classification.For training and testing the AI-based Spam Classification Model, a comprehensive dataset of messages, including both legitimate and spam messages, will be essential. This dataset will serve as the foundation for training the model to differentiate between the two message categories. You can access the dataset on Kaggle at the following URL: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## About the Dataset
The "SMS Spam Collection Dataset" is a dataset available on Kaggle. It contains SMS messages categorized as either "ham" (legitimate) or "spam." It's used for training spam classification models and contains label information (ham or spam) and the text content of the messages. The dataset is cleaned and preprocessed for machine learning tasks. Researchers use it to evaluate the performance of spam classification algorithms.

The key features in the dataset include:
The key features of the "SMS Spam Collection Dataset" typically include:

Label (Class): This feature categorizes each SMS message as either "ham" (legitimate) or "spam." It serves as the target variable for classification tasks.

Text Content: The main feature containing the actual text of the SMS messages. It's the primary data used for analyzing and classifying messages as either ham or spam.

These two features are the most essential components of the dataset, and they are used to train machine learning models for spam classification. The "Label" feature is the target variable that the models aim to predict based on the text content feature. Other potential features like message length, sentiment analysis, or metadata may be extracted or engineered from the text content to enhance the model's performance in spam detection.

## Understanding the Code
The code consists of the following sections:
- Importing necessary libraries and reading the dataset.
- Data preprocessing, cleaning, and feature engineering.
- Exploratory data analysis with visualizations.
- Building and evaluating regression models:
  - Linear Regression
  - Polynomial Regression
  - Decision Tree Regression
  - Random Forest Regression
- Building and evaluating a classification model for spam classification
- Saving the trained Random Forest Classifier model for spam filteration

## Data Sources
The dataset used in this code ('spam.csv') is assumed to be available in the same directory as the Jupyter Notebook. This dataset contains various spam messages, which are used to train and evaluate the models.

