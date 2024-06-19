# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning algorithms. The dataset used contains credit card transactions labeled as fraudulent or legitimate.

## Dataset
The dataset used for this project is available on Kaggle and contains credit card transactions. It includes features such as transaction amount, time, and anonymized transaction information.

Dataset source: Credit Card Fraud Detection - Kaggle (https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Dependencies
To run this project, the following Python libraries are required:
- pandas
- scikit-learn
- matplotlib
- seaborn

You can install the dependencies using pip:

pip install pandas scikit-learn matplotlib seaborn

## Usage
1. Upload the Dataset: Upload the creditcard.csv dataset file to your Google Colab session.
2. Run the Notebook Cells: Execute each cell in the Google Colab notebook to:
   - Load and explore the dataset.
   - Preprocess the data (e.g., standardization, feature engineering).
   - Train different machine learning models (Logistic Regression, Decision Trees, Random Forests).
   - Evaluate model performance using metrics like accuracy, precision, recall, and ROC AUC score.
   - Visualize results, including ROC curves and confusion matrices.

## Models
Three machine learning models are implemented for comparison:
- Logistic Regression
- Decision Trees
- Random Forests

Each model is trained and evaluated to detect fraudulent transactions based on the provided dataset.

## Performance
- The performance of each model is evaluated using metrics such as accuracy, precision, recall, and ROC AUC score.
- ROC curves are plotted to visualize the trade-off between true positive rate and false positive rate for each model.

