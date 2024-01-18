#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ved Agrawal

DS 2500

11/10

Analyzing banks that have been insured by the FDIC and determinig whether
they have failed or not.

"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    """
    Loads data from CSV files.
    Returns:
        tuple: A tuple containing two pandas DataFrames, one for institutions and one for banklist.
    """
    
    institutions_df = pd.read_csv('institutions.csv')
    
    banklist_df = pd.read_csv('banklist.csv', encoding='cp1252')
    
    return institutions_df, banklist_df

def preprocess_data(institutions_df, banklist_df):
    """
    Preprocesses the data by merging and cleaning.
    Args:
        institutions_df (DataFrame): The DataFrame containing data from institutions.csv.
        banklist_df (DataFrame): The DataFrame containing data from banklist.csv.
    Returns:
        DataFrame: A merged and preprocessed DataFrame.
    """
    
    banklist_df.rename(columns={'Cert ': 'CERT'}, inplace=True)
    
    banklist_df['Failed'] = 1
    
    merged_df = pd.merge(institutions_df, banklist_df[['CERT', 'Failed']], on='CERT', how='left')
    
    merged_df['Failed'].fillna(0, inplace=True)
    
    return merged_df

def select_and_scale_features(merged_df, features, features_for_normalization):
    """
    Selects and manually scales the features required for model training using min-max normalization.
    Args:
        merged_df (DataFrame): The merged DataFrame containing all data.
        features (list): List of feature names to be selected.
        features_for_normalization (list): List of feature names to be normalized.
    Returns:
        DataFrame: The DataFrame with selected and manually scaled features.
    """
    
    selected_df = merged_df[features]
    
    selected_df.dropna(inplace=True)

    for feature in features_for_normalization:
        
        min_value = selected_df[feature].min()
        
        max_value = selected_df[feature].max()
        
        selected_df[feature] = (selected_df[feature] - min_value) / (max_value - min_value)

    return selected_df

def train_models(X, y, k_values, X_train, y_train, kfold):
    """
    Trains KNN models for a range of k values and evaluates them using cross-validation.
    Args:
        X_train (DataFrame): Training data features.
        y_train (Series): Training data labels.
        k_values (range): A range of k values for KNN.
        kfold (KFold): A KFold cross-validation object.
    Returns:
        dict: Three dictionaries containing accuracy, precision, and recall scores for each k.
    """
    
    accuracy_scores = {}
    
    precision_scores = {}
    
    recall_scores = {}

    for k in k_values:
        
        knn = KNeighborsClassifier(n_neighbors=k)
        
        knn.fit(X_train, y_train)
        
        scores = cross_validate(knn, X, y, cv=kfold, scoring=['accuracy', 'precision_macro', 'recall_macro'])
        
        accuracy_scores[k] = np.mean(scores['test_accuracy'])
        
        precision_scores[k] = np.mean(scores['test_precision_macro'])
        
        recall_scores[k] = np.mean(scores['test_recall_macro'])

    return accuracy_scores, precision_scores, recall_scores

def find_optimal_k(accuracy_scores, precision_scores, recall_scores):
    """
    Determines the optimal k values for accuracy, precision, and recall.
    Args:
        accuracy_scores (dict): A dictionary with k values as keys and accuracy scores as values.
        precision_scores (dict): A dictionary with k values as keys and precision scores as values.
        recall_scores (dict): A dictionary with k values as keys and recall scores as values.
    Returns:
        tuple: A tuple containing the optimal k values for accuracy, precision, recall, and the lowest mean accuracy.
    """
    
    best_k_accuracy = max(accuracy_scores, key=accuracy_scores.get)
    
    best_k_precision = max(precision_scores, key=precision_scores.get)
    
    best_k_recall = max(recall_scores, key=recall_scores.get)
    
    lowest_mean_accuracy = min(accuracy_scores.values())
    
    return best_k_accuracy, best_k_precision, best_k_recall, lowest_mean_accuracy

def evaluate_model(X_test, y_test, knn_model):
    """
    Evaluates the KNN model on the test set.
    Args:
        X_test (DataFrame): Test data features.
        y_test (Series): Test data labels.
        knn_model (KNeighborsClassifier): A trained KNN classifier.
    Returns:
        tuple: A tuple containing the F1 score for the failed class and the number of correct predictions for non-failure.
    """
    
    y_pred = knn_model.predict(X_test)
    
    f1_score_failed_class = f1_score(y_test, y_pred, labels=[1])
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    correct_predictions_not_fail = tn
    
    return f1_score_failed_class, correct_predictions_not_fail

def predict_bank_status(model, bank_cert, institutions_df, features_for_normalization):
    """
    Predicts the status of a specific bank using the trained model.
    Args:
        model (KNeighborsClassifier): A trained KNN classifier.
        bank_cert (int): The FDIC certificate number of the bank.
        institutions_df (DataFrame): The DataFrame containing data from institutions.csv.
        features_for_normalization (list): List of feature names to be normalized.
    Returns:
        str: The predicted status ('Failed' or 'Did Not Fail') of the bank.
    """
    
    bank_features = institutions_df[institutions_df['CERT'] == bank_cert][features_for_normalization]
    
    if bank_features.empty:
        
        return "Bank not found in dataset"

    # Normalize features
    
    for feature in features_for_normalization:
        
        min_value = institutions_df[feature].min()
        
        max_value = institutions_df[feature].max()
        
        bank_features[feature] = (bank_features[feature] - min_value) / (max_value - min_value)
    
    prediction = model.predict(bank_features)[0]
    
    return "Failed" if prediction == 1 else "Did Not Fail"

def plot_confusion_matrix_heatmap(model, X_test, y_test):
    """
    Plots a heatmap of the confusion matrix for the given model and test data.
    Args:
        model: Trained KNeighborsClassifier model.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)

    labels = ['Active', 'Failed']

    plt.figure(figsize=(8, 6))
    
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap='Blues', xticklabels=labels, yticklabels=labels)
    
    plt.title('Confusion Matrix Heatmap when K is Optimized for Recall')
    
    plt.xlabel('Predicted Labels')
    
    plt.ylabel('Actual Labels')
    
    plt.show()
    
def plot_performance_metrics(k_values, accuracy_scores, precision_scores, recall_scores):
        
        """
        Plots the performance metrics (accuracy, precision, recall) for different values of k.
        Args:
            k_values: Range of k values used in the model.
            accuracy_scores: Dictionary of accuracy scores for each k.
            precision_scores: Dictionary of precision scores for each k.
            recall_scores: Dictionary of recall scores for each k.
        """
        plt.figure(figsize=(18, 6))

        # Plotting Accuracy
        
        plt.subplot(1, 3, 1)
        
        plt.plot(k_values, [accuracy_scores[k] for k in k_values], marker='o')
        
        plt.title('Accuracy Scores for Different k Values')
        
        plt.xlabel('k Value')
        
        plt.ylabel('Accuracy Score')
        
        plt.grid(True)

        # Plotting Precision
        
        plt.subplot(1, 3, 2)
        
        plt.plot(k_values, [precision_scores[k] for k in k_values], marker='o', color='green')
        
        plt.title('Precision Scores for Different k Values')
        
        plt.xlabel('k Value')
        
        plt.ylabel('Precision Score')
        
        plt.grid(True)

        # Plotting Recall
        
        plt.subplot(1, 3, 3)
        
        plt.plot(k_values, [recall_scores[k] for k in k_values], marker='o', color='red')
        
        plt.title('Recall Scores for Different k Values')
        
        plt.xlabel('k Value')
        
        plt.ylabel('Recall Score')
        
        plt.grid(True)

        plt.tight_layout()
        
        plt.show()


def main():
    """
    Main function to orchestrate the data loading, preprocessing, model training, evaluation,
    and prediction for a specific bank's status.
    Outputs the answers to the assignment questions.
    """
    
    institutions_df, banklist_df = load_data()
    
    merged_df = preprocess_data(institutions_df, banklist_df)
    
    features = ['ASSET', 'DEP', 'DEPDOM', 'NETINC', 'OFFDOM', 'ROA', 'ROAPTX', 'ROE', 'Failed']
    
    features_for_normalization = ['ASSET', 'DEP', 'DEPDOM', 'NETINC', 'OFFDOM', 'ROA', 'ROAPTX', 'ROE']
    
    selected_df = select_and_scale_features(merged_df, features, features_for_normalization)
    
    X, y = selected_df.drop('Failed', axis=1), selected_df['Failed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    kfold = KFold(n_splits=4, random_state=0, shuffle=True)
    
    k_values = range(4, 19)

    accuracy_scores, precision_scores, recall_scores = train_models(X, y, k_values, X_train, y_train, kfold)
    
    best_k_accuracy, best_k_precision, best_k_recall, lowest_mean_accuracy = find_optimal_k(accuracy_scores, precision_scores, recall_scores)

    knn_final = KNeighborsClassifier(n_neighbors=best_k_accuracy)
    
    knn_final.fit(X_train, y_train)
    
    f1_score_failed_class, correct_predictions_not_fail = evaluate_model(X_test, y_test, knn_final)
    
    southern_community_bank_status = predict_bank_status(knn_final, 35251, institutions_df, features_for_normalization)
    
    # Train a model with the optimal k for recall
     
    knn_optimal_recall = KNeighborsClassifier(n_neighbors=best_k_recall)
     
    knn_optimal_recall.fit(X_train, y_train)

    plot_confusion_matrix_heatmap(knn_optimal_recall, X_test, y_test)
    
    plot_performance_metrics(k_values, accuracy_scores, precision_scores, recall_scores)

    answers = {
        
        "Optimal k for Accuracy": best_k_accuracy,
        
        "Lowest Mean Accuracy": lowest_mean_accuracy,
        
        "Optimal k for Precision": best_k_precision,
        
        "Optimal k for Recall": best_k_recall,
        
        "F1 Score for Failed Class": f1_score_failed_class,
        
        "Correct Predictions Not Fail": correct_predictions_not_fail,
        
        "Southern Community Bank Prediction": southern_community_bank_status
    }

    print(answers)
    
  

if __name__ == "__main__":
    main()