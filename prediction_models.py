"""
Module 6 Assignment
Written by : Shivi Jain, Naina Gupta
Description: Functions to replace incorrect values, decision tree model and Random Forest Model
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score


# To replace '(null)' values with 'UNKNOWN'
def replace_null_with_unknown(dataset):
    # Select columns with text values
    text_columns = dataset.select_dtypes(include=['object']).columns

    # Replace '(test)' with 'unknown'
    dataset[text_columns] = dataset[text_columns].replace('(null)', 'UNKNOWN')

    return dataset

def get_age_label(age_group):
    if age_group == 'UNKNOWN':
        return age_group
    elif age_group == '<18' :
        return "Child"
    elif age_group == '65+':
        return "Senior"
    elif age_group == '25-44' or age_group == '45-64' or age_group == '18-24':
        return "Adult"
    elif int(age_group) < 0 :
        return "Child"
    else:
        return "Senior"



# To create Decision Tree model
def decisiontree_model(df1):
    # Add training and testing to the model.
    # 80% of the dataset is utilized in training the model
    # 20% is used as a test data
    # X_train and X_test: stores the Independent variables
    # y_train, y_test: stores the Dependent variables
    X = df1[['LAW_CAT_CD', 'SUSP_AGE_LAB', 'SUSP_RACE']]  # Features
    y = df1['SUSP_SEX']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
    classifier.fit(X_train, y_train)
    plt.figure(figsize=(50,20))
    tree.plot_tree(classifier, feature_names=X_train.columns, fontsize=30)
    
    # Shape of the training and testing datasets
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Descriptive statistics of the training dataset
    print("Training data statistics:")
    print(X_train.describe())

    # Descriptive statistics of the testing dataset
    print("Testing data statistics:")
    print(X_test.describe())


    predictions = classifier.predict(X_test)

    accuracy = classifier.score(X_test, y_test)
    print("\n Accuracy of this Model is : ", accuracy)
    return predictions;

# To create Random Forest Model
def randomforest_model(df2):
    # Add training and testing to the model.
    # 80% of the dataset is utilized in training the model
    # 20% is used as a test data
    # X_train and X_test: stores the Independent variables
    # y_train, y_test: stores the Dependent variables

    # Select the features and target variable
    features = ['KY_CD', 'BORO_NM', 'SUSP_SEX', 'SUSP_RACE', 'LAW_CAT_CD']
    target = 'SUSP_AGE_LAB'

    # Preprocess the data (optional)
    # Handle missing values, categorical encoding, feature scaling, etc.

    # Split the data into training and testing sets
    X = df2[features]
    y = df2[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    # Get feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.show()
    
    # Shape of the training and testing datasets
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Descriptive statistics of the training dataset
    print("Training data statistics:")
    print(X_train.describe())

    # Descriptive statistics of the testing dataset
    print("Testing data statistics:")
    print(X_test.describe())

    # Number of Trees
    num_trees = model.n_estimators
    print("Number of Trees:", num_trees)


    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of this Model:", accuracy)

    return  y_pred;

