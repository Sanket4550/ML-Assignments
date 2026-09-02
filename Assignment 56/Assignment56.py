# Assignment56
# Fraudulent Transaction Detection Using Ensemble Learning

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def main():

    # Load the dataset
    filename = "Fraudulent_Transaction_Detection.csv"
    data = pd.read_csv(filename)

    print("Dataset:")
    print(data)

    print("\nDataset Shape:", data.shape)

    # Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Remove rows containing missing values
    data = data.dropna()

    # Separate input and output variables
    X = data.drop("Fraud", axis=1)
    Y = data["Fraud"]

    print("\nInput Variables:")
    print(X.columns.tolist())

    print("\nOutput Variable: Fraud")

    # Convert categorical variables into numerical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split dataset into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # --------------------------------------------------
    # 1. Decision Tree Classifier
    # --------------------------------------------------

    decision_tree = DecisionTreeClassifier(
        random_state=42
    )

    decision_tree.fit(X_train, Y_train)

    dt_pred = decision_tree.predict(X_test)

    # --------------------------------------------------
    # 2. Bagging Classifier
    # --------------------------------------------------

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(
            random_state=42
        ),
        n_estimators=10,
        random_state=42
    )

    bagging.fit(X_train, Y_train)

    bagging_pred = bagging.predict(X_test)

    # --------------------------------------------------
    # 3. Random Forest Classifier
    # --------------------------------------------------

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    random_forest.fit(X_train, Y_train)

    rf_pred = random_forest.predict(X_test)

    # --------------------------------------------------
    # 4. AdaBoost Classifier
    # --------------------------------------------------

    adaboost = AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    )

    adaboost.fit(X_train, Y_train)

    ada_pred = adaboost.predict(X_test)

    # --------------------------------------------------
    # 5. Voting Classifier
    # --------------------------------------------------

    voting = VotingClassifier(
        estimators=[
            (
                "decision_tree",
                DecisionTreeClassifier(
                    random_state=42
                )
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            ),
            (
                "adaboost",
                AdaBoostClassifier(
                    n_estimators=50,
                    random_state=42
                )
            )
        ],
        voting="hard"
    )

    voting.fit(X_train, Y_train)

    voting_pred = voting.predict(X_test)

    # --------------------------------------------------
    # 6. Evaluation Function
    # --------------------------------------------------

    def evaluate_model(name, y_true, y_pred):

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        matrix = confusion_matrix(
            y_true,
            y_pred
        )

        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)

        print("Accuracy  :", accuracy)
        print("Precision :", precision)
        print("Recall    :", recall)
        print("F1 Score  :", f1)

        print("\nConfusion Matrix:")
        print(matrix)

        return accuracy, precision, recall, f1

    # --------------------------------------------------
    #  Evaluate all models
    # --------------------------------------------------

    dt_result = evaluate_model(
        "Decision Tree",
        Y_test,
        dt_pred
    )

    bagging_result = evaluate_model(
        "Bagging Classifier",
        Y_test,
        bagging_pred
    )

    rf_result = evaluate_model(
        "Random Forest Classifier",
        Y_test,
        rf_pred
    )

    ada_result = evaluate_model(
        "AdaBoost Classifier",
        Y_test,
        ada_pred
    )

    voting_result = evaluate_model(
        "Voting Classifier",
        Y_test,
        voting_pred
    )

    # --------------------------------------------------
    # Final Comparison
    # --------------------------------------------------

    print("\n\nFINAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Algorithm':<25}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 75)

    results = [
        ("Decision Tree", dt_result),
        ("Bagging", bagging_result),
        ("Random Forest", rf_result),
        ("AdaBoost", ada_result),
        ("Voting", voting_result)
    ]

    for name, result in results:

        accuracy, precision, recall, f1 = result

        print(
            f"{name:<25}"
            f"{accuracy:>12.4f}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{f1:>12.4f}"
        )

    print("=" * 75)

    # --------------------------------------------------
    #  Recommend the best model
    # --------------------------------------------------

    best_model = max(
        results,
        key=lambda item: item[1][3]
    )

    print(
        "\nRecommended Model based on highest F1 Score:",
        best_model[0]
    )


if __name__ == "__main__":
    main()
