# Assignment55
# Customer Loan Approval Using Voting Classification

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def main():

    # 1. Load the dataset
    filename = "Customer_Loan_Approval.csv"
    data = pd.read_csv(filename)

    print("Dataset:")
    print(data)

    print("\nDataset Shape:", data.shape)

    # 2. Check for missing values
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Remove rows containing missing values
    data = data.dropna()

    # 3. Separate input and output variables
    X = data.drop("LoanApproved", axis=1)
    Y = data["LoanApproved"]

    print("\nInput Variables:")
    print(X.columns.tolist())

    print("\nOutput Variable: LoanApproved")

    # Convert categorical variables into numerical variables
    X = pd.get_dummies(X, drop_first=True)

    # 4. Split dataset into training and testing data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    # Scaling for Logistic Regression and KNN
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)

    lr.fit(X_train_scaled, Y_train)

    lr_pred = lr.predict(X_test_scaled)

    # 6. Train Decision Tree
    dt = DecisionTreeClassifier(random_state=42)

    dt.fit(X_train, Y_train)

    dt_pred = dt.predict(X_test)

    # 7. Train KNN
    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train_scaled, Y_train)

    knn_pred = knn.predict(X_test_scaled)

    # 8. Calculate individual accuracy
    lr_accuracy = accuracy_score(Y_test, lr_pred)
    dt_accuracy = accuracy_score(Y_test, dt_pred)
    knn_accuracy = accuracy_score(Y_test, knn_pred)

    print("\nIndividual Model Accuracy:")

    print("Logistic Regression :", lr_accuracy)
    print("Decision Tree       :", dt_accuracy)
    print("KNN                 :", knn_accuracy)

    # 9. Create Hard Voting Classifier
    hard_voting = VotingClassifier(
        estimators=[
            (
                "lr",
                Pipeline([
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42
                        )
                    )
                ])
            ),

            (
                "dt",
                DecisionTreeClassifier(
                    random_state=42
                )
            ),

            (
                "knn",
                Pipeline([
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        KNeighborsClassifier(
                            n_neighbors=5
                        )
                    )
                ])
            )
        ],
        voting="hard"
    )

    hard_voting.fit(X_train, Y_train)

    hard_pred = hard_voting.predict(X_test)

    hard_accuracy = accuracy_score(
        Y_test,
        hard_pred
    )

    # 10. Display Hard Voting accuracy
    print("\nHard Voting Accuracy:")
    print(hard_accuracy)

    # 11. Create Soft Voting Classifier
    soft_voting = VotingClassifier(
        estimators=[
            (
                "lr",
                Pipeline([
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42
                        )
                    )
                ])
            ),

            (
                "dt",
                DecisionTreeClassifier(
                    random_state=42
                )
            ),

            (
                "knn",
                Pipeline([
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        KNeighborsClassifier(
                            n_neighbors=5
                        )
                    )
                ])
            )
        ],
        voting="soft"
    )

    soft_voting.fit(X_train, Y_train)

    soft_pred = soft_voting.predict(X_test)

    soft_accuracy = accuracy_score(
        Y_test,
        soft_pred
    )

    # 12. Display Soft Voting accuracy
    print("\nSoft Voting Accuracy:")
    print(soft_accuracy)

    # 13. Compare all models
    print("\nComparison:")
    print("-" * 42)

    print(f"{'Model':<25}{'Accuracy':>12}")

    print("-" * 42)

    print(
        f"{'Logistic Regression':<25}"
        f"{lr_accuracy:>12.4f}"
    )

    print(
        f"{'Decision Tree':<25}"
        f"{dt_accuracy:>12.4f}"
    )

    print(
        f"{'KNN':<25}"
        f"{knn_accuracy:>12.4f}"
    )

    print(
        f"{'Hard Voting':<25}"
        f"{hard_accuracy:>12.4f}"
    )

    print(
        f"{'Soft Voting':<25}"
        f"{soft_accuracy:>12.4f}"
    )

    print("-" * 42)


if __name__ == "__main__":
    main()
