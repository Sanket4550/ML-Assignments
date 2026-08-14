import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("MarvellousInfosystems_PlayPredictor.csv")

print("Dataset:")
print(df)

# Encode categorical columns
weather_encoder = LabelEncoder()
temperature_encoder = LabelEncoder()
play_encoder = LabelEncoder()

df["Weather"] = weather_encoder.fit_transform(df["Weather"])
df["Temperature"] = temperature_encoder.fit_transform(df["Temperature"])
df["Play"] = play_encoder.fit_transform(df["Play"])

# Features and target
X = df[["Weather", "Temperature"]]
y = df["Play"]

# Create and train KNN model
# Assignment specifies K = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Take input from user
print("\nEnter weather and temperature:")
weather = input("Weather (Sunny/Overcast/Rainy): ")
temperature = input("Temperature (Hot/Cool/Mild): ")

# Encode input
weather_value = weather_encoder.transform([weather])[0]
temperature_value = temperature_encoder.transform([temperature])[0]

# Prediction
prediction = model.predict([[weather_value, temperature_value]])
result = play_encoder.inverse_transform(prediction)

print("\nPrediction:", result[0])


# Function to calculate accuracy by splitting the dataset
def CheckAccuracy(k=3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy for K = {k}: {accuracy * 100:.2f}%")
    return accuracy


# Calculate accuracy for different K values
print("\nAccuracy for different values of K:")
for k in [1, 3, 5, 7]:
    CheckAccuracy(k)
