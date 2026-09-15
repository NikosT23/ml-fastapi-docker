import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(f"Model accuracy: {accuracy}")

joblib.dump(model, "iris_model.joblib")
print("Model saved as iris_model.joblib")
