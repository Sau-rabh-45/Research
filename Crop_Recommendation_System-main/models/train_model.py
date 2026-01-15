# train_model.py
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from utils.data_cleaning import clean_dataset

# Load and Clean data
df = clean_dataset("data/crop_data.csv")

X = df.drop("label", axis=1)
y = df["label"] # string crop names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Decision Tree Model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# Save Model
with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(model, f)

print("Cleaned data -> Model trained -> Saved successfully")