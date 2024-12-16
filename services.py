import torch
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

MODEL_DIR = "models"
class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Define CustomNN Architecture Inline
class CustomNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(CustomNN, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)
# Inline preprocessing (matching training steps)

def preprocess_input(user_input, training_columns):
    """
    Preprocess user input using saved encoders and scaler.
    """
    import joblib
    import pandas as pd

    # Load saved encoders and scaler
    encoders = joblib.load('models/encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Encode categorical variables
    categorical_cols = encoders.keys()
    for col in categorical_cols:
        if col in input_df:
            input_df[col] = input_df[col].map(encoders[col]).fillna(-1).astype(int)

    # Ensure column order matches training
    input_df = input_df[training_columns]

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    return input_scaled

# Load Selected Model
def load_model(model_name, input_dim=None):
    if model_name == "custom_nn":
        model = CustomNN(input_dim)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "custom_nn_model.pth")))
        model.eval()
        return model
    elif model_name == "mlp":
        model = MLP(input_dim)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mlp_model.pth")))
        model.eval()
        return model
    elif model_name == "random_forest":
        return joblib.load(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
    elif model_name == "linear_regression":
        return joblib.load(os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
    else:
        raise ValueError(f"Model '{model_name}' not found!")

# Predict Function
def predict_score(model_name, user_input):
    """Make predictions using the specified model."""
    training_columns = ['School', 'Gender', 'Age', 'HomeAddress', 'FamilySize', 'ParentStatus', 'MotherEducation', 'FatherEducation', 'MotherJob', 'FatherJob', 'SchoolReason', 'Guardian', 'TravelTime', 'StudyTime', 'PastFailures', 'SchoolSupport', 'FamilySupport', 'PaidClasses', 'ExtracurricularActivities', 'NurserySchool', 'HigherEducation', 'InternetAccess', 'RomanticRelationship', 'FamilyRelationship', 'FreeTime', 'GoingOutWithFriends', 'WorkdayAlcoholConsumption', 'WeekendAlcoholConsumption', 'HealthStatus', 'Absences']

    preprocessed_input = preprocess_input(user_input, training_columns)

    # Load the model
    if model_name in ["custom_nn", "mlp"]:
        input_tensor = torch.FloatTensor(preprocessed_input)
        input_dim = preprocessed_input.shape[1]
        model = load_model(model_name, input_dim=input_dim)
        prediction = model(input_tensor).detach().numpy()[0][0]
    else:
        model = load_model(model_name)
        prediction = model.predict(preprocessed_input)[0]

    return prediction

if __name__ == "__main__":
    # Columns from training data (match exactly)
    training_columns = ['School', 'Gender', 'Age', 'HomeAddress', 'FamilySize', 'ParentStatus', 'MotherEducation', 'FatherEducation', 'MotherJob', 'FatherJob', 'SchoolReason', 'Guardian', 'TravelTime', 'StudyTime', 'PastFailures', 'SchoolSupport', 'FamilySupport', 'PaidClasses', 'ExtracurricularActivities', 'NurserySchool', 'HigherEducation', 'InternetAccess', 'RomanticRelationship', 'FamilyRelationship', 'FreeTime', 'GoingOutWithFriends', 'WorkdayAlcoholConsumption', 'WeekendAlcoholConsumption', 'HealthStatus', 'Absences']
    print(len(training_columns))
    print("Select a model to make predictions:")
    print("1. Custom Neural Network\n2. MLP\n3. Random Forest\n4. Linear Regression")
    choice = int(input("Enter your choice (1-4): "))

    model_mapping = {
        1: "custom_nn",
        2: "mlp",
        3: "random_forest",
        4: "linear_regression"
    }

    selected_model = model_mapping.get(choice)
    if not selected_model:
        print("Invalid choice. Exiting...")
        exit()

    # Example user input (matching columns)
    user_input = {
    'School': 'GP',
    'Gender': 'F',
    'Age': '18',
    'HomeAddress': 'U',
    'FamilySize': 'GT3',
    'ParentStatus': 'A',
    'MotherEducation': '4',
    'FatherEducation': '4',
    'MotherJob': 'at_home',
    'FatherJob': 'teacher',
    'SchoolReason': 'course',
    'Guardian': 'mother',
    'TravelTime': '2',
    'StudyTime': '2',
    'PastFailures': '0',
    'SchoolSupport': 'yes',
    'FamilySupport': 'no',
    'PaidClasses': 'no',
    'ExtracurricularActivities': 'no',
    'NurserySchool': 'yes',
    'HigherEducation': 'yes',
    'InternetAccess': 'no',
    'RomanticRelationship': 'no',
    'FamilyRelationship': '4',
    'FreeTime': '3',
    'GoingOutWithFriends': '4',
    'WorkdayAlcoholConsumption': '1',
    'WeekendAlcoholConsumption': '1',
    'HealthStatus': '3',
    'Absences': '6'
}

    print(len(user_input))
    # Prediction
    print("\nPredicting...")
    prediction = predict(selected_model, user_input, training_columns)
    print(f"Prediction from {selected_model}: {prediction:.2f}")
