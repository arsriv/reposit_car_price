import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import joblib

# Load the dataset
car_data = pd.read_csv(r"C:\Users\arman\Desktop\sample2\car_prediction_project\predictor\static\data\Cleaned_Car_data.csv")

# Prepare features and target
X = car_data[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car_data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create preprocessing pipeline
preprocessor = make_column_transformer(
    (OneHotEncoder(), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Create the model pipeline
model = make_pipeline(preprocessor, LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))

# Save the retrained model
joblib.dump(model, r"C:\Users\arman\Desktop\sample2\car_prediction_project\predictor\static\data\LinearRegressionModel.pkl")