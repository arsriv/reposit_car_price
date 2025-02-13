from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import numpy as np
import os
import joblib
from django.conf import settings

# Load the model and data
model_path = os.path.join(settings.BASE_DIR, 'predictor', 'static', 'data', 'LinearRegressionModel.pkl')
data_path = os.path.join(settings.BASE_DIR, 'predictor', 'static', 'data', 'Cleaned_Car_data.csv')

# Load the dataset
car_data = pd.read_csv(data_path)

def index(request):
    companies = sorted(car_data['company'].unique())
    car_models = sorted(car_data['name'].unique())
    years = sorted(car_data['year'].unique(), reverse=True)
    fuel_types = car_data['fuel_type'].unique()
    return render(request, 'index.html', {
        'companies': companies,
        'car_models': car_models,
        'years': years,
        'fuel_types': fuel_types
    })

def predict(request):
    if request.method == 'POST':
        company = request.POST.get('company')
        car_model = request.POST.get('car_models')
        year = int(request.POST.get('year'))
        fuel_type = request.POST.get('fuel_type')
        driven = int(request.POST.get('kilo_driven'))

        # Prepare input data for the model
        input_data = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [year],
            'kms_driven': [driven],
            'fuel_type': [fuel_type]
        })

        # Load the model
        model = joblib.load(model_path)

        # Make prediction
        prediction = model.predict(input_data)
        prediction = np.round(prediction[0], 2)  # Round to 2 decimal places

        # Return prediction as JSON response
        return JsonResponse({'prediction': prediction})
        CORS_ALLOW_ALL_ORIGINS = True  # Allow all frontend requests
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
CORS_ALLOW_HEADERS = ["*"]
