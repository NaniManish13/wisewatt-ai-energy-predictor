from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load Dataset
file_path = 'energy_consumption.csv'
df = pd.read_csv(file_path)

# Encode Appliance Names
encoder = LabelEncoder()
df['Appliance'] = encoder.fit_transform(df['Appliance'])

# Features and Target
X = df[['Temperature', 'Humidity', 'Usage_Hours', 'Power_Rating', 'Appliance']]
y = df['Energy_Consumption']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Accuracy
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)


@app.route('/')
def home():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    usage_hours = float(request.form['usage_hours'])
    power_rating = float(request.form['power_rating'])
    appliance_name = request.form['appliance']

    appliance = encoder.transform([appliance_name])[0]

    sample = [[temperature, humidity, usage_hours, power_rating, appliance]]

    predicted_energy = model.predict(sample)[0]

    if predicted_energy > 6:
        suggestion = "Reduce usage during peak hours and enable energy-saving mode."
    else:
        suggestion = "Energy consumption is optimal."

    return render_template(
        'index.html',
        prediction=round(predicted_energy, 2),
        suggestion=suggestion,
        mae=round(mae, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)