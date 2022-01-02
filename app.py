from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('KNN_car_collision_ds.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    Acceleration_RMSE = request.form.get('Acceleration_RMSE')
    Gyroscope_X = request.form.get('Gyroscope_X')
    Magnetometer_RMSE = request.form.get('Magnetometer_RMSE')

    input_query = np.array(
        [[Acceleration_RMSE, Gyroscope_X, Magnetometer_RMSE]])

    result = model.predict(input_query)[0]

    return jsonify({'Collision': str(result)})


if __name__ == '_main_':
    app.run(debug=True)
