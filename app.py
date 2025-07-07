from flask import Flask,render_template,request
import joblib
import numpy as np

app=Flask(__name__)

model = joblib.load('student_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('result.html', prediction="Error: Model or scaler not loaded properly.", is_error=True)

    try:
        try:
            hours = float(request.form['hours'])
            prev_score = float(request.form['score'])
            sleep = float(request.form['sleep'])
            papers = float(request.form['papers'])
        except ValueError:
            return render_template('result.html', prediction="Error: Enter valid numeric inputs.", is_error=True)

        extra = request.form['extra'].strip().lower()

        if hours < 0 or prev_score < 0 or sleep < 0 or papers < 0:
            return render_template('result.html', prediction="Error: Negative values not allowed.", is_error=True)

        if hours == 0 and prev_score == 0 and sleep == 0 and papers == 0:
            return render_template('result.html', prediction="Error: All values cannot be zero.", is_error=True)

        if extra not in ['yes', 'no']:
            return render_template('result.html', prediction="Error: Invalid extracurricular value.", is_error=True)

        extra_encoded = 1 if extra == 'yes' else 0

        input_features = np.array([[hours, prev_score, sleep, papers]])
        input_scaled = scaler.transform(input_features)
        input_final = np.concatenate([input_scaled, [[extra_encoded]]], axis=1)

        result = model.predict(input_final)[0]
        result = max(0, round(result, 2))

        return render_template('result.html', prediction=result, is_error=False)

    except Exception as e:
        print("Exception in prediction:", e)
        return render_template('result.html', prediction=f"Error: {str(e)}", is_error=True)


if __name__ == '__main__':
    app.run(debug=True)