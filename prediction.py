import pickle
from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)

# Load model and preprocessing objects
model = pickle.load(open('promotion-prediction.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))
encoder_department = pickle.load(open('encoding_department.pkl', 'rb'))
encoder_education = pickle.load(open('encoding_education.pkl', 'rb'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html', education_options=encoder_education.classes_, department_options=encoder_department.classes_)

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        department = request.form.get('department')
        education = request.form.get('education')
        no_of_trainings = request.form.get('trainings')
        age = request.form.get('age')
        previous_year_rating = request.form.get('rating')
        length_of_service = request.form.get('service')
        KPIs = request.form.get('kpis')
        awards_won = request.form.get('awards')
        avg_training_score = request.form.get('score')

        # Check if any field is empty
        if any(value is None or value == '' for value in [department, education, no_of_trainings, age,
                                                           previous_year_rating, length_of_service,
                                                           KPIs, awards_won, avg_training_score]):
            print('-------------------------------------------')
            return render_template('predict.html',error_message="All fields are required", education_options=encoder_education.classes_, department_options=encoder_department.classes_)


        # Convert types and process as before
        try:
            no_of_trainings = float(no_of_trainings)
            age = float(age)
            previous_year_rating = float(previous_year_rating)
            length_of_service = float(length_of_service)
            KPIs = int(KPIs)
            awards_won = int(awards_won)
            avg_training_score = float(avg_training_score)

            # Encode categorical variables
            education_encoded = encoder_education.transform([[education]])[0]
            department_encoded = encoder_department.transform([[department]])[0]

            # Prepare input data for prediction
            total = [[department_encoded, education_encoded, no_of_trainings, age, previous_year_rating,
                      length_of_service, KPIs, awards_won, avg_training_score]]

            # Scale the input data
            total_scaled = scaler.transform(total)

            # Make prediction
            prediction = model.predict(total_scaled)
            if prediction == 0:
                text = 'Sorry, you are not eligible for promotion.'
                color = "rgb(255, 46, 10)"
            elif prediction == 1:
                text = 'You are eligible for promotion.'
                color = "rgb(79, 255, 73)"

            return render_template('submit.html', text=text , color = color)
        
        except ValueError:
            return render_template('predict.html',error_message="Incorrect details encountered , Please pass correct details", education_options=encoder_education.classes_, department_options=encoder_department.classes_)
    else:
        return redirect(url_for('home'))  # Redirect to home if GET request

if __name__ == '__main__':
    app.run()
