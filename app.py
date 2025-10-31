


from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Import the ML predictor
from utils.ml_predictor import EngagementPredictor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize the real ML predictor
predictor = EngagementPredictor()

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/students')
def students():
    return render_template('students.html')

@app.route('/student/<student_id>')
def student_detail(student_id):
    return render_template('student_detail.html', student_id=student_id)

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

# API Routes
@app.route('/api/dashboard-stats')
def dashboard_stats():
    stats = {
        'total_students': 1247,
        'at_risk': 186,
        'intervention_rate': 87,
        'avg_attendance': 78.3,
        'risk_trend': -12.5
    }
    return jsonify(stats)

@app.route('/api/students')
def get_students():
    students = generate_sample_students(50)
    return jsonify(students)

@app.route('/api/student/<student_id>')
def get_student_detail(student_id):
    student = generate_student_detail(student_id)
    return jsonify(student)

@app.route('/api/predict', methods=['POST'])
def predict_engagement():
    try:
        data = request.json
        prediction = predictor.predict(data)
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'prediction': predictor._fallback_prediction(data)
        })

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict risk for multiple students at once"""
    try:
        students_data = request.json.get('students', [])
        predictions = []
        
        for student_data in students_data:
            prediction = predictor.predict(student_data)
            predictions.append({
                'student_id': student_data.get('id', 'unknown'),
                'prediction': prediction
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/model-info')
def model_info():
    """Get information about the current ML model"""
    model_info = {
        'model_type': 'Random Forest' if predictor.model else 'Rule-based Fallback',
        'feature_names': predictor.feature_names,
        'status': 'loaded' if predictor.model else 'fallback'
    }
    return jsonify(model_info)

@app.route('/api/risk-distribution')
def risk_distribution():
    # Generate realistic risk distribution based on model
    risks = []
    for _ in range(1000):
        # Generate random student data
        student_data = {
            'gpa': np.random.uniform(2.0, 4.0),
            'attendance': np.random.uniform(60, 95),
            'course_load': np.random.randint(12, 21),
            'library_visits': np.random.randint(0, 31),
            'lms_interactions': np.random.randint(10, 101),
            'extracurricular': np.random.randint(0, 11),
            'financial_aid': np.random.choice(['Yes', 'No']),
            'work_study': np.random.choice(['None', 'Part-time', 'Full-time']),
            'first_generation': np.random.choice(['Yes', 'No'])
        }
        prediction = predictor.predict(student_data)
        risks.append(prediction['risk_score'])
    
    fig = px.histogram(
        x=risks, 
        nbins=20, 
        title='Risk Score Distribution',
        labels={'x': 'Risk Score', 'y': 'Number of Students'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter")
    )
    
    return jsonify(json.loads(fig.to_json()))

# Helper functions
def generate_sample_students(count):
    students = []
    programs = ['Computer Science', 'Engineering', 'Business', 'Arts', 'Science']
    
    for i in range(count):
        # Generate student data
        student_data = {
            'gpa': round(np.random.uniform(2.0, 4.0), 2),
            'attendance': round(np.random.uniform(60, 95), 1),
            'course_load': np.random.randint(12, 21),
            'library_visits': np.random.randint(0, 31),
            'lms_interactions': np.random.randint(10, 101),
            'extracurricular': np.random.randint(0, 11),
            'financial_aid': np.random.choice(['Yes', 'No']),
            'work_study': np.random.choice(['None', 'Part-time', 'Full-time']),
            'first_generation': np.random.choice(['Yes', 'No'])
        }
        
        # Get real prediction from ML model
        prediction = predictor.predict(student_data)
        
        students.append({
            'id': f'S{10000 + i}',
            'name': f'Student {i+1}',
            'email': f'student{i+1}@university.edu',
            'program': np.random.choice(programs),
            'attendance': student_data['attendance'],
            'gpa': student_data['gpa'],
            'risk_score': round(prediction['risk_score'], 1),
            'risk_level': prediction['risk_level'],
            'last_activity': (datetime.now() - timedelta(days=np.random.randint(0, 30))).strftime('%Y-%m-%d')
        })
    return students

def generate_student_detail(student_id):
    # Generate realistic student data
    student_data = {
        'gpa': round(np.random.uniform(2.5, 3.8), 2),
        'attendance': round(np.random.uniform(70, 90), 1),
        'course_load': np.random.randint(12, 18),
        'library_visits': np.random.randint(5, 20),
        'lms_interactions': np.random.randint(30, 80),
        'extracurricular': np.random.randint(1, 6),
        'financial_aid': np.random.choice(['Yes', 'No']),
        'work_study': np.random.choice(['None', 'Part-time']),
        'first_generation': np.random.choice(['Yes', 'No'])
    }
    
    # Get real prediction
    prediction = predictor.predict(student_data)
    
    return {
        'id': student_id,
        'name': 'Alex Johnson',
        'program': 'Computer Science',
        'risk_score': prediction['risk_score'],
        'demographics': {
            'age': 20,
            'gender': 'Male',
            'residency': 'On-campus'
        },
        'academic': {
            'gpa': student_data['gpa'],
            'attendance': student_data['attendance'],
            'completed_credits': 45,
            'course_load': student_data['course_load']
        },
        'behavioral': {
            'library_visits': student_data['library_visits'],
            'lms_interactions': student_data['lms_interactions'],
            'extracurricular': student_data['extracurricular']
        },
        'socio_economic': {
            'financial_aid': student_data['financial_aid'],
            'work_study': student_data['work_study'],
            'first_generation': student_data['first_generation']
        },
        'prediction_details': prediction
    }

if __name__ == '__main__':
    app.run(debug=True)