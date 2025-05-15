from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
from .ml_models import MLModel
import pandas as pd
import os

views = Blueprint('views', __name__)
ml_model = MLModel()

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        note = request.form.get('note')
        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)
            db.session.add(new_note)
            db.session.commit()
            flash('Note added!', category='success')
    return render_template("home.html", user=current_user)

@views.route('/ml-dashboard', methods=['GET'])
@login_required
def ml_dashboard():
    return render_template("ml_dashboard.html", user=current_user)

@views.route('/train-model', methods=['POST'])
@login_required
def train_model():
    try:
        # Get training data from the uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Separate features and target
        X = df.drop('target', axis=1)  # Assuming 'target' is the label column
        y = df['target']
        
        # Train the model
        metrics = ml_model.train_model(X, y)
        
        # Save the model
        model_path = os.path.join('website', 'static', 'models')
        os.makedirs(model_path, exist_ok=True)
        ml_model.save_model(os.path.join(model_path, 'trained_model.joblib'))
        
        return jsonify({
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get input data
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Load the model if not already loaded
        model_path = os.path.join('website', 'static', 'models', 'trained_model.joblib')
        if not hasattr(ml_model, 'model') or ml_model.model is None:
            ml_model.load_model(model_path)
        
        # Make predictions
        predictions, probabilities = ml_model.predict(df)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@views.route('/delete-note', methods=['POST'])
@login_required
def delete_note():  
    try:
        note_data = json.loads(request.data)
        note_id = note_data.get('noteId')
        
        if not note_id:
            return jsonify({'error': 'Note ID is required'}), 400

        note = Note.query.get(note_id)
        if note:
            if note.user_id == current_user.id:
                db.session.delete(note)
                db.session.commit()
                return jsonify({'message': 'Note deleted successfully'})
            return jsonify({'error': 'Unauthorized'}), 403
        return jsonify({'error': 'Note not found'}), 404
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Server error'}), 500
