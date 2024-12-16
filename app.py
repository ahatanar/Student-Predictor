from flask import Flask, render_template, request, jsonify
from models import load_models
from services import preprocess_input, predict_grade, get_model_metrics

app = Flask(__name__)

models = load_models()

@app.route('/')
def home():
    """Render the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions."""
    # Parse user input
    user_input = request.form.to_dict()
    
    # Preprocess and predict
    processed_input = preprocess_input(user_input)
    prediction = predict_grade(models['mlp_model'], processed_input)
    
    return jsonify({'prediction': round(prediction, 2)})

@app.route('/metrics')
def metrics():
    """Display model metrics."""
    metrics = get_model_metrics()
    return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)