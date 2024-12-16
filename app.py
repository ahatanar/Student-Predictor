from flask import Flask, render_template, request, jsonify
from services import predict_score

# from models import load_models
# from services import preprocess_input, predict_grade, get_model_metrics

app = Flask(__name__)

# models = load_models()

@app.route('/')
def home():
    """Render the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions."""
    try:
        # Parse user input
        user_input = request.form.to_dict()

        # Extract model name and remove it from user input
        model_name = user_input.pop('ml_model')

        
        prediction = predict_score(model_name, user_input)

        prediction  = str(round(prediction, 2))
        return jsonify({'prediction': prediction})
    except Exception as e:
        # Handle errors
        return jsonify({'error': str(e)}), 400

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

# @app.route('/metrics')
# def metrics():
#     """Display model metrics."""
#     metrics = get_model_metrics()
#     return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)