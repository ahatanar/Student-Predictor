from flask import Flask, render_template, request, jsonify
from services import predict_score


app = Flask(__name__)


@app.route('/')
def home():
    """Render the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions."""
    try:
        user_input = request.form.to_dict()

        model_name = user_input.pop('ml_model')

        
        prediction = predict_score(model_name, user_input)

        prediction  = str(round(prediction, 2))
        response = jsonify({'prediction': prediction})
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/info')
def info():
    return render_template('info.html')

# @app.route('/metrics')
# def metrics():
#     """Display model metrics."""
#     metrics = get_model_metrics()
#     return render_template('metrics.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)