from flask import Flask
import os
from joblib import dump, load

app = Flask(__name__)

@app.route('/where', methods=['POST'])
def recommend():
    # Parse the parameters you need
    token = request.form.get('token', None)
    command = request.form.get('command', None)
    text = request.form.get('text', None)

    if text == None or text == '':
        return "Please enter a question."

    model = load("naivebayes.joblib")
    ret = model.predict([text])
    return '#' + ret[0]

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)