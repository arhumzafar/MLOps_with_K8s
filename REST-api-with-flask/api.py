## Arhum Zafar -- delete

"""
api.py
~~~~~~

Here we define the REST API for the ML model. It simply be used for testing Docker and Kubernetes.

To test locally, go to your command line interface (CLI), using 'python api.py'
to start the service. Next, in another terminal window, enter:

```
curl http://localhost:5000/score \
--request POST \
--header "Content-Type: application/json" \
--data '{"X": [1, 2]}'
```

To test the API.
"""


from typing import Iterable

from flask import Flask, jsonify, make_response, request, Response

app = Flask(__name__)


@app.route('/score', methods=['POST'])
def score() -> Response:
    """Score data using an machine learning model.

    The API endpoint expects a JSON payload with a field called 'X'
    containing an iterable sequence of features to send to the model. 
    The data is then parsed into a Python dictionary, where it's then
    made available via 'request.json'.

    If 'X' cannot be found in the data, or cannot be parsed, an exception
    will be raised. Otherwise, it will return a JSON payload with a field 
    titled 'score' containing the model's prediction, which can then be compared
    to the other model(s), if needed.
    """

    try:
        features = request.json['X']
        prediction = model_predict(features)
        return make_response(jsonify({'score': prediction}))
    except KeyError:
        raise RuntimeError('"X" cannot be be found in JSON payload.')


def model_predict(x: Iterable[float]) -> Iterable[float]:
    """Dummy prediction function."""
    return x


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
