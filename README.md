# Deploying ML Models with Kubernetes for model scoring

Arhum Zafar // Fall 2021

<br>

A typical workflow for deploying a machine learning (ML) model into production -- *usually models created/trained using the sklearn, XGBoost, tensorflow, or keras* packages that are ready to make predictions on new data -- is to deploy these models as [REST API micro-services](https://restfulapi.net/), usually hosted from [Docker](https://www.docker.com/) containers. After the model has been containerized, it can then be deployed to a cloud environment where it can then be able to handle anything it needs to be continuously available for service -- i.e. scaling, load balancing, drift-detection, and more.
<br>
<br>
The widely-known problem is that configuration details for cloud deployment are specific to each target cloud provider(s) -- *the process, topology, syntax, etc. for Microsoft Azure is not the same as AWS, which again is different for Google Cloud Platform (GCP)*. This means that there’s a ton of prerequisite knowledge that needs to be gained for each cloud provider, which is just too difficult and time consuming. Additionally, this would also mean writing an unnecessary amount of YAML, which is purely [inappropriate software practice](https://news.ycombinator.com/item?id=21101695). All in all, it is almost impossible to test your deployment strategies locally -- especially at scale -- as networking issues can be hard to debug.
<br>
<br>
[Kubernetes (K8s)](https://kubernetes.io/) is a contained-based orchestration platform that addresses the above issues in a seamless manner. To keep things short, it provides a method to define all of a microservices application topologies, as well as their service-level requirements for ensuring continuous availability. What’s most important is that **K8s is cloud provider agnostic**, it can be ran on-prem, and even locally on your machine -- all you need is a Kubernetes cluster. Additionally, Kubernetes’ [kubeflow](https://www.kubeflow.org/), K8’s ML deployment platform, has been the [most popular ML deployment framework](https://blog.kubeflow.org/kubeflow-continues-to-move-to-production#:~:text=Similar%20to%20previous%20years%2C%20Kubeflow,being%20widely%20deployed%20as%20well.&text=Although%20the%20usage%20patterns%20for,components%20in%20their%20ML%20Platform.) over the past 2 years and is becoming a de facto name in the modern ML toolkit. <br>
This README was written to be followed alongside the code in this repo. You’ll find the Python modules, Docker config files, and the K8s instructions that will guide you in taking your ML model and deploying it as a production-level REST API for model scoring. **Please note:** this is not a thorough guide on using K8s and Docker for ML -- *Although I encourage anyone working with ML to learn the two, this repo isn’t the place to do it.*
<br>
<br>
We go through the ML deployment process using two approaches:

1.  A principle approach using containerization (Docker) and K8s.
2.  A deployment using the [Seldon-Core](https://www.seldon.io/) K8s framework -- A native Kubernetes framework that fast-tracks the deployment process for ML.
<br>

### Containerizing a simple ML model using Docker and Flask
Let’s begin by showing how the Python module works and how it integrates with the Dockerfile. The APi is found in `api.py`, alongside the `Dockerfile`, both are inside the `REST-api-with-flask` directory. The contents are shown below:

```bash
py-flask-ml-score-api/
| Dockerfile
| Pipfile
| Pipfile.lock
| api.py
```

### Using Flask in our API module
The given API module uses [Flask](https://flask.palletsprojects.com/en/2.0.x/) to define a web service (app), a function (score) that executes in response to the app.route function. The corresponding code is shown below:
```python
from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def score():
    features = request.json['X']
    return make_response(jsonify({'score': features}))
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
If we run this locally -- by calling `python run api.py` -- we can read our function at `http://localhost:5000/score`. This function takes the data sent to it as JSON (de-serialized as a Python dictionary) and makes it available as the `request` variable, which then returns a response in JSON form.<br>
 
In the example, an array of features, `X`, will be passed to a ML model. In this case, the model is a simple identity function -- chosen for demonstration purposes. In a similar manner, we could have also loaded an sklearn or keras model and passed the data to the appropriate `predict` method, which then returns the score for the features in JSON format. <br>

### Defining the Docker Image

A `Dockerfile` is the configuration file used for Docker. It allows you to define the contents and configure the operation of a Docker container. The syntax, when not executed as a container, is referred to as a **Docker Image**.

  

```docker

FROM python:3.8-slim
WORKDIR /usr/src/app
COPY . .
RUN pip install pipenv
RUN pipenv install
EXPOSE 5000
CMD ["pipenv", "run", "python", "api.py"]
```
In the above `Dockerfile` we will:

-   Start by using a pre-configured Docker image.
-   Next, we copy the contents of the ‘REST-api-with-flask` local directory to a directory on the image called `/usr/src/app`.
-   Use `pip` to install Pipenv for dependency management.
-   Following the installation, use Pipenv to install the dependencies described in `Pipfile.lock` into a virtual environment on the image.
-   Configure port `5000` to be exposed to the public. <br>

Building a custom Docker Image and asking the Docker daemon to run it will launch the ML model scoring API on port 5000. This is as if it were on a dedicated virtual machine. You can learn more about this [here](https://docs.docker.com/get-started/). <br>
### Building a Docker Image
Assuming that Docker is running locally and the client is logged into an account on [DockerHub](https://hub.docker.com/). Open up your terminal/CLI in the project's root directory and run this to build the image.
```bash
docker build --tag arhumzafar/test-ml-score-api REST-api-with-flask
```
Above, 'arhumzafar' is the name of the DockerHub account that I'll push the image to. <br>
#### Testing
To test that the image can be used to create a Docker container that operates correctly:

```docker
docker run --rm --name test-api -p 5000:5000 -d arhumzafar/test-ml-score-api
```

We then check that the newly created container is listed as running:
```docker
docker ps
```
And then test the exposed API endpoint using,

```bash
curl http://localhost:5000/score \
--request POST \
--header "Content-Type: application/json" \
--data '{"X": [1, 2]}'
```

Where you should expect a response along the lines of,

```json
{"score":[1,2]}
```

All our test model does is return the input data - i.e. it is the identity function. Only a few lines of additional code are required to modify this service to load a SciKit Learn model from disk and pass new data to it's 'predict' method for generating predictions. Now that the container has been confirmed as operational, we can stop it,
```bash
docker stop test-api
```
Where we can now see that our chosen naming convention for the image is intrinsically linked to our target image registry (you will need to insert your own account ID where required).

## Installing Kubernetes for Local Development and Testing

  

There are two options for installing a single-node Kubernetes cluster that is suitable for local development and testing: via the [Docker Desktop](https://www.docker.com/products/docker-desktop) client, or via [Minikube](https://github.com/kubernetes/minikube).

  

### Installing Kubernetes via Docker Desktop


If you have been using Docker on a Mac, then the chances are that you will have been doing this via the Docker Desktop application. If not (e.g. if you installed Docker Engine via Homebrew), then Docker Desktop can be downloaded [here](https://www.docker.com/products/docker-desktop). Docker Desktop now comes bundled with Kubernetes, which can be activated by going to `Preferences -> Kubernetes` and selecting `Enable Kubernetes`. It will take a while for Docker Desktop to download the Docker images required to run Kubernetes, so be patient. After it has finished, go to `Preferences -> Advanced` and ensure that at least 2 CPUs and 4 GiB have been allocated to the Docker Engine, which are the the minimum resources required to deploy a single Seldon ML component.

  

To interact with the Kubernetes cluster you will need the `kubectl` Command Line Interface (CLI) tool, which will need to be downloaded separately. The easiest way to do this on a Mac is via Homebrew - i.e with `brew install kubernetes-cli`. Once you have `kubectl` installed and a Kubernetes cluster up-and-running, test that everything is working as expected by running,

```bash
kubectl cluster-info
```
