# Deploying ML Models with Kubernetes for model scoring

Arhum Zafar // Fall 2021

<br>

A typical workflow for deploying a machine learning (ML) model into production -- *usually models created/trained using the sklearn, XGBoost, tensorflow, or keras* packages that are ready to make predictions on new data -- is to deploy these models as [REST API micro-services](https://restfulapi.net/), usually hosted from [Docker](https://www.docker.com/) containers. After the model has been containerized, it can then be deployed to a cloud environment where it can then be able to handle anything it needs to be continuously available for service -- i.e. scaling, load balancing, drift-detection, and more.
<br>
<br>
The widely-known problem is that configuration details for cloud deployment are specific to each target cloud provider(s) -- *the process, topology, syntax, etc. for Microsoft Azure is not the same as AWS, which again is different for Google Cloud Platform (GCP)*. This means that there’s a ton of prerequisite knowledge that needs to be gained for each cloud provider, which is just too difficult and time consuming. Additionally, this would also mean writing an unnecessary amount of YAML, which [isn't fun at all](https://news.ycombinator.com/item?id=21101695). All in all, it is almost impossible to test your deployment strategies locally -- especially at scale -- as networking issues can be hard to debug.
<br>
<br>
[Kubernetes (K8s)](https://kubernetes.io/) is a contained-based orchestration platform that addresses the above issues in a seamless manner. To keep things short, it provides a method to define all of a microservices application topologies, as well as their service-level requirements for ensuring continuous availability. What’s most important is that **K8s is cloud provider agnostic**, it can be ran on-prem, and even locally on your machine -- all you need is a Kubernetes cluster. Additionally, Kubernetes’ [kubeflow](https://www.kubeflow.org/), K8’s ML deployment platform, has been the [most popular ML deployment framework](https://blog.kubeflow.org/kubeflow-continues-to-move-to-production#:~:text=Similar%20to%20previous%20years%2C%20Kubeflow,being%20widely%20deployed%20as%20well.&text=Although%20the%20usage%20patterns%20for,components%20in%20their%20ML%20Platform.) over the past 2 years and is becoming a de facto name in the modern ML toolkit.
<br>
<br>
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
REST-api-with-flask/
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

This will return something that similar to what's below:
`Kubernetes master is running at https://kubernetes.docker.internal:6443
KubeDNS is running at https://kubernetes.docker.internal:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy`

`To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.`
<br>
<br>

### Installing Kubernetes via Minikube

Using Mac OS X, follow the below steps to get Minikube up and running:
- make sure the [Homebrew](https://brew.sh) package manager for OS X is installed; then,
- install VirtualBox using, `brew cask install virtualbox` (you may need to approve installation via OS X System Preferences); and then,
- install Minikube using, `brew cask install minikube`.

To start the test cluster run,

```bash
minikube start --memory 4096
```

Where we have specified the minimum amount of memory required to deploy a single Seldon ML component. **Be patient** - Minikube may take a while to start. To test that the cluster is operational run,

```bash
kubectl cluster-info
```

Where `kubectl` is the standard Command Line Interface (CLI) client for interacting with the Kubernetes API (which was installed as part of Minikube, but is also available separately).

### Deploying the Containerized ML Model to K8s
To begin to test our model scoring API, we first start by deploying the service within a K8s [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod-overview/), whose rollout is managed by a [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/), which in in-turn creates a [ReplicaSet](https://kubernetes.io/docs/concepts/workloads/controllers/replicaset/) - a Kubernetes resource that ensures a minimum number of pods (or replicas), running our service are operational at any given time. This is achieved with,

```bash
kubectl create deployment test-ml-score-api --image=alexioannides/test-ml-score-api:latest
```

To check on the status of the deployment run,

```bash
kubectl rollout status deployment ml-score-api
```

And to see the pods that is has created run,

```bash
kubectl get pods
```

It is possible to use [port forwarding](https://en.wikipedia.org/wiki/Port_forwarding) to test an individual container without exposing it to the public internet. To use this, open a separate terminal and run (for example),

```bash
kubectl port-forward ml-score-api-szd4j 5000:5000
```
Now, `ml-score-api-szd4j` is the name of the pod currently active on the cluster, as determined from the `kubectl get pods` command. Then from your original terminal, to repeat our test request against the same container running on Kubernetes run,

```bash
curl http://localhost:5000/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": [1, 2]}'
```

To expose the container as a (load balanced) [service](https://kubernetes.io/docs/concepts/services-networking/service/) to the outside world, we have to create a Kubernetes service that references it. This is achieved with the following command,

```bash
kubectl expose deployment ml-score-api --port 5000 --type=LoadBalancer --name ml-score-api-lb
```

If you are using Docker Desktop, then this will automatically emulate a load balancer at `http://localhost:5000`. To find where Minikube has exposed its emulated load balancer run,

```bash
minikube service list
```
Next, we test the application -- using Docker Desktop,
```bash
curl http://localhost:5000/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": [1, 2]}'
```

Upon it's creation, one thing that I've learned is that neither Docker or Minikube setup a load balancer -- *which would be vital if this was done on the cloud*.

### Getting Started & Configuring Multi-node Cluster on the Google Cloud Platform.

To perform a service on a real-world Kubernetes cluster with more resources than those available on a laptop, the easiest way is to use a managed Kubernetes platform from a cloud provider. We will use Kubernetes Engine on [Google Cloud Platform (GCP)](https://cloud.google.com).

Before anything else, sign up for an account and create a project specifically for this repo. After that, ensure that the GCP SDK is installed locally.  
```bash
brew cask install google-cloud-sdk
```

Or by downloading an installation image [directly from GCP](https://cloud.google.com/sdk/docs/quickstart-macos). Note, that if you haven't already installed Kubectl, then you will need to do so now, which can be done using the GCP SDK,

```bash
gcloud components install kubectl
```

We then need to initialise the SDK,

```bash
gcloud init
```

Which will open a browser and guide you through the necessary authentication steps. Make sure you pick the project you created, together with a default zone and region (if this has not been set via Compute Engine -> Settings).

### Initializing a K8s cluster
Using the GCP UI, find the Kubernetes Engine to get the Kubernetes API to start-up. From the CLI, we enter:
```bash
gcloud container clusters create k8s-test-cluster --num-nodes 3 --machine-type g1-small
```
It's natural for this step to take a couple minutes -- especially if this is your first time using K8s via GCP. Take note that this will automatically switch your `kubectl` context to point to the cluster on GCP, as you will see if you run, `kubectl config get-contexts`. To switch back to the Docker Desktop client use `kubectl config use-context docker-desktop`.

### Launching the containerized ML model scorer on GCP

Run the following commands in sequence:
```bash
kubectl create deployment ml-score-api --image=arhumzafar/ml-score-api:latest
kubectl expose deployment ml-score-api --port 5000 --type=LoadBalancer --name ml-score-api-lb
```

But, to find the external IP address for the GCP cluster we will need to use,

```bash
kubectl get services
```

And then we can test our service on GCP - for example,

```bash
curl http://35.246.92.213:5000/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": [1, 2]}'
```
Or, we could again use port forwarding to attach to a single pod - for example,

```bash
kubectl port-forward ml-score-api-nl4sc 5000:5000
```

And then in a separate terminal,

```bash
curl http://localhost:5000/score \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"X": [1, 2]}'
```


If you are running both with Kubernetes locally and with a cluster on GCP, then you can switch Kubectl [context](https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/) from one cluster to the other, as follows,

```bash
kubectl config use-context docker-desktop
```

Where the list of available contexts can be found using,

```bash
kubectl config get-contexts
```
### Using YAML Files to Define and Deploy the ML Model Scoring Service
For now, we've been using Kubectl commands to define and deploy the model scoring service. However, that is only fine for demonstrative purposes. In practice, the standard way of defining entire Kubernetes deployments is with YAML files,  posted to the Kubernetes API. The `ml-score-with-flask.yaml` file in the `REST-api-with-flask` directory is an example of how our ML model scoring service can be defined in a single YAML file. 

This can now be deployed using a single command,

```bash
kubectl apply -f REST-api-with-flask/ml-score-with-flask.yaml
```

### Using Helm to simplify the process.
Writing YAML files for Kubernetes can get repetitive and hard to manage, especially if there is a lot of 'copy-paste' involved, when only a handful of parameters need to be changed from one deployment to the next,  but there is a 'wall of YAML' that needs to be modified. Enter [Helm](https://helm.sh//) - a framework for creating, executing and managing Kubernetes deployment templates. What follows is a very high-level demonstration of how Helm can be used to deploy our ML model scoring service - for a comprehensive discussion of Helm's full capabilities (and here are a lot of them), please refer to the [official documentation](https://docs.helm.sh). Seldon-Core can also be deployed using Helm and we will cover this in more detail later on.

### Installing Helm

As before, the easiest way to install Helm onto Mac OS X is to use the Homebrew package manager,

```bash
brew install kubernetes-helm
```

Helm relies on a dedicated deployment server, referred to as the 'Tiller', to be running within the same Kubernetes cluster we wish to deploy our applications to. Before we deploy Tiller we need to create a cluster-wide super-user role to assign to it, so that it can create and modify Kubernetes resources in any namespace. To achieve this, we start by creating a [Service Account](https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/) that is destined for our tiller. A Service Account is a means by which a pod (and any service running within it), when associated with a Service Accoutn, can authenticate itself to the Kubernetes API, to be able to view, create and modify resources. We create this in the `kube-system` namespace (a common convention) as follows,

```bash
kubectl --namespace kube-system create serviceaccount tiller
```

We then create a binding between this Service Account and the `cluster-admin` [Cluster Role](https://kubernetes.io/docs/reference/access-authn-authz/rbac/), which as the name suggest grants cluster-wide admin rights,

```bash
kubectl create clusterrolebinding tiller \
    --clusterrole cluster-admin \
    --serviceaccount=kube-system:tiller
```

We can now deploy the Helm Tiller to a Kubernetes cluster, with the desired access rights using,

```bash
helm init --service-account tiller
```

### Deploying with Helm

To create a fresh Helm deployment definition - referred to as a 'chart' in Helm terminology - run,

```bash
helm create NAME-OF-YOUR-HELM-CHART
```

This creates a new directory - e.g. `helm-ml-score-app` as included with this repository - with the following high-level directory structure,

```bash
helm-ml-score-app/
 | -- charts/
 | -- templates/
 | Chart.yaml
 | values.yaml
```

Briefly, the `charts` directory contains other charts that our new chart will depend on (we will not make use of this), the `templates` directory contains our Helm templates, `Chart.yaml` contains core information for our chart (e.g. name and version information) and `values.yaml` contains default values to render our templates with (in the case that no values are set from the command line).

The next step is to delete all of the files in the `templates` directory (apart from `NOTES.txt`), and to replace them with our own. We start with `namespace.yaml` for declaring a namespace for our app,

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: {{ .Values.app.namespace }}
```

Anyone familiar with HTML template frameworks (e.g. Jinja), will be familiar with the use of ``{{}}`` for defining values that will be injected into the rendered template. In this specific instance `.Values.app.namespace` injects the `app.namespace` variable, whose default value defined in `values.yaml`. Next we define a deployment of pods in `deployment.yaml`,

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: {{ .Values.app.name }}
    env: {{ .Values.app.env }}
  name: {{ .Values.app.name }}
  namespace: {{ .Values.app.namespace }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {{ .Values.app.name }}
  template:
    metadata:
      labels:
        app: {{ .Values.app.name }}
        env: {{ .Values.app.env }}
    spec:
      containers:
      - image: {{ .Values.app.image }}
        name: {{ .Values.app.name }}
        ports:
        - containerPort: {{ .Values.containerPort }}
          protocol: TCP
```

And the details of the load balancer service in `service.yaml`,

```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Values.app.name }}-lb
  labels:
    app: {{ .Values.app.name }}
  namespace: {{ .Values.app.namespace }}
spec:
  type: LoadBalancer
  ports:
  - port: {{ .Values.containerPort }}
    targetPort: {{ .Values.targetPort }}
  selector:
    app: {{ .Values.app.name }}
```

So here, what we have done, is split-out each component of the deployment details from `ml-score-with-flask.yaml` into its own file and then define template variables for each parameter of the configuration that is most likely to change from one deployment to the next. To test and examine the rendered template, without having to attempt a deployment, run,

```bash
helm install helm-ml-score-app --debug --dry-run
```

If you are happy with the results of the 'dry run', then execute the deployment and generate a release from the chart using,

```bash
helm install helm-ml-score-app --name test-ml-app
```

This will automatically print the status of the release, together with the name that Helm has ascribed to it (e.g. 'willing-yak') and the contents of `NOTES.txt` rendered to the terminal. To list all available Helm releases and their names use,

```bash
helm list
```

And to the status of all their constituent components (e.g. pods, replication controllers, service, etc.) use for example,

```bash
helm status test-ml-app
```

The ML scoring service can now be tested in exactly the same way as we have done previously (above). Once you have convinced yourself that it's working as expected, the release can be deleted using,

```bash
helm delete test-ml-app
```
### Building the Docker Image for use with Seldon

Seldon requires that the Docker image for the ML scoring service be structured in a particular way:

- the ML model has to be wrapped in a Python class with a `predict` method with a particular signature (or interface) - for example, in `MLScore.py` (deliberately named after the Python class contained within it) we have,

```python
class MLScore:
    """
    Model template. You can load your model parameters in __init__ from
    a location accessible at runtime
    """

    def __init__(self):
        """
        Load models and add any initialization parameters (these will
        be passed at runtime from the graph definition parameters
        defined in your seldondeployment kubernetes resource manifest).
        """
        print("Initializing")

    def predict(self, X, features_names):
        """
        Return a prediction.

        Parameters
        ----------
        X : array-like
        feature_names : array of feature names (optional)
        """
        print("Predict called - will run identity function")
        return X
```

- the `seldon-core` Python package must be installed (we use `pipenv` to manage dependencies as discussed above and in the Appendix below); and,
- the container starts by running the Seldon service using the `seldon-core-microservice` entry-point provided by the `seldon-core` package - both this and the point above can be seen the `DockerFile`,

```docker
FROM python:3.8-slim
COPY . /app
WORKDIR /app
RUN pip install pipenv
RUN pipenv install
EXPOSE 5000

# Define environment variable
ENV MODEL_NAME MLScore
ENV API_TYPE REST
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0

CMD pipenv run seldon-core-microservice $MODEL_NAME $API_TYPE --service-type $SERVICE_TYPE --persistence $PERSISTENCE
```

