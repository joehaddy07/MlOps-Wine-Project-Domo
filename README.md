wine-quality-mlops/
│
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yaml
│
├── data/
│   ├── raw/
│   │   └── wine.csv
│   │
│   └── processed/
│       └── README.md
│
├── notebooks/
│   └── wine-analysis.ipynb
│
├── src/
│   ├── main.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   └── mlflow/
│       ├── __init__.py
│       └── tracking.py
│
├── tests/
│   │
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── artifacts/
│   │
│   ├── models/
│   ├── plots/
│   └── reports/
│
├── app/
│   │
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
│
├── helm/
│   │
│   └── wine-quality/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
│
├── requirements.txt
│
├── requirements-dev.txt
│
├── Makefile
│
├── README.md
│
└── .gitignore


============================================================================
Portforwing 

kubectl port-forward svc/wine-app 8501:80
===========================================================================


Project Write-Up: End-to-End MLOps Pipeline for Wine Quality Prediction
One-line summary

Built and debugged a fully automated CI/CD pipeline that trains a wine quality prediction model, tracks it with MLflow, containerizes it, and deploys it to Kubernetes — triggered automatically on every push to main.

Project Overview

This project implements a production-style MLOps pipeline covering the full lifecycle of a machine learning application: data processing, model training, experiment tracking, containerization, and Kubernetes deployment — all automated through a single GitHub Actions workflow running on a self-hosted runner.

Repository structure:

wine-quality-mlops/
├── .github/workflows/mlops-pipeline.yaml   # CI/CD orchestration
├── src/                                     # Training pipeline (data, features, models, evaluation, MLflow tracking)
├── tests/                                   # Unit tests (pytest)
├── app/                                     # Streamlit inference UI + its own Dockerfile
├── helm/wine-quality/                       # Kubernetes deployment via Helm chart
├── artifacts/                               # Model outputs, plots, reports
└── notebooks/                               # Exploratory analysis

Tech stack: GitHub Actions, MLflow, Docker, Kind (Kubernetes-in-Docker), Helm, Streamlit, Python 3.12, pytest.

Architecture / Pipeline Flow
Push to main
   → Checkout code
   → Install Python + dependencies
   → Run unit tests (pytest)
   → Train model + log experiment to MLflow
   → Build Docker image (tagged with immutable git SHA)
   → Push image to Docker Hub
   → Load exact image into Kind cluster
   → Deploy via Helm upgrade --install
   → Verify deployed image matches expected SHA
   → Wait for rollout, verify pods/services
   → On failure: collect full Kubernetes diagnostics automatically

The pipeline is split into two GitHub Actions jobs: build-train (test, train, containerize, push) and deploy (Kubernetes/Helm), connected with needs: so deployment only runs after a successful build.

Key Design Decisions (talking points)

1. Immutable image tagging with git SHA, not latest Every image is tagged with ${{ github.sha }} rather than latest. This guarantees traceability between a specific commit and the exact container running in the cluster, and avoids the classic failure mode where a cluster silently keeps running a stale cached image because the tag never changed.

2. Hard verification gates, not just logging After Helm deploys, the pipeline queries the live deployment's image (kubectl get deployment -o jsonpath=...) and compares it against the expected SHA, failing the build (exit 1) on mismatch. This turns "deployment looks fine" into an actual automated assertion rather than a hopeful assumption.

3. Decoupled rollout wait from deploy helm upgrade --wait=false is used deliberately, followed by a separate kubectl rollout status --timeout=300s step. This gives cleaner diagnostics — if a rollout hangs, you know exactly which stage it stalled at, rather than getting a generic Helm timeout with no context.

4. Structured failure diagnostics The workflow includes a dedicated diagnostics block (if: failure()) that automatically dumps pod status, ReplicaSets, deployment description, Kubernetes events, current container logs, and previous (crashed) container logs. This was directly responsible for diagnosing every bug below — without --previous logs in particular, several of these root causes would have been invisible.

Debugging Deep-Dive (this is the strongest interview material)

The pipeline's initial version deployed "successfully" by CI's own metrics multiple times while the application was actually broken — a good illustration of why deploy success and application health are two different things that need separate verification.

Bug 1 — Model registry step was a no-op

The MLflow "verification" step in CI only printed a comment (echo "registration handled by src/main.py") — it didn't actually check anything. This meant a broken model registration or stage transition in the training code would pass CI silently. Fix approach discussed: replace the echo with a real MLflow Model Registry query that asserts the expected model version and stage exist, failing the build if not — same pattern as the image-verification gate.

Bug 2 — Wrong Docker build context, missing dependency

Pods were stuck in CrashLoopBackOff with ModuleNotFoundError: No module named streamlit. Root cause: the Dockerfile was built with the repo root as build context (docker build -f app/Dockerfile .), but COPY requirements.txt . inside the Dockerfile pointed at the root requirements.txt (used for training deps like MLflow/scikit-learn) rather than app/requirements.txt (which actually listed streamlit). The build succeeded with zero errors — it just installed the wrong dependency set. Fix: COPY app/requirements.txt . and COPY app/ ., scoping the image correctly to the app's own code and dependencies instead of pulling in the entire monorepo.

Bug 3 — Wrong entrypoint filename

After fixing the dependency issue, pods crashed again with Error: Invalid value: File does not exist: app.py. The actual file was named apps.py, not app.py — a naming mismatch between the file and the Dockerfile's CMD. Diagnosed with find app/ -name "*.py" to confirm the real filename rather than assuming. Fix: aligned the CMD to reference the correct filename.

Bug 4 — Kind networking gap (not a code bug at all)

Once pods reached 1/1 Running with clean Streamlit startup logs, the app still wasn't reachable in the browser. This wasn't an application bug — Kind (Kubernetes-in-Docker) runs cluster nodes as Docker containers, so a NodePort service doesn't automatically map to the host machine's network the way it would on a real cloud cluster. Diagnosis approach: verified pod health and Streamlit logs first (kubectl logs) to rule out an app-level issue, then used kubectl port-forward svc/wine-app 8501:80 to confirm the app itself worked, isolating the problem to cluster networking config rather than code. Permanent fix: recreating the Kind cluster with extraPortMappings in its config to forward the NodePort to the host.

The throughline across all four bugs: each one produced a visually similar symptom ("deployment failing" or "app not live"), but each had a completely different root cause spanning four different layers of the stack — CI verification logic, Docker build context, container entrypoint config, and cluster networking. Methodically isolating each layer (pod status → container logs → previous-container logs → app-level logs → network reachability) rather than guessing was what actually resolved it.

Security Hardening

Added a Trivy vulnerability scan as a CI gate between image build and push (--severity HIGH,CRITICAL, --ignore-unfixed), currently running in report-only mode (--exit-code 0) while establishing a baseline of existing findings, with a clear path to flipping it into a hard blocking gate (--exit-code 1) once triaged. This reflects a deliberate, staged rollout of a new security control rather than introducing a hard gate that could immediately break a pipeline that was still stabilizing.

Skills This Project Demonstrates
CI/CD design: multi-job GitHub Actions workflows, job dependencies, immutable versioning, verification gates
MLOps: MLflow experiment tracking and Model Registry, tying model artifacts to deployable containers
Containerization: Docker build contexts, multi-file COPY scoping, debugging silent dependency-resolution failures
Kubernetes: Helm chart deployment, rollout status monitoring, reading pod/deployment/event diagnostics, understanding Kind-specific networking behavior vs. real cloud clusters
Systematic debugging: methodically isolating root cause across CI, container, and cluster layers instead of pattern-matching to the first plausible explanation
Security-mindedness: proactively adding vulnerability scanning, with a sensible staged (report-then-enforce) rollout strategy
Possible Interview Talking Points / Questions to Prepare For
"Walk me through a production bug you debugged" → use the Bug 2/3/4 sequence; it's a genuinely good story of layered root-causing.
"How do you verify a Kubernetes deployment actually succeeded, not just that kubectl apply returned 0?" → the image-mismatch gate and rollout-status separation.
"How would you handle secrets in this pipeline?" → Docker Hub credentials and MLflow tracking URI are already pulled from GitHub Actions secrets, not hardcoded — good to mention explicitly if asked.
"What would you add next?" → real MLflow registry verification (Bug 1 fix), flipping Trivy to a hard gate, moving off Kind's manual port-forward workaround to a proper Ingress or LoadBalancer setup for anything beyond local dev.


============================================================================

============================================================================


End-to-End MLOps Wine Quality Prediction Platform
1. Project Summary

Project: End-to-End MLOps Pipeline for Wine Quality Prediction

Objective:
Build an automated machine-learning platform that takes a wine-quality dataset, trains a classification model, tracks experiments and models with MLflow, packages the prediction application into a Docker container, and automatically deploys the application to Kubernetes using Helm.

The entire workflow is triggered whenever code is pushed to the main branch.

One-minute interview explanation

"I built an end-to-end MLOps platform for wine quality prediction. The project starts with a Python machine-learning pipeline that loads and preprocesses the wine dataset, trains a classification model, evaluates it, and tracks the experiment with MLflow.

After the model is trained successfully, GitHub Actions builds a Docker image for a Streamlit inference application. Instead of using latest, I tag the image with the Git commit SHA, which gives me immutable versioning and allows me to trace a running container back to the exact source-code commit that produced it.

The image is pushed to Docker Hub and then loaded into a Kind Kubernetes cluster. Helm manages the Kubernetes deployment, including replicas, resources, services, and the image version. Finally, the pipeline verifies the Kubernetes rollout and collects diagnostics automatically if something fails.

One of the most valuable parts of the project was debugging several real deployment failures. I encountered missing Python dependencies inside the container, an incorrect Streamlit entrypoint, CrashLoopBackOff situations, and Kind networking issues. I diagnosed these by moving systematically from Kubernetes deployment status to pod status, container logs, previous container logs, ReplicaSets, and Kubernetes events.

So the project demonstrates not only machine learning, but the complete MLOps lifecycle: development, testing, experiment tracking, containerization, CI/CD, security scanning, Kubernetes deployment, observability, and troubleshooting."

2. Business / Technical Problem

The initial problem is simple:

Given chemical properties of a wine, predict whether the wine should be classified as good quality or bad quality.

The ML problem itself is only one part of the project.

A real production environment also needs to answer:

How was the model trained?
Which dataset was used?
Which code version produced the model?
What metrics did the model achieve?
Can another developer reproduce the training?
How do we package the model/application?
How do we deploy it?
How do we update the application?
How do we know the new deployment actually works?
What happens when the deployment fails?
How do we roll back?
How do we scan the container for vulnerabilities?

That is where the MLOps portion of the project becomes important.

3. High-Level Architecture

The architecture can be explained as:

                    Developer
                       |
                       | git push
                       v
                +---------------+
                |    GitHub     |
                |   Repository  |
                +-------+-------+
                        |
                        v
              +---------------------+
              |   GitHub Actions    |
              |   Self-hosted       |
              |      Runner         |
              +----------+----------+
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
      Pytest          ML Training     MLflow
                         |              |
                         |              |
                         +-------> Experiment
                                  Tracking /
                                  Registry
                         |
                         v
                  Docker Build
                         |
                         v
                  Trivy Scan
                         |
                         v
                    Docker Hub
                         |
                         v
                 Kind Kubernetes
                    Cluster
                         |
                         v
                      Helm
                         |
                         v
                  Deployment
                   3 Replicas
                         |
                         v
                   Streamlit App
                         |
                         v
                    End User

The important point to emphasize in an interview is:

The project does not stop at model training. It automates the path from source code to a running ML application.

4. Repository Structure
wine-quality-mlops/
│
├── .github/
│   └── workflows/
│       └── mlops-pipeline.yaml
│
├── data/
│   ├── raw/
│   │   └── wine.csv
│   │
│   └── processed/
│       └── README.md
│
├── notebooks/
│   └── wine-analysis.ipynb
│
├── src/
│   ├── main.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   │
│   └── mlflow/
│       ├── __init__.py
│       └── tracking.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── artifacts/
│   ├── models/
│   ├── plots/
│   └── reports/
│
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── helm/
│   └── wine-quality/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
│
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── README.md
└── .gitignore
How I would explain the structure

"I separated the repository into logical responsibilities. src contains the ML pipeline, tests contains automated tests, app contains the inference application and container configuration, and helm contains Kubernetes deployment configuration. This separation prevents the training environment and serving environment from becoming tightly coupled."

5. Data Layer

The raw dataset is located under:

data/raw/wine.csv

The data-loading logic is separated into:

src/data/load_data.py

The purpose of this separation is to avoid putting all of the ML logic into one notebook or one Python script.

A typical pipeline is:

Raw CSV
   |
   v
load_data.py
   |
   v
DataFrame
   |
   v
preprocessing.py
   |
   v
Features + Target
6. Exploratory Data Analysis

The project also contains:

notebooks/wine-analysis.ipynb

The notebook is used for exploratory analysis.

Typical activities include:

inspecting the dataset shape
viewing the first records
checking missing values
generating descriptive statistics
examining the distribution of wine quality
examining relationships between chemical properties and quality

For example:

wine_dataset.shape
wine_dataset.head()
wine_dataset.isnull().sum()
wine_dataset.describe()

And visualization:

sns.countplot(x="quality", data=wine_dataset)

The important interview distinction is:

The notebook is primarily for exploration and analysis, while the production training pipeline lives under src.

That's an important MLOps principle.

7. Machine Learning Model

The project uses a classification approach.

The dataset is divided into:

X = Features
Y = Target

Then:

Dataset
   |
   +---- Training Data
   |
   +---- Testing Data

using:

train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=3
)

The model is trained using scikit-learn.

You explored both Logistic Regression and Random Forest during the project.

For an interview, you can explain the model choice like this:

"I initially experimented with Logistic Regression because it provides a strong baseline for binary classification and is easy to interpret. I also explored Random Forest because wine-quality relationships can be nonlinear and involve interactions between multiple chemical properties. Random Forest can capture those nonlinear relationships without requiring me to manually engineer complex interaction terms."

8. Why Random Forest?

This is a likely interview question.

Logistic Regression

Logistic Regression essentially learns a weighted relationship:

features → weighted combination → probability → class

It works well when the decision boundary is relatively simple.

Random Forest

Random Forest builds multiple decision trees.

                 Dataset
                    |
       +------------+------------+
       |            |            |
       v            v            v
     Tree 1       Tree 2       Tree 3
       |            |            |
       +------------+------------+
                    |
                    v
                  Voting
                    |
                    v
                Prediction

Advantages:

captures nonlinear relationships
captures feature interactions
generally doesn't require feature scaling
robust to noise
provides feature importance
works well for tabular datasets

For this project:

"Because this is structured/tabular data with potentially nonlinear relationships between chemical properties and wine quality, Random Forest is a reasonable model to evaluate against a simpler Logistic Regression baseline."

Don't say:

"Random Forest is always better."

Instead say:

"I compare models using validation metrics rather than assuming one algorithm is automatically superior."

That sounds much stronger in an interview.

9. Model Evaluation

The project uses:

accuracy_score()

to evaluate classification performance.

Conceptually:

Accuracy =
Correct Predictions
-------------------
Total Predictions

For a more production-oriented system, I would also evaluate:

precision
recall
F1-score
confusion matrix
ROC-AUC where appropriate

because accuracy alone can be misleading when classes are imbalanced.

10. MLflow

MLflow provides experiment tracking and model lifecycle management.

The training process can record things such as:

Experiment
    |
    +-- Parameters
    |
    +-- Metrics
    |
    +-- Model
    |
    +-- Artifacts
    |
    +-- Dataset information

For example:

Wine Quality Experiment


Run 1
 ├── Model: Random Forest
 ├── Parameters
 ├── Accuracy
 ├── Model artifact
 └── Plots


Run 2
 ├── Model: Logistic Regression
 ├── Parameters
 ├── Accuracy
 └── Model artifact
Why MLflow?

Without MLflow, you could end up with:

model_final.pkl
model_final_v2.pkl
model_final_really_final.pkl
model_final_really_final_2.pkl

MLflow gives you structured experiment tracking.

11. MLflow + MLOps

A strong interview answer is:

"MLflow gives me reproducibility and traceability around the machine-learning portion of the system, while GitHub Actions gives me automation around the software delivery lifecycle."

Think of it as:

Git
 |
 | source-code version
 v
GitHub Actions
 |
 | pipeline version
 v
MLflow
 |
 | model/experiment version
 v
Docker
 |
 | application version
 v
Kubernetes
 |
 | deployment version
 v
Running application

This gives you traceability across the entire lifecycle.

12. GitHub Actions CI/CD

The workflow is triggered by:

on:
  push:
    branches:
      - main

Therefore:

Developer
    |
    | git push
    v
main
    |
    v
GitHub Actions

The workflow is split into two jobs.

Job 1 — build-train
Checkout
   ↓
Install Python
   ↓
Install Dependencies
   ↓
Run pytest
   ↓
Train Model
   ↓
MLflow Tracking
   ↓
Docker Login
   ↓
Docker Build
   ↓
Docker Push
Job 2 — deploy
Checkout
   ↓
Verify Docker
   ↓
Verify Kind
   ↓
Configure kubectl
   ↓
Load Image
   ↓
Helm Upgrade
   ↓
Rollout Status
   ↓
Verify Pods
   ↓
Verify Services

The jobs are connected with:

needs: build-train

Therefore deployment does not start if the build/test/training stage fails.

13. Why Use a Self-Hosted Runner?

The workflow uses:

runs-on: self-hosted

This was important because the deployment environment includes a local Kind cluster.

The runner has access to:

Docker
Kind
kubectl
Helm

Therefore the workflow can directly interact with:

joe-cluster

A good interview response:

"I used a self-hosted GitHub Actions runner because the Kubernetes target was a local Kind cluster running in my infrastructure. A GitHub-hosted runner would not automatically have access to that local cluster."

14. Docker Containerization

The Streamlit application is packaged as a Docker image.

The image contains:

Python
   +
Application dependencies
   +
Streamlit application
   +
Model/application artifacts

The Dockerfile installs dependencies and starts Streamlit on:

8501

The container exposes:

EXPOSE 8501

and runs:

streamlit run ...
15. The Important Docker Bug You Debugged

One of the strongest interview stories is:

CrashLoopBackOff
        |
        v
kubectl logs
        |
        v
No module named streamlit

At first glance, Kubernetes appeared to be the problem.

It wasn't.

The actual problem was the Docker image.

The Dockerfile was using the wrong requirements file.

The repository had different dependency responsibilities:

requirements.txt
    ↓
ML/training environment


app/requirements.txt
    ↓
Streamlit inference application

The container needed Streamlit, but the image wasn't installing the application's requirements correctly.

Root cause

The image successfully built because Docker had no reason to consider the missing Python package an error.

The failure happened only when the container executed:

python -m streamlit

which resulted in:

No module named streamlit
Lesson

A successful Docker build does not necessarily mean the application inside the image can actually start.

That's a very good interview point.

16. Second Container Bug — Wrong Entrypoint

Another deployment failure involved the application filename.

The Docker command expected:

app.py

but the actual file was:

apps.py

This produced an application startup failure.

You diagnosed it rather than guessing by checking the actual repository contents.

The lesson:

Container startup configuration is part of the application contract. The Dockerfile's CMD must match the actual application structure.

17. Kubernetes

The application is deployed to a Kind Kubernetes cluster.

The deployment uses:

3 replicas

Conceptually:

                 Kubernetes Service
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Pod 1          Pod 2          Pod 3
        1/1            1/1            1/1

This provides multiple application instances.

18. Why Three Replicas?

You can answer:

"I used three replicas to demonstrate Kubernetes workload management and provide basic redundancy. If one pod fails, Kubernetes can maintain the desired replica count by creating another pod."

For a production system, the exact replica count would depend on:

traffic
resource consumption
availability requirements
autoscaling
cost
19. Helm

Instead of hardcoding Kubernetes manifests everywhere, the project uses Helm.

For example:

replicaCount: 3


image:
  repository: joehaddy/wine-streamlit-app
  tag: "..."
  pullPolicy: IfNotPresent

This makes the deployment configurable.

The pipeline executes:

helm upgrade --install \
  wine-app \
  ./helm/wine \
  --set image.repository=... \
  --set image.tag=${IMAGE_TAG} \
  --set replicaCount=3

Therefore the CI/CD pipeline can dynamically deploy a new image version.

20. The Most Important Versioning Decision

Initially you were using:

latest

This creates a problem.

Suppose:

latest → Version A

Then you build Version B:

latest → Version B

Kubernetes may still have the old image cached.

You don't have a reliable deployment identity.

Instead, the pipeline uses:

IMAGE_TAG: ${{ github.sha }}

For example:

9c2b34dcc9e8fdc2d7c13c51fcfc4bfefcd4d88d

Now:

Git Commit
      |
      v
Docker Image
      |
      v
Kubernetes Deployment

Everything is traceable.

21. Immutable Deployment

This gives you:

Commit A
   ↓
Image A
   ↓
Deployment A


Commit B
   ↓
Image B
   ↓
Deployment B

Instead of:

Commit A ──┐
           ├──> latest
Commit B ──┤
           └──> ????

This is one of the strongest design decisions in your project.

22. Why IfNotPresent Works Here

Your Helm values use:

pullPolicy: IfNotPresent

Normally you might think:

"Shouldn't we use Always in CI/CD?"

With immutable SHA tags, IfNotPresent is reasonable for your Kind setup because:

Commit A → SHA-A
Commit B → SHA-B
Commit C → SHA-C

Each deployment references a different tag.

Therefore Kubernetes doesn't need to pull SHA-B when it already has SHA-A.

And your workflow explicitly does:

kind load docker-image ...

which loads the exact image into the Kind cluster.

This is a much cleaner approach than repeatedly deploying:

latest
23. Kubernetes Rollout Verification

The pipeline doesn't simply execute Helm and assume success.

It runs:

kubectl rollout status \
  deployment/wine-app \
  --timeout=300s

This verifies that Kubernetes actually completes the rollout.

For example:

Deployment desired: 3
Updated:             3
Available:           3
Ready:               3

If instead you get:

1 out of 3 new replicas have been updated

the pipeline fails.

That's exactly what happened during development.

24. CrashLoopBackOff Debugging

One of your strongest real-world debugging experiences was:

GitHub Actions
      |
      v
Helm deployment
      |
      v
Rollout timeout
      |
      v
kubectl get pods
      |
      v
CrashLoopBackOff

You then used:

kubectl logs <pod>

and:

kubectl logs <pod> --previous

The second command is particularly important for containers that have already crashed and restarted.

You eventually found:

/usr/local/bin/python:
No module named streamlit

This allowed you to move the investigation from Kubernetes into Docker.

25. Kubernetes Events

You also used:

kubectl get events --sort-by=.lastTimestamp

This showed events such as:

Created container
Started container
Back-off restarting failed container

This is useful because Kubernetes events provide cluster-level information that application logs don't necessarily show.

26. ReplicaSets

You used:

kubectl get rs -l app=wine-app

This showed multiple versions:

wine-app-56fbb4b798
wine-app-5c85bcf748
wine-app-65c5668bb5
wine-app-86648bc654
...

with different image SHA values.

This is actually a great demonstration of how Kubernetes RollingUpdates work.

For example:

Old ReplicaSet
      |
      | scale down
      v
New ReplicaSet
      |
      | scale up
      v
New Pods
27. Kubernetes RollingUpdate

The deployment uses Kubernetes' RollingUpdate strategy.

Conceptually:

Version A


Pod A
Pod A
Pod A


       ↓


Pod A
Pod A
Pod B


       ↓


Pod A
Pod B
Pod B


       ↓


Pod B
Pod B
Pod B

The goal is to replace the old version gradually instead of destroying all replicas at once.

28. Why the Rollout Was Timing Out

The important thing is that the GitHub Actions error:

deployment "wine-app" exceeded its progress deadline

was not itself the root cause.

It was a symptom.

The actual root cause was:

Pod
 ↓
Container starts
 ↓
Python executes
 ↓
Streamlit missing
 ↓
Process exits
 ↓
CrashLoopBackOff
 ↓
Pod never becomes Ready
 ↓
Deployment rollout cannot complete
 ↓
kubectl rollout status times out

This is an excellent interview explanation.

29. Kind Networking

You also discovered an important distinction.

A pod being:

1/1 Running

doesn't necessarily mean you can access the application from your host browser.

Kind runs Kubernetes nodes as Docker containers.

Therefore networking behaves differently from something like an AWS EKS cluster with an external LoadBalancer.

You used:

kubectl port-forward svc/wine-app 8501:80

to prove that the application itself was healthy.

That isolated the problem:

Application → healthy
Pod → healthy
Service → healthy
Host → cannot directly reach NodePort

That is an excellent example of layered troubleshooting.

30. Security — Trivy

The pipeline also includes Trivy scanning.

The scan focuses on:

HIGH
CRITICAL

vulnerabilities.

During the initial stabilization period, the scan was configured in report-only mode:

exit-code 0

rather than immediately blocking the pipeline.

The strategy was:

Phase 1
Report vulnerabilities


        ↓


Phase 2
Review findings


        ↓


Phase 3
Remediate


        ↓


Phase 4
Make scan blocking

This is a sensible CI/CD security strategy because introducing a hard gate before understanding the baseline can unnecessarily break delivery.

31. Secrets Management

Sensitive values are not hardcoded.

For example:

${{ secrets.DOCKERHUB_USERNAME }}
${{ secrets.DOCKERHUB_TOKEN }}
${{ secrets.MLFLOW_TRACKING_URI }}

This is preferable to:

username: joehaddy
password: mypassword

because credentials remain outside the source code.

Interview answer:

"I use GitHub Actions Secrets for sensitive configuration such as Docker Hub credentials and the MLflow tracking URI. The workflow references them at runtime rather than committing them to Git."

32. Failure Diagnostics

A particularly useful improvement is automatic diagnostics when the deployment fails.

For example:

kubectl get pods
kubectl get rs
kubectl describe deployment wine-app
kubectl get events
kubectl logs ...
kubectl logs ... --previous

This turns:

Deployment failed

into something much more actionable:

Deployment failed


↓
Pod CrashLoopBackOff


↓
Container logs


↓
No module named streamlit


↓
Docker image dependency problem
33. What Makes This an MLOps Project?

This is an important interview question.

A pure ML project might look like:

Dataset
 ↓
Notebook
 ↓
Model
 ↓
Accuracy

Your project is:

Dataset
 ↓
Data processing
 ↓
Model training
 ↓
Evaluation
 ↓
MLflow
 ↓
Automated testing
 ↓
Docker
 ↓
Security scanning
 ↓
Docker Registry
 ↓
Helm
 ↓
Kubernetes
 ↓
Application
 ↓
Automated verification

Therefore you're demonstrating the operational lifecycle of ML software.

34. CI vs CD

Another common interview question.

Continuous Integration

Your CI portion includes:

Checkout
 ↓
Install dependencies
 ↓
pytest
 ↓
Train model
 ↓
Track experiment
 ↓
Build image
 ↓
Security scan
Continuous Deployment

Your CD portion includes:

Push image
 ↓
Load image
 ↓
Helm deployment
 ↓
Kubernetes rollout
 ↓
Verification

So:

CI validates and packages the software; CD delivers it to the runtime environment.

35. What Happens When a Developer Pushes Code?

This is probably the single most important sequence to memorize.

Say:

"When a developer pushes to main, GitHub Actions starts the pipeline."

Then:

1. Checkout source code


2. Install Python 3.12


3. Install dependencies


4. Run pytest


5. Train the model


6. Track the experiment with MLflow


7. Build the Streamlit Docker image


8. Tag the image with the Git SHA


9. Push the image to Docker Hub


10. Configure access to the Kind cluster


11. Load the exact SHA-tagged image into Kind


12. Helm upgrades the application


13. Kubernetes performs a rolling update


14. kubectl rollout status waits for readiness


15. Pipeline verifies pods/services/deployment


16. If anything fails, diagnostics are collected

That's your 90-second pipeline explanation.

36. How Would You Roll Back?

Because you use immutable image tags and Helm, rollback becomes much safer.

You can inspect releases:

helm history wine-app

Then rollback:

helm rollback wine-app <REVISION>

Alternatively, redeploy a known-good image SHA.

For example:

Current:


wine-streamlit-app:SHA-B


Problem detected.


Rollback:


wine-streamlit-app:SHA-A

This is another advantage of avoiding latest.

37. What Would You Improve?

This is a very important senior-level interview question.

You should not say:

"Nothing. The project is complete."

Instead:

1. Real MLflow Registry verification

Currently the workflow's verification should be upgraded from a simple informational message to an actual API/query assertion.

For example:

Pipeline
   |
   v
MLflow
   |
   +--> Verify experiment
   |
   +--> Verify run
   |
   +--> Verify metrics
   |
   +--> Verify model
2. Stronger model evaluation

Instead of relying primarily on accuracy:

Accuracy
Precision
Recall
F1
Confusion Matrix

could be evaluated.

3. Model quality gate

You could establish:

if accuracy < threshold:
      fail pipeline

or use an appropriate metric depending on the business objective.

This prevents a technically successful training run from deploying a poor model.

4. Trivy enforcement

Move from:

exit-code 0

to:

exit-code 1

after vulnerabilities are reviewed and remediated.

5. Production Kubernetes

Kind is excellent for local development and demonstrating Kubernetes automation.

For production, I would move toward:

AWS EKS

or another managed Kubernetes platform.

6. Ingress / LoadBalancer

Instead of relying on:

kubectl port-forward

a production architecture would use:

Ingress
     ↓
Service
     ↓
Pods

or a cloud LoadBalancer.

7. Autoscaling

Add:

Horizontal Pod Autoscaler

so replicas can respond to workload.

8. Observability

Add:

Prometheus
Grafana

for metrics and dashboards, and potentially centralized logging.

38. Interview Question: "What Was Your Hardest Problem?"

I would answer this one using your CrashLoopBackOff incident.

Strong answer

"The hardest problem was a Kubernetes deployment that continuously timed out during rollout. Initially GitHub Actions only reported that the deployment had not completed, so I didn't assume Kubernetes itself was broken.

I checked the deployment and saw that only one of three new replicas was being updated. I then checked the pods and found CrashLoopBackOff. I inspected the container logs and previous container logs and found No module named streamlit.

I traced that back to the Docker image. The application had its own requirements file, but the Docker build was installing the wrong dependency set. The Docker image could build successfully because Docker doesn't validate whether the application will successfully start. The failure only appeared at runtime.

After correcting the Dockerfile and rebuilding the image with a new Git SHA, Kubernetes received the new immutable image. I then encountered an entrypoint filename mismatch and corrected that as well.

The experience taught me to troubleshoot from the outside in: deployment → ReplicaSet → pod → container → application logs, instead of immediately changing Kubernetes configuration."

That's a very strong DevOps/MLOps interview story.

39. Interview Question: "Why Didn't You Just Use latest?"

Answer:

"I deliberately moved away from latest because it doesn't provide reliable version traceability. With Git SHA tags, every image corresponds to one exact commit. That means I can identify exactly which source code produced a container, avoid ambiguity caused by mutable tags, and roll back to a known-good image."

40. Interview Question: "Why Helm?"

Answer:

"Helm gives me parameterized Kubernetes deployments. Instead of hardcoding image repositories, tags, replica counts, resources, and service configuration into separate manifests, I can define defaults in values.yaml and override deployment-specific values from CI/CD."

For example:

--set image.tag=${IMAGE_TAG}

means the pipeline can dynamically deploy the exact image produced by that commit.

41. Interview Question: "Why Kubernetes?"

Answer:

"Kubernetes gives me declarative application management. I define the desired state, such as three replicas running a specific image, and Kubernetes continuously works to maintain that state. It also provides rolling updates, self-healing, service discovery, resource management, and scaling capabilities."

42. Interview Question: "Why Docker?"

Answer:

"Docker packages the application and its dependencies into a consistent runtime environment. This eliminates the common situation where the application works on my development machine but fails in the deployment environment because of different Python packages or system dependencies."

43. Interview Question: "Why MLflow?"

Answer:

"MLflow addresses the reproducibility and lifecycle-management side of machine learning. It allows me to track experiments, parameters, metrics, artifacts, and models rather than relying on manually saved files and notebooks."

44. Interview Question: "What Happens If a Pod Crashes?"

Kubernetes detects that the actual state differs from the desired state.

If:

Desired:
3 replicas


Actual:
2 healthy replicas

Kubernetes attempts to create another pod.

This is one of the reasons Kubernetes is valuable for application deployment.

However, your debugging demonstrated an important limitation:

Kubernetes can restart a broken application, but it cannot fix an application that is fundamentally misconfigured.

If the image doesn't contain Streamlit:

Pod starts
 ↓
Python fails
 ↓
Container exits
 ↓
Kubernetes restarts it
 ↓
Python fails again

which results in:

CrashLoopBackOff
45. The Most Important Lessons From the Project

I would emphasize these five lessons during interviews.

Lesson 1 — Automation

Don't manually train, build, push, and deploy.

git push
   ↓
automation
   ↓
running application
Lesson 2 — Version Everything

Not just source code.

Version:

Source
Model
Docker Image
Deployment

The Git SHA provides the connection between these layers.

Lesson 3 — Deployment Success ≠ Application Success

This is critical.

helm upgrade
     ↓
successful command

doesn't necessarily mean:

application is healthy

You need:

rollout status
pod readiness
application logs
service verification
Lesson 4 — Debug Layer by Layer

Your debugging model became:

GitHub Actions
      ↓
Helm
      ↓
Deployment
      ↓
ReplicaSet
      ↓
Pod
      ↓
Container
      ↓
Application
      ↓
Network

This is much better than randomly changing configurations.

Lesson 5 — MLOps Is More Than Machine Learning

The ML model is only one component.

The real system is:

ML
+
Software Engineering
+
CI/CD
+
Docker
+
Kubernetes
+
Security
+
Observability
+
Automation
46. Your Resume Version

I would condense the project to something like this on your resume:

End-to-End MLOps Wine Quality Prediction Platform
Designed and implemented an end-to-end MLOps CI/CD pipeline using GitHub Actions, MLflow, Docker, Helm, and Kubernetes for automated model training and deployment.
Automated data processing, model training, evaluation, MLflow experiment tracking, Docker image creation, vulnerability scanning, and Kubernetes deployment.
Implemented Git SHA-based immutable Docker image versioning, enabling traceability between source-code commits, container images, and Kubernetes deployments.
Deployed a containerized Streamlit ML inference application to a 3-replica Kind Kubernetes cluster using Helm and Kubernetes RollingUpdates.
Implemented automated deployment verification using kubectl rollout status, pod/service validation, image verification, and Kubernetes failure diagnostics.
Debugged production-style CrashLoopBackOff, missing container dependencies, Docker build-context issues, entrypoint mismatches, rollout timeouts, and Kind networking problems using Kubernetes logs, events, ReplicaSets, and deployment inspection.
Integrated Trivy container vulnerability scanning and GitHub Actions Secrets for secure CI/CD operations.
47. The 30-Second Version

If the interviewer says:

"Tell me about one of your MLOps projects."

Say:

"I built an end-to-end MLOps platform for wine quality prediction. The pipeline is triggered by a Git push and uses GitHub Actions to run tests, train the model, track experiments with MLflow, build and scan a Docker image, and deploy the application to Kubernetes using Helm. I use the Git commit SHA as the Docker image tag so every deployment is immutable and traceable back to source code.

One of the most valuable parts of the project was troubleshooting Kubernetes rollout failures. I had pods entering CrashLoopBackOff, and instead of assuming Kubernetes was the problem, I traced the issue through ReplicaSets, pod status, container logs, and previous logs until I found that Streamlit wasn't installed in the image. I corrected the Docker dependency configuration, rebuilt the image, and redeployed it through the pipeline.

The project ultimately gave me hands-on experience across ML, CI/CD, Docker, Kubernetes, Helm, MLflow, security scanning, and production-style troubleshooting."

That is the version I would memorize.

48. Your Core Interview Story

If you remember only one architecture, remember this:

                 GIT PUSH
                    │
                    ▼
            ┌───────────────┐
            │ GitHub Actions│
            └───────┬───────┘
                    │
              ┌─────▼─────┐
              │   Pytest   │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │ ML Training│
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │   MLflow   │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │   Docker   │
              │ Git SHA Tag│
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │   Trivy    │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │ Docker Hub │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │    Kind    │
              │ Kubernetes │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │    Helm    │
              └─────┬─────┘
                    │
          ┌─────────▼─────────┐
          │   Deployment      │
          │    3 Replicas     │
          └─────────┬─────────┘
                    │
              ┌─────▼─────┐
              │ Streamlit  │
              │ ML App     │
              └────────────┘

The central message for the interview is not "I built a wine classifier."

It's:

"I built an automated, versioned, tested, containerized, observable deployment pipeline that takes machine-learning code from Git commit all the way to a running Kubernetes application, and I demonstrated that I could systematically troubleshoot failures across the CI, Docker, Kubernetes, and networking layers."