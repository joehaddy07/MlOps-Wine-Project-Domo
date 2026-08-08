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