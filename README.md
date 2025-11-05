# 🛡️ Network Security System - ML-Powered Phishing Detection

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.15-0194E2.svg)](https://mlflow.org/)

An end-to-end machine learning system for detecting phishing websites using network security data. Built with production-grade MLOps practices including automated training pipelines, experiment tracking, and real-time inference API.

## 🎯 Project Highlights

- **Production-Ready ML Pipeline**: Modular architecture with data ingestion, validation, transformation, and training components
- **Model Performance**: 97.6% F1-score on test data with ensemble learning (XGBoost, Random Forest, Gradient Boosting)
- **MLOps Integration**: Experiment tracking with MLflow, model versioning, and automated retraining capabilities
- **RESTful API**: FastAPI-based inference service with Swagger documentation
- **Data Quality Assurance**: Automated data validation and drift detection using statistical tests
- **Scalable Design**: Configuration-driven architecture supporting multiple environments

## 📊 Model Performance

| Metric | Train | Test |
|--------|-------|------|
| **F1 Score** | 0.991 | 0.976 |
| **Precision** | 0.987 | 0.966 |
| **Recall** | 0.994 | 0.985 |

## 🏗️ Architecture

### ML Training Pipeline
```
MongoDB → Data Ingestion → Data Validation → Feature Engineering → Model Training → Model Registry
    ↓           ↓                ↓                    ↓                  ↓              ↓
Raw Data   CSV Export    Schema/Drift Check    Preprocessing    GridSearchCV    MLflow Tracking
```

### Inference Pipeline
```
API Request → File Upload → Data Preprocessing → Model Prediction → JSON Response
     ↓            ↓               ↓                      ↓                ↓
FastAPI    CSV/Excel    Saved Preprocessor      Trained Model      Predictions
```

## 🚀 Key Features

### 1. **Modular ML Pipeline**
- **Data Ingestion**: Automated data extraction from MongoDB with connection pooling
- **Data Validation**: 
  - Schema validation (31 numerical features)
  - Column presence checks
  - Data drift detection using Kolmogorov-Smirnov test
  - Automated drift reports generation
- **Data Transformation**: 
  - Feature scaling using StandardScaler
  - Robust preprocessing pipeline
  - Saved transformers for inference consistency
- **Model Training**:
  - 7 ML algorithms comparison (Logistic Regression, KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost)
  - Automated hyperparameter tuning with GridSearchCV
  - Best model selection based on F1-score
  - Model serialization with pickle

### 2. **MLOps & Experiment Tracking**
- **MLflow Integration**:
  - Experiment tracking with DagHub
  - Model versioning and registry
  - Hyperparameter logging
  - Metric visualization
- **Artifact Management**:
  - Timestamped artifact directories
  - Model checkpointing
  - Preprocessor versioning

### 3. **Production-Grade API**
- **FastAPI Implementation**:
  - RESTful endpoints for training and prediction
  - File upload support (CSV/Excel)
  - CORS middleware for cross-origin requests
  - Automatic API documentation (Swagger/ReDoc)
- **Model Serving**:
  - Real-time predictions
  - Batch inference support
  - HTML table rendering for results

### 4. **Error Handling & Logging**
- Custom exception handling throughout pipeline
- Comprehensive logging with timestamps
- Detailed error messages with line numbers

## 📁 Project Structure

```
networkSecuritySystem/
├── network_security/
│   ├── components/
│   │   ├── data_ingestion.py          # MongoDB data extraction
│   │   ├── data_validation.py         # Schema & drift validation
│   │   ├── data_transformation.py     # Feature engineering
│   │   └── model_trainer.py           # Model training & evaluation
│   ├── entity/
│   │   ├── config_entity.py           # Configuration dataclasses
│   │   └── artifact_entity.py         # Pipeline artifact definitions
│   ├── constants/
│   │   └── training_pipeline.py       # Pipeline constants & configs
│   ├── utils/
│   │   ├── main_utils/
│   │   │   └── utils.py               # Helper functions (save/load, GridSearchCV)
│   │   └── ml_utils/
│   │       ├── model/estimator.py     # NetworkModel wrapper class
│   │       └── metric/                # Evaluation metrics
│   ├── exceptions/
│   │   └── exception.py               # Custom exception classes
│   └── logging/
│       └── logger.py                  # Logging configuration
├── data_schema/
│   └── schema.yaml                    # Data schema definition (31 features)
├── app.py                             # FastAPI application
├── main.py                            # Training pipeline orchestration
├── requirements.txt                   # Python dependencies
└── README.md
```

## 🛠️ Technology Stack

### Core ML/Data Science
- **Python 3.12**: Primary language
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: ML algorithms, preprocessing, metrics
- **XGBoost**: Gradient boosting framework
- **SciPy**: Statistical tests for drift detection

### MLOps & Tracking
- **MLflow**: Experiment tracking, model registry
- **DagHub**: Remote MLflow server
- **Pickle/Dill**: Model serialization

### Database & Data
- **MongoDB**: Data storage
- **PyMongo**: MongoDB driver
- **Certifi**: SSL certificate verification

### API & Deployment
- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Jinja2**: Template rendering
- **Python-dotenv**: Environment management


## 🚀 Usage

### Training the Model

```bash
# Run complete training pipeline
python main.py
```

This will:
1. Ingest data from MongoDB
2. Validate data quality and detect drift
3. Transform features and create preprocessor
4. Train multiple models with hyperparameter tuning
5. Log experiments to MLflow
6. Save best model to `final_model/`


### Making Predictions

#### Via cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.csv"
```

#### Via Python
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("test.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Training via API

```bash
curl -X GET "http://localhost:8000/train"
```

## 📊 Data Schema

The system expects 31 numerical features related to network security:

| Feature | Type | Description |
|---------|------|-------------|
| having_IP_Address | int64 | IP address present in URL |
| URL_Length | int64 | Length of URL |
| Shortining_Service | int64 | URL shortening service used |
| having_At_Symbol | int64 | '@' symbol present |
| double_slash_redirecting | int64 | '//' after protocol |
| ... | ... | ... (31 total features) |
| Result | int64 | Target variable (0: Safe, 1: Phishing) |

Full schema: `data_schema/schema.yaml`

## 🔧 Configuration

### Pipeline Configuration
Located in `network_security/constants/training_pipeline.py`:

```python
# Data Ingestion
DATA_INGESTION_COLLECTION_NAME = "NetworkData"
DATA_INGESTION_DATABASE_NAME = "aryan"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# Model Training
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD = 0.05
```

### Hyperparameter Grids
Configure in `model_trainer.py` for each algorithm:
- Logistic Regression: penalty, C, solver, max_iter
- KNN: n_neighbors, weights, algorithm
- Random Forest: n_estimators, max_depth, criterion
- XGBoost: learning_rate, max_depth, n_estimators, subsample
- And more...

## 📈 MLflow Tracking

View experiments at: https://dagshub.com/pycoder49/networkSecuritySystem.mlflow

Logged metrics:
- Training & test F1-scores
- Precision & Recall
- Model parameters
- Training artifacts


## 🎯 Future Enhancements

- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Implement real-time data streaming with Kafka
- [ ] Add model monitoring and alerting
- [ ] Containerize with Docker
- [ ] Deploy on Kubernetes
- [ ] Add A/B testing framework
- [ ] Implement model explainability (SHAP, LIME)
- [ ] Create web dashboard for predictions
- [ ] Add automated retraining on data drift detection
- [ ] Implement feature store for better feature management

## 👨‍💻 Author

**Aryan Ahuja**
- Email: aryan-a@outlook.com
- GitHub: [@pycoder49](https://github.com/pycoder49)
- DagHub: [pycoder49/networkSecuritySystem](https://dagshub.com/pycoder49/networkSecuritySystem)
