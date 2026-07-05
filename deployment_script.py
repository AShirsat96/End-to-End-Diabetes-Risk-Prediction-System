"""
=========================================================
XGBoost Model Deployment to AWS SageMaker (Serverless)
=========================================================

This script deploys a trained XGBoost diabetes risk model
to AWS SageMaker using Serverless Inference.

Workflow:
1. Prepare model artifacts
2. Define custom inference logic
3. Package model for SageMaker
4. Upload artifacts to Amazon S3
5. Deploy serverless SageMaker endpoint

NOTE:
- Model training and evaluation are done separately
  in Google Colab.
- This script handles DEPLOYMENT ONLY.
"""

# =========================================================
# 1. IMPORT REQUIRED LIBRARIES
# =========================================================
import os
import json
import shutil
import numpy as np

import sagemaker
from sagemaker import Session
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.serverless import ServerlessInferenceConfig
from sagemaker import image_uris


# =========================================================
# 2. SET UP DEPLOYMENT DIRECTORY
# =========================================================
# SageMaker expects a model.tar.gz file containing
# model artifacts and inference code.

DEPLOY_DIR = "/content/deploy"
os.makedirs(DEPLOY_DIR, exist_ok=True)


# =========================================================
# 3. COPY EXPORTED XGBOOST MODEL PARAMETERS
# =========================================================
# This JSON file is produced in Google Colab after
# feature selection, training, and evaluation.

SRC_MODEL_PATH = "/content/models/xgboost_tuned_model.json"
DST_MODEL_PATH = os.path.join(DEPLOY_DIR, "model_parameters.json")

shutil.copy(SRC_MODEL_PATH, DST_MODEL_PATH)

print("Model parameters copied to deploy directory.")


# =========================================================
# 4. CREATE THE SAGEMAKER INFERENCE SCRIPT
# =========================================================
# This script defines how SageMaker:
# - Loads the model
# - Parses requests
# - Runs inference
# - Returns predictions

inference_code = """
import json
import numpy as np

def model_fn(model_dir):
    '''
    Load model parameters from model_parameters.json
    '''
    with open(f"{model_dir}/model_parameters.json", "r") as f:
        params = json.load(f)
    return params

def input_fn(request_body, request_content_type):
    '''
    Parse incoming JSON payload
    '''
    return json.loads(request_body)

def predict_fn(inputs, params):
    '''
    Perform XGBoost-style logistic inference
    '''
    features = params["features"]
    weights = np.array(params["weights"])
    bias = params["bias"]
    threshold = params["threshold"]

    # Build feature vector in correct order
    x = np.array([float(inputs[f]) for f in features])

    # Logistic function
    log_odds = bias + np.dot(weights, x)
    prob = 1.0 / (1.0 + np.exp(-log_odds))

    return {
        "probability": float(prob),
        "risk_score": float(prob * 100.0),
        "label": int(prob >= threshold)
    }

def output_fn(prediction, accept):
    '''
    Serialize output as JSON
    '''
    return json.dumps(prediction)
"""

with open(os.path.join(DEPLOY_DIR, "inference.py"), "w") as f:
    f.write(inference_code)

print("Inference script created.")


# =========================================================
# 5. PACKAGE MODEL ARTIFACT FOR SAGEMAKER
# =========================================================
# SageMaker requires a model.tar.gz archive.

os.chdir(DEPLOY_DIR)
os.system("tar -czvf model.tar.gz model_parameters.json inference.py")

print("model.tar.gz created.")


# =========================================================
# 6. UPLOAD MODEL ARTIFACT TO AMAZON S3
# =========================================================
sess = Session()

S3_BUCKET = "diabetes-group-project-bucket"
S3_PREFIX = "diabetes-xgboost-model"

model_artifact = sess.upload_data(
    path="model.tar.gz",
    bucket=S3_BUCKET,
    key_prefix=S3_PREFIX
)

print("Model artifact uploaded to S3:")
print(model_artifact)


# =========================================================
# 7. CONFIGURE SAGEMAKER XGBOOST MODEL
# =========================================================
# Retrieve the official SageMaker XGBoost container image

region = sess.boto_region_name

EXECUTION_ROLE = "arn:aws:iam::430118821681:role/SageMakerExecutionRole"

image_uri = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.7-1"
)

xgb_model = XGBoostModel(
    model_data=model_artifact,
    role=EXECUTION_ROLE,
    entry_point="inference.py",
    image_uri=image_uri,
    sagemaker_session=sess
)


# =========================================================
# 8. DEPLOY USING SERVERLESS INFERENCE
# =========================================================
# Serverless inference provides:
# - Automatic scaling
# - No idle compute costs

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=2
)

ENDPOINT_NAME = "diabetes-risk-xgboost-v4"

predictor = xgb_model.deploy(
    endpoint_name=ENDPOINT_NAME,
    serverless_inference_config=serverless_config
)

print("===================================================")
print("SageMaker Serverless Endpoint Deployed Successfully")
print(f"Endpoint Name: {ENDPOINT_NAME}")
print("===================================================")
