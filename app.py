
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

# Azure OpenAI
from openai import AzureOpenAI

# Data storage
from data_storage import data_store

# Feature imports
from upload_handler import add_upload_routes
from nlp_tools_backend import add_nlp_tools_routes
from automl_backend import add_automl_routes
from data_profiling_backend import add_data_profiling_routes
from llm_code_generation_backend import add_llm_code_generation_routes
from llm_mesh_backend import add_llm_mesh_routes
from automated_feature_engineering_backend import add_feature_engineering_routes
from ai_copilot_backend import add_ai_copilot_routes
from genai_docs_backend import add_genai_docs_routes
from data_drift_detection_backend import add_data_drift_detection_routes
from model_training_deployment_backend import add_model_training_deployment_routes
from embedded_ml_clustering_backend import add_embedded_ml_clustering_routes
from time_series_forecasting_backend import add_time_series_forecasting_routes
from ai_driven_cataloging_backend import add_ai_driven_cataloging_routes
from model_governance_backend import add_model_governance_routes
from code_free_modeling_backend import add_code_free_modeling_routes
from prompt_engineering_interface import add_prompt_engineering_routes
from integration_notebooks_backend import add_integration_notebooks_routes
from ai_data_enrichment_backend import add_ai_data_enrichment_routes
from deploy_to_alteryx_promote import add_alteryx_promote_routes
from out_of_box_ml_backend import add_out_of_box_ml_routes
from clustering_backend import add_clustering_routes
from object_detection import object_detection_bp
from text_classification_summarization_backend import add_text_classification_routes

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "7c0d2e2f35e020ec485f18271ca26451"

# Azure OpenAI Configuration
try:
    client = AzureOpenAI(
        api_key="F6Zf4y3wEP8HoxVunVjWjIqyrbiEVw6YnNRdj5plfoulznvYVNLOJQQJ99BDAC77bzfXJ3w3AAABACOGqRfA",
        api_version="2024-04-01-preview",
        azure_endpoint="https://gen-ai-llm-deployment.openai.azure.com/"
    )
    deployment_name = "gpt-4o"
    logger.info("Azure OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
    client = None

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create temp folder for plots
TEMP_FOLDER = 'static/temp'
os.makedirs(TEMP_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Register all feature routes
add_upload_routes(app)
add_nlp_tools_routes(app, data_store, client)
add_automl_routes(app, data_store, client)
add_data_profiling_routes(app, data_store, client)
add_llm_code_generation_routes(app, data_store, client)
add_llm_mesh_routes(app, data_store, client)
add_feature_engineering_routes(app, data_store, client)
add_ai_copilot_routes(app, data_store, client)
add_genai_docs_routes(app, data_store, client)
add_data_drift_detection_routes(app, data_store, client)
add_model_training_deployment_routes(app, data_store, client)
add_embedded_ml_clustering_routes(app, data_store, client)
add_time_series_forecasting_routes(app, data_store, client)
add_ai_driven_cataloging_routes(app, data_store, client)
add_model_governance_routes(app, data_store, client)
add_code_free_modeling_routes(app, data_store, client)
add_prompt_engineering_routes(app, data_store, client)
add_integration_notebooks_routes(app, data_store, client)
add_ai_data_enrichment_routes(app, data_store, client)
add_alteryx_promote_routes(app, data_store, client)
add_out_of_box_ml_routes(app, data_store, client)
add_clustering_routes(app, data_store, client)
app.register_blueprint(object_detection_bp)
add_text_classification_routes(app, data_store, client)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)