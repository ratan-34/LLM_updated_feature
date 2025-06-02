from flask import request, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from datetime import datetime
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

logger = logging.getLogger(__name__)

def add_model_training_deployment_routes(app, data_store, client):
    """Add Model Training & Deployment routes to the Flask app"""
    
    @app.route('/model-training-deployment')
    def model_training_deployment():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Model Training & Deployment route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            logger.info(f"Session set for Model Training & Deployment: {session_id}")
            return app.send_static_file('templates/model-training-deployment.html')
        except Exception as e:
            logger.error(f"Error in model_training_deployment route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/model-training/dataset-info', methods=['GET'])
    def api_model_training_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Model training dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze columns for ML suitability
            columns_info = []
            potential_targets = []
            feature_columns = []
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Determine ML suitability
                ml_suitable = "Yes"
                ml_role = []
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    ml_role.append("Feature")
                    if unique_count < 20 and unique_count > 1:
                        ml_role.append("Target (Classification)")
                        potential_targets.append({'name': col, 'type': 'classification'})
                    elif unique_count > 20:
                        ml_role.append("Target (Regression)")
                        potential_targets.append({'name': col, 'type': 'regression'})
                    
                    if missing_pct < 50:
                        feature_columns.append(col)
                    
                elif pd.api.types.is_object_dtype(df[col]):
                    if unique_count < 50:
                        ml_role.append("Feature (Categorical)")
                        ml_role.append("Target (Classification)")
                        potential_targets.append({'name': col, 'type': 'classification'})
                        if missing_pct < 50:
                            feature_columns.append(col)
                    else:
                        ml_suitable = "Limited"
                        ml_role.append("High Cardinality")
                
                if missing_pct > 70:
                    ml_suitable = "No"
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.2f}%",
                    'unique_count': int(unique_count),
                    'ml_suitable': ml_suitable,
                    'ml_role': ml_role
                })

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_info': columns_info,
                'potential_targets': potential_targets,
                'feature_columns': feature_columns,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in api_model_training_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-training/train', methods=['POST'])
    def api_model_training_train():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"Model training requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            target_column = data.get('target_column')
            feature_columns = data.get('feature_columns', [])
            model_type = data.get('model_type', 'auto')
            problem_type = data.get('problem_type', 'auto')
            test_size = data.get('test_size', 0.2)
            ai_model = data.get('ai_model', 'gpt-4o')
            
            if not target_column:
                return jsonify({'error': 'No target column specified'}), 400
            
            if not feature_columns:
                return jsonify({'error': 'No feature columns selected'}), 400
            
            df = data_store[session_id]['df']
            
            # Train models
            start_time = time.time()
            training_results = train_ml_models(
                df, target_column, feature_columns, model_type, problem_type, test_size, ai_model, client
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store training results
            training_id = str(uuid.uuid4())
            data_store[f"training_{training_id}"] = {
                'training_results': training_results,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'parameters': {
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'model_type': model_type,
                    'problem_type': problem_type,
                    'test_size': test_size
                }
            }
            
            training_results['training_id'] = training_id
            training_results['processing_time'] = processing_time
            
            return jsonify(training_results)
        
        except Exception as e:
            logger.error(f"Error in api_model_training_train: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-training/deploy', methods=['POST'])
    def api_model_training_deploy():
        try:
            data = request.json
            session_id = data.get('session_id')
            training_id = data.get('training_id')
            model_name = data.get('model_name', 'best_model')
            deployment_type = data.get('deployment_type', 'local')
            
            if not session_id or not training_id:
                return jsonify({'error': 'Missing session_id or training_id'}), 400
            
            training_key = f"training_{training_id}"
            if training_key not in data_store:
                return jsonify({'error': 'Training results not found'}), 404
            
            training_data = data_store[training_key]
            
            # Deploy model
            deployment_result = deploy_model(training_data, model_name, deployment_type)
            
            # Store deployment info
            deployment_id = str(uuid.uuid4())
            data_store[f"deployment_{deployment_id}"] = {
                'deployment_result': deployment_result,
                'training_id': training_id,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            deployment_result['deployment_id'] = deployment_id
            
            return jsonify(deployment_result)
        
        except Exception as e:
            logger.error(f"Error in api_model_training_deploy: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-training/download-model', methods=['POST'])
    def api_model_training_download_model():
        try:
            data = request.json
            session_id = data.get('session_id')
            training_id = data.get('training_id')
            
            if not session_id or not training_id:
                return jsonify({'error': 'Missing session_id or training_id'}), 400
            
            training_key = f"training_{training_id}"
            if training_key not in data_store:
                return jsonify({'error': 'Training results not found'}), 404
            
            training_data = data_store[training_key]
            
            # Create model package
            model_package_path = create_model_package(training_data, training_id)
            
            return send_file(model_package_path, as_attachment=True, 
                           download_name=f"ml_model_package_{training_id}.zip")
        
        except Exception as e:
            logger.error(f"Error in api_model_training_download_model: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def train_ml_models(df, target_column, feature_columns, model_type, problem_type, test_size, ai_model, client):
    """Train multiple ML models and compare performance"""
    try:
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.mode().iloc[0] if pd.api.types.is_object_dtype(y) else y.mean())
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Determine problem type automatically if needed
        if problem_type == 'auto':
            if pd.api.types.is_object_dtype(y) or y.nunique() < 20:
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        
        # Encode target if classification
        target_encoder = None
        if problem_type == 'classification' and pd.api.types.is_object_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models_to_train = get_models_to_train(model_type, problem_type)
        model_results = []
        trained_models = {}
        
        for model_name, model in models_to_train.items():
            try:
                # Train model
                if model_name in ['SVM', 'Logistic Regression']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                if problem_type == 'classification':
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                    }
                    primary_metric = metrics['accuracy']
                else:
                    metrics = {
                        'mse': float(mean_squared_error(y_test, y_pred)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        'r2_score': float(r2_score(y_test, y_pred))
                    }
                    primary_metric = metrics['r2_score']
                
                model_results.append({
                    'model_name': model_name,
                    'metrics': metrics,
                    'primary_metric': primary_metric,
                    'training_time': time.time()
                })
                
                trained_models[model_name] = {
                    'model': model,
                    'scaler': scaler if model_name in ['SVM', 'Logistic Regression'] else None,
                    'label_encoders': label_encoders,
                    'target_encoder': target_encoder
                }
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Sort by performance
        model_results.sort(key=lambda x: x['primary_metric'], reverse=(problem_type == 'classification'))
        
        # Create performance visualization
        performance_viz = create_model_performance_visualization(model_results, problem_type)
        
        # Generate AI insights
        if client:
            insights = generate_training_insights(model_results, problem_type, ai_model, client)
        else:
            insights = generate_fallback_training_insights(model_results, problem_type)
        
        return {
            'problem_type': problem_type,
            'model_results': model_results,
            'best_model': model_results[0]['model_name'] if model_results else None,
            'trained_models': trained_models,
            'data_info': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'feature_count': len(feature_columns),
                'target_classes': int(y.nunique()) if problem_type == 'classification' else None
            },
            'visualizations': [performance_viz] if performance_viz else [],
            'insights': insights,
            'preprocessing': {
                'scaler': scaler,
                'label_encoders': label_encoders,
                'target_encoder': target_encoder
            }
        }
    
    except Exception as e:
        logger.error(f"Error in train_ml_models: {str(e)}")
        raise

def get_models_to_train(model_type, problem_type):
    """Get models to train based on type and problem"""
    models = {}
    
    if problem_type == 'classification':
        if model_type == 'auto' or model_type == 'all':
            models.update({
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42)
            })
        elif model_type == 'random_forest':
            models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            models['Logistic Regression'] = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            models['SVM'] = SVC(random_state=42)
    
    else:  # regression
        if model_type == 'auto' or model_type == 'all':
            models.update({
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'SVM': SVR()
            })
        elif model_type == 'random_forest':
            models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear_regression':
            models['Linear Regression'] = LinearRegression()
        elif model_type == 'svm':
            models['SVM'] = SVR()
    
    return models

def create_model_performance_visualization(model_results, problem_type):
    """Create visualization comparing model performance"""
    try:
        if not model_results:
            return None
        
        plt.figure(figsize=(12, 6))
        
        model_names = [result['model_name'] for result in model_results]
        
        if problem_type == 'classification':
            # Classification metrics
            accuracies = [result['metrics']['accuracy'] for result in model_results]
            f1_scores = [result['metrics']['f1_score'] for result in model_results]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            plt.subplot(1, 2, 1)
            plt.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            plt.bar(x + width/2, f1_scores, width, label='F1 Score', alpha=0.8)
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Classification Model Performance')
            plt.xticks(x, model_names, rotation=45)
            plt.legend()
            plt.ylim(0, 1)
            
            # Precision vs Recall
            precisions = [result['metrics']['precision'] for result in model_results]
            recalls = [result['metrics']['recall'] for result in model_results]
            
            plt.subplot(1, 2, 2)
            plt.scatter(recalls, precisions, s=100, alpha=0.7)
            for i, name in enumerate(model_names):
                plt.annotate(name, (recalls[i], precisions[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision vs Recall')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            
        else:
            # Regression metrics
            r2_scores = [result['metrics']['r2_score'] for result in model_results]
            rmse_scores = [result['metrics']['rmse'] for result in model_results]
            
            plt.subplot(1, 2, 1)
            plt.bar(model_names, r2_scores, alpha=0.8, color='skyblue')
            plt.xlabel('Models')
            plt.ylabel('R² Score')
            plt.title('Model R² Scores')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            plt.bar(model_names, rmse_scores, alpha=0.8, color='lightcoral')
            plt.xlabel('Models')
            plt.ylabel('RMSE')
            plt.title('Model RMSE')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return {
            'type': 'model_performance',
            'title': f'{problem_type.title()} Model Performance Comparison',
            'data': plot_data
        }
    
    except Exception as e:
        logger.error(f"Error creating model performance visualization: {str(e)}")
        return None

def generate_training_insights(model_results, problem_type, ai_model, client):
    """Generate AI insights about model training results"""
    try:
        # Prepare summary for AI
        best_model = model_results[0] if model_results else None
        performance_summary = {
            'problem_type': problem_type,
            'models_trained': len(model_results),
            'best_model': best_model['model_name'] if best_model else None,
            'best_performance': best_model['primary_metric'] if best_model else None,
            'model_comparison': [
                {
                    'name': result['model_name'],
                    'performance': result['primary_metric']
                } for result in model_results[:3]
            ]
        }
        
        prompt = f"""
        Analyze the machine learning model training results and provide insights.
        
        Training Summary:
        {json.dumps(performance_summary, indent=2)}
        
        Provide 3-5 insights about:
        1. Model performance and selection
        2. Potential improvements and next steps
        3. Deployment considerations
        4. Model reliability and limitations
        
        Format as JSON:
        {{
            "insights": [
                {{
                    "title": "Insight title",
                    "description": "Detailed explanation and recommendation",
                    "category": "Performance/Deployment/Improvement"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are a machine learning expert providing model training insights in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response.get('insights', [])
    
    except Exception as e:
        logger.error(f"Error generating training insights: {str(e)}")
        return generate_fallback_training_insights(model_results, problem_type)

def generate_fallback_training_insights(model_results, problem_type):
    """Generate fallback insights when AI is not available"""
    insights = []
    
    if not model_results:
        return [{'title': 'Training Failed', 'description': 'No models were successfully trained.', 'category': 'Error'}]
    
    best_model = model_results[0]
    
    if problem_type == 'classification':
        if best_model['primary_metric'] > 0.9:
            insights.append({
                'title': 'Excellent Performance',
                'description': f'{best_model["model_name"]} achieved {best_model["primary_metric"]:.3f} accuracy. Model is ready for deployment.',
                'category': 'Performance'
            })
        elif best_model['primary_metric'] > 0.7:
            insights.append({
                'title': 'Good Performance',
                'description': f'{best_model["model_name"]} achieved {best_model["primary_metric"]:.3f} accuracy. Consider feature engineering for improvement.',
                'category': 'Performance'
            })
        else:
            insights.append({
                'title': 'Needs Improvement',
                'description': f'Best accuracy is {best_model["primary_metric"]:.3f}. Consider more data or different features.',
                'category': 'Improvement'
            })
    else:
        if best_model['primary_metric'] > 0.8:
            insights.append({
                'title': 'Strong Predictive Power',
                'description': f'{best_model["model_name"]} achieved R² of {best_model["primary_metric"]:.3f}. Good for deployment.',
                'category': 'Performance'
            })
        elif best_model['primary_metric'] > 0.5:
            insights.append({
                'title': 'Moderate Performance',
                'description': f'R² score is {best_model["primary_metric"]:.3f}. Model captures some patterns but has room for improvement.',
                'category': 'Performance'
            })
        else:
            insights.append({
                'title': 'Poor Fit',
                'description': f'Low R² score of {best_model["primary_metric"]:.3f}. Consider different features or models.',
                'category': 'Improvement'
            })
    
    insights.append({
        'title': 'Model Selection',
        'description': f'{best_model["model_name"]} performed best among {len(model_results)} models tested.',
        'category': 'Performance'
    })
    
    insights.append({
        'title': 'Next Steps',
        'description': 'Consider hyperparameter tuning and cross-validation for better performance.',
        'category': 'Improvement'
    })
    
    return insights

def deploy_model(training_data, model_name, deployment_type):
    """Deploy trained model"""
    try:
        training_results = training_data['training_results']
        best_model_name = training_results['best_model']
        
        if model_name == 'best_model':
            model_name = best_model_name
        
        if model_name not in training_results['trained_models']:
            raise ValueError(f"Model {model_name} not found in training results")
        
        model_info = training_results['trained_models'][model_name]
        
        # Create deployment package
        deployment_info = {
            'model_name': model_name,
            'deployment_type': deployment_type,
            'model_performance': next(
                (result for result in training_results['model_results'] if result['model_name'] == model_name),
                None
            ),
            'deployment_status': 'Success',
            'deployment_url': None,
            'api_endpoint': None
        }
        
        if deployment_type == 'local':
            # Save model locally
            model_path = save_model_locally(model_info, model_name)
            deployment_info['model_path'] = model_path
            deployment_info['deployment_notes'] = 'Model saved locally for inference'
            
        elif deployment_type == 'api':
            # Create API endpoint (simulated)
            api_endpoint = f"/api/predict/{model_name.lower().replace(' ', '_')}"
            deployment_info['api_endpoint'] = api_endpoint
            deployment_info['deployment_notes'] = 'API endpoint created for model inference'
            
        elif deployment_type == 'cloud':
            # Cloud deployment (simulated)
            deployment_info['deployment_url'] = f"https://ml-models.example.com/{model_name.lower().replace(' ', '_')}"
            deployment_info['deployment_notes'] = 'Model deployed to cloud platform'
        
        return deployment_info
    
    except Exception as e:
        logger.error(f"Error in deploy_model: {str(e)}")
        return {
            'deployment_status': 'Failed',
            'error': str(e),
            'deployment_notes': 'Deployment failed due to error'
        }

def save_model_locally(model_info, model_name):
    """Save model to local file system"""
    try:
        # Create models directory
        models_dir = 'static/models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model
        model_filename = f"{model_name.lower().replace(' ', '_')}_model.joblib"
        model_path = os.path.join(models_dir, model_filename)
        
        joblib.dump(model_info, model_path)
        
        return model_path
    
    except Exception as e:
        logger.error(f"Error saving model locally: {str(e)}")
        raise

def create_model_package(training_data, training_id):
    """Create downloadable model package"""
    try:
        import zipfile
        
        # Create temp directory for package
        package_dir = f'static/temp/model_package_{training_id}'
        os.makedirs(package_dir, exist_ok=True)
        
        training_results = training_data['training_results']
        best_model_name = training_results['best_model']
        
        # Save best model
        if best_model_name and best_model_name in training_results['trained_models']:
            model_info = training_results['trained_models'][best_model_name]
            model_path = os.path.join(package_dir, 'model.joblib')
            joblib.dump(model_info, model_path)
        
        # Create model info file
        model_info_path = os.path.join(package_dir, 'model_info.json')
        with open(model_info_path, 'w') as f:
            json.dump({
                'model_name': best_model_name,
                'problem_type': training_results['problem_type'],
                'performance': training_results['model_results'][0] if training_results['model_results'] else None,
                'data_info': training_results['data_info'],
                'timestamp': training_data['timestamp']
            }, f, indent=2)
        
        # Create README
        readme_path = os.path.join(package_dir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"""# ML Model Package

## Model Information
- **Model**: {best_model_name}
- **Type**: {training_results['problem_type']}
- **Created**: {training_data['timestamp']}

## Files
- `model.joblib`: Trained model and preprocessing components
- `model_info.json`: Model metadata and performance metrics
- `README.md`: This file

## Usage
```python
import joblib
model_info = joblib.load('model.joblib')
model = model_info['model']
# Use model for predictions

## Model Performance
{json.dumps(training_results['model_results'][0] if training_results['model_results'] else {}, indent=2)}
""")
        
        # Create ZIP package
        zip_path = f'static/temp/ml_model_package_{training_id}.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, package_dir)
                    zipf.write(file_path, arcname)
        
        return zip_path
    
    except Exception as e:
        logger.error(f"Error creating model package: {str(e)}")
        raise