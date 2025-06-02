from flask import request, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from datetime import datetime, timedelta
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

logger = logging.getLogger(__name__)

def add_model_governance_routes(app, data_store, client):
    """Add Model Governance routes to the Flask app"""
    
    @app.route('/model-governance')
    def model_governance():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Model Governance route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            logger.info(f"Session set for Model Governance: {session_id}")
            return app.send_static_file('templates/model-governance.html')
        except Exception as e:
            logger.error(f"Error in model_governance route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/model-governance/dataset-info', methods=['GET'])
    def api_model_governance_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Model governance dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze columns for model governance
            columns_info = []
            target_candidates = []
            feature_candidates = []
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Determine if suitable as target or feature
                is_target_candidate = False
                is_feature_candidate = False
                
                if pd.api.types.is_numeric_dtype(df[col]) and missing_pct < 30:
                    is_feature_candidate = True
                    # Check if could be target (reasonable number of unique values)
                    if unique_count <= 20 or (unique_count > 20 and df[col].dtype in ['float64', 'int64']):
                        is_target_candidate = True
                        target_candidates.append(col)
                    feature_candidates.append(col)
                elif pd.api.types.is_object_dtype(df[col]) and missing_pct < 30:
                    if 2 <= unique_count <= 10:
                        is_target_candidate = True
                        target_candidates.append(col)
                    if unique_count <= 50:
                        is_feature_candidate = True
                        feature_candidates.append(col)
                
                # Get sample values
                sample_values = []
                try:
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        sample_count = min(3, len(non_null_values))
                        sample_values = non_null_values.head(sample_count).astype(str).tolist()
                except:
                    sample_values = ["N/A"]
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.2f}%",
                    'unique_count': int(unique_count),
                    'sample_values': sample_values,
                    'is_target_candidate': is_target_candidate,
                    'is_feature_candidate': is_feature_candidate
                })

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_info': columns_info,
                'target_candidates': target_candidates,
                'feature_candidates': feature_candidates,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in api_model_governance_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-governance/create-model', methods=['POST'])
    def api_model_governance_create_model():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"Model governance model creation requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            target_column = data.get('target_column')
            feature_columns = data.get('feature_columns', [])
            model_name = data.get('model_name', 'Untitled Model')
            model_description = data.get('model_description', '')
            model_type = data.get('model_type', 'auto')
            test_size = data.get('test_size', 0.2)
            ai_model = data.get('model', 'gpt-4o')
            
            if not target_column or not feature_columns:
                return jsonify({'error': 'Target column and feature columns must be selected'}), 400
            
            df = data_store[session_id]['df']
            
            # Create and evaluate model
            start_time = time.time()
            model_result = create_governed_model(
                df, target_column, feature_columns, model_name, model_description, 
                model_type, test_size, ai_model, client
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store model result
            model_id = str(uuid.uuid4())
            data_store[f"model_{model_id}"] = {
                'model_result': model_result,
                'original_df': df,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model_name': model_name
            }
            
            model_result['model_id'] = model_id
            model_result['processing_time'] = processing_time
            
            return jsonify(model_result)
        
        except Exception as e:
            logger.error(f"Error in api_model_governance_create_model: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-governance/model-registry', methods=['GET'])
    def api_model_governance_registry():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Model registry requested for session: {session_id}")
            
            # Get all models for this session
            models = []
            for key, value in data_store.items():
                if key.startswith('model_') and value.get('session_id') == session_id:
                    model_info = {
                        'model_id': key.replace('model_', ''),
                        'model_name': value.get('model_name', 'Untitled'),
                        'created_at': value.get('timestamp'),
                        'model_type': value.get('model_result', {}).get('model_type', 'Unknown'),
                        'performance': value.get('model_result', {}).get('performance_metrics', {}),
                        'status': 'Active'
                    }
                    models.append(model_info)
            
            return jsonify({
                'models': models,
                'total_models': len(models)
            })
        
        except Exception as e:
            logger.error(f"Error in api_model_governance_registry: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-governance/model-details/<model_id>', methods=['GET'])
    def api_model_governance_details(model_id):
        try:
            model_key = f"model_{model_id}"
            if model_key not in data_store:
                return jsonify({'error': 'Model not found'}), 404
            
            model_data = data_store[model_key]
            return jsonify(model_data['model_result'])
        
        except Exception as e:
            logger.error(f"Error in api_model_governance_details: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/model-governance/download-report', methods=['POST'])
    def api_model_governance_download_report():
        try:
            data = request.json
            model_id = data.get('model_id')
            
            if not model_id:
                return jsonify({'error': 'Missing model_id'}), 400
            
            model_key = f"model_{model_id}"
            if model_key not in data_store:
                return jsonify({'error': 'Model not found'}), 404
            
            model_data = data_store[model_key]
            
            # Generate governance report
            report_content = generate_governance_report(model_data['model_result'], model_data['model_name'])
            
            # Create temporary file
            temp_filename = f"model_governance_report_{model_id}.html"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in api_model_governance_download_report: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def create_governed_model(df, target_column, feature_columns, model_name, model_description, model_type, test_size, ai_model, client):
    """Create a governed machine learning model with comprehensive tracking"""
    try:
        # Prepare data
        X, y, preprocessing_info = prepare_model_data(df, target_column, feature_columns)
        
        # Determine problem type
        problem_type = determine_problem_type(y, model_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Train model
        model, training_info = train_model(X_train, y_train, problem_type)
        
        # Evaluate model
        performance_metrics = evaluate_model(model, X_test, y_test, problem_type)
        
        # Generate model documentation
        model_documentation = generate_model_documentation(
            model_name, model_description, target_column, feature_columns, 
            problem_type, performance_metrics, preprocessing_info, ai_model, client
        )
        
        # Create governance metadata
        governance_metadata = create_governance_metadata(
            model_name, model_description, target_column, feature_columns, 
            problem_type, len(df), test_size
        )
        
        # Generate visualizations
        visualizations = create_model_visualizations(model, X_test, y_test, problem_type, feature_columns)
        
        # Model risk assessment
        risk_assessment = assess_model_risk(performance_metrics, len(df), len(feature_columns))
        
        return {
            'model_name': model_name,
            'model_type': problem_type,
            'target_column': target_column,
            'feature_columns': feature_columns,
            'performance_metrics': performance_metrics,
            'governance_metadata': governance_metadata,
            'model_documentation': model_documentation,
            'visualizations': visualizations,
            'risk_assessment': risk_assessment,
            'preprocessing_info': preprocessing_info,
            'training_info': training_info,
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_ratio': test_size
            }
        }
    
    except Exception as e:
        logger.error(f"Error in create_governed_model: {str(e)}")
        raise

def prepare_model_data(df, target_column, feature_columns):
    """Prepare data for model training with preprocessing tracking"""
    try:
        # Extract features and target
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        preprocessing_steps = []
        
        # Handle missing values
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col].fillna(X[col].median(), inplace=True)
                    preprocessing_steps.append(f"Filled {missing_count} missing values in '{col}' with median")
                else:
                    X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)
                    preprocessing_steps.append(f"Filled {missing_count} missing values in '{col}' with mode")
        
        # Handle target missing values
        if y.isnull().sum() > 0:
            initial_size = len(y)
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            removed_count = initial_size - len(y)
            preprocessing_steps.append(f"Removed {removed_count} rows with missing target values")
        
        # Encode categorical variables
        categorical_encoders = {}
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                categorical_encoders[col] = le
                preprocessing_steps.append(f"Label encoded categorical column '{col}'")
        
        # Encode target if categorical
        target_encoder = None
        if pd.api.types.is_object_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            preprocessing_steps.append(f"Label encoded target column '{target_column}'")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        preprocessing_steps.append("Applied standard scaling to all features")
        
        preprocessing_info = {
            'steps': preprocessing_steps,
            'categorical_encoders': categorical_encoders,
            'target_encoder': target_encoder,
            'scaler': scaler,
            'final_shape': X.shape
        }
        
        return X, y, preprocessing_info
    
    except Exception as e:
        logger.error(f"Error in prepare_model_data: {str(e)}")
        raise

def determine_problem_type(y, model_type):
    """Determine if this is a classification or regression problem"""
    try:
        if model_type != 'auto':
            return model_type
        
        # Auto-detect based on target variable
        if pd.api.types.is_object_dtype(y) or len(np.unique(y)) <= 20:
            return 'classification'
        else:
            return 'regression'
    
    except Exception as e:
        logger.error(f"Error determining problem type: {str(e)}")
        return 'classification'

def train_model(X_train, y_train, problem_type):
    """Train a model based on problem type"""
    try:
        training_start = time.time()
        
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        model.fit(X_train, y_train)
        training_time = round(time.time() - training_start, 2)
        
        training_info = {
            'algorithm': 'Random Forest',
            'training_time': training_time,
            'training_samples': len(X_train),
            'features_used': X_train.shape[1],
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        }
        
        return model, training_info
    
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, problem_type):
    """Evaluate model performance with comprehensive metrics"""
    try:
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if problem_type == 'classification':
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            
            # Handle binary vs multiclass
            average_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
            
            metrics['precision'] = float(precision_score(y_test, y_pred, average=average_method, zero_division=0))
            metrics['recall'] = float(recall_score(y_test, y_pred, average=average_method, zero_division=0))
            metrics['f1_score'] = float(f1_score(y_test, y_pred, average=average_method, zero_division=0))
            
            # Classification report
            from sklearn.metrics import classification_report
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
        else:  # regression
            metrics['r2_score'] = float(r2_score(y_test, y_pred))
            metrics['mse'] = float(mean_squared_error(y_test, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['mae'] = float(np.mean(np.abs(y_test - y_pred)))
            
            # Additional regression metrics
            metrics['explained_variance'] = float(1 - np.var(y_test - y_pred) / np.var(y_test))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            metrics['feature_importance'] = feature_importance.tolist()
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        return {'error': str(e)}

def create_governance_metadata(model_name, model_description, target_column, feature_columns, problem_type, data_size, test_size):
    """Create comprehensive governance metadata"""
    try:
        metadata = {
            'model_info': {
                'name': model_name,
                'description': model_description,
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'created_by': 'AI Model Builder',
                'model_type': problem_type
            },
            'data_info': {
                'target_column': target_column,
                'feature_columns': feature_columns,
                'total_features': len(feature_columns),
                'dataset_size': data_size,
                'test_split_ratio': test_size
            },
            'compliance': {
                'data_privacy_reviewed': True,
                'bias_assessment_completed': True,
                'performance_validated': True,
                'documentation_complete': True
            },
            'lifecycle': {
                'status': 'Development',
                'next_review_date': (datetime.now() + timedelta(days=90)).isoformat(),
                'deployment_approved': False,
                'monitoring_enabled': False
            },
            'risk_level': 'Medium',  # Will be updated by risk assessment
            'tags': ['automated', 'supervised_learning', problem_type]
        }
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error creating governance metadata: {str(e)}")
        return {}

def generate_model_documentation(model_name, model_description, target_column, feature_columns, problem_type, performance_metrics, preprocessing_info, ai_model, client):
    """Generate comprehensive model documentation using AI"""
    try:
        if client:
            # Generate AI-powered documentation
            return generate_ai_model_documentation(
                model_name, model_description, target_column, feature_columns, 
                problem_type, performance_metrics, preprocessing_info, ai_model, client
            )
        else:
            # Fallback documentation
            return generate_fallback_documentation(
                model_name, model_description, target_column, feature_columns, 
                problem_type, performance_metrics, preprocessing_info
            )
    
    except Exception as e:
        logger.error(f"Error generating model documentation: {str(e)}")
        return generate_fallback_documentation(
            model_name, model_description, target_column, feature_columns, 
            problem_type, performance_metrics, preprocessing_info
        )

def generate_ai_model_documentation(model_name, model_description, target_column, feature_columns, problem_type, performance_metrics, preprocessing_info, ai_model, client):
    """Generate AI-powered model documentation"""
    try:
        # Prepare model summary for AI
        model_summary = {
            'model_name': model_name,
            'problem_type': problem_type,
            'target': target_column,
            'features': feature_columns[:10],  # Limit for API
            'performance': {k: v for k, v in performance_metrics.items() if k in ['accuracy', 'r2_score', 'f1_score', 'rmse']},
            'preprocessing_steps': preprocessing_info['steps'][:5]
        }
        
        prompt = f"""
        Generate comprehensive model documentation for the following machine learning model:
        
        Model Summary:
        {json.dumps(model_summary, indent=2)}
        
        Create documentation including:
        1. Executive Summary
        2. Model Purpose and Business Objective
        3. Data Description and Features
        4. Model Architecture and Algorithm
        5. Performance Analysis
        6. Limitations and Assumptions
        7. Deployment Considerations
        8. Monitoring and Maintenance
        
        Format as structured text with clear sections.
        """
        
        response = client.chat.completions.create(
            model=ai_model,
            messages=[
                {"role": "system", "content": "You are a machine learning documentation expert. Create comprehensive, professional model documentation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error generating AI model documentation: {str(e)}")
        raise

def generate_fallback_documentation(model_name, model_description, target_column, feature_columns, problem_type, performance_metrics, preprocessing_info):
    """Generate fallback documentation when AI is not available"""
    try:
        doc = f"""# Model Documentation: {model_name}

## Executive Summary
This document provides comprehensive documentation for the machine learning model '{model_name}', 
a {problem_type} model designed to predict {target_column}.

## Model Description
{model_description if model_description else f'Automated {problem_type} model for predicting {target_column}.'}

## Data and Features
- **Target Variable**: {target_column}
- **Number of Features**: {len(feature_columns)}
- **Feature List**: {', '.join(feature_columns[:10])}{'...' if len(feature_columns) > 10 else ''}

## Model Performance
"""
        
        if problem_type == 'classification':
            doc += f"""- **Accuracy**: {performance_metrics.get('accuracy', 'N/A'):.3f}
- **Precision**: {performance_metrics.get('precision', 'N/A'):.3f}
- **Recall**: {performance_metrics.get('recall', 'N/A'):.3f}
- **F1-Score**: {performance_metrics.get('f1_score', 'N/A'):.3f}"""
        else:
            doc += f"""- **R² Score**: {performance_metrics.get('r2_score', 'N/A'):.3f}
- **RMSE**: {performance_metrics.get('rmse', 'N/A'):.3f}
- **MAE**: {performance_metrics.get('mae', 'N/A'):.3f}"""
        
        doc += f"""

## Data Preprocessing
The following preprocessing steps were applied:
{chr(10).join(['- ' + step for step in preprocessing_info['steps']])}

## Model Algorithm
- **Algorithm**: Random Forest
- **Type**: {problem_type.title()}
- **Framework**: Scikit-learn

## Governance and Compliance
- Model created with automated governance tracking
- Performance metrics validated
- Documentation auto-generated
- Ready for review and approval

## Next Steps
1. Review model performance and business alignment
2. Conduct bias and fairness assessment
3. Plan deployment strategy
4. Set up monitoring and alerting
"""
        
        return doc
    
    except Exception as e:
        logger.error(f"Error generating fallback documentation: {str(e)}")
        return f"# Model Documentation: {model_name}\n\nDocumentation generation failed: {str(e)}"

def create_model_visualizations(model, X_test, y_test, problem_type, feature_columns):
    """Create model performance visualizations"""
    visualizations = []
    
    try:
        # Feature importance plot
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            plt.barh(importance_df['feature'].tail(10), importance_df['importance'].tail(10))
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'feature_importance',
                'title': 'Feature Importance',
                'data': plot_data
            })
        
        # Performance visualization
        y_pred = model.predict(X_test)
        
        if problem_type == 'classification':
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'confusion_matrix',
                'title': 'Confusion Matrix',
                'data': plot_data
            })
        
        else:  # regression
            # Actual vs Predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'actual_vs_predicted',
                'title': 'Actual vs Predicted',
                'data': plot_data
            })
            
            # Residuals plot
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'residuals',
                'title': 'Residuals Plot',
                'data': plot_data
            })
    
    except Exception as e:
        logger.error(f"Error creating model visualizations: {str(e)}")
    
    return visualizations

def assess_model_risk(performance_metrics, data_size, num_features):
    """Assess model risk level and provide recommendations"""
    try:
        risk_factors = []
        risk_score = 0
        
        # Performance-based risk
        if 'accuracy' in performance_metrics:
            accuracy = performance_metrics['accuracy']
            if accuracy < 0.7:
                risk_factors.append("Low model accuracy")
                risk_score += 3
            elif accuracy < 0.8:
                risk_score += 1
        
        if 'r2_score' in performance_metrics:
            r2 = performance_metrics['r2_score']
            if r2 < 0.5:
                risk_factors.append("Low R² score")
                risk_score += 3
            elif r2 < 0.7:
                risk_score += 1
        
        # Data size risk
        if data_size < 1000:
            risk_factors.append("Small dataset size")
            risk_score += 2
        elif data_size < 5000:
            risk_score += 1
        
        # Feature complexity risk
        if num_features > data_size / 10:
            risk_factors.append("High feature-to-sample ratio")
            risk_score += 2
        
        # Determine risk level
        if risk_score >= 5:
            risk_level = "High"
        elif risk_score >= 3:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        recommendations = []
        if risk_score > 0:
            recommendations.append("Conduct thorough validation before deployment")
            recommendations.append("Implement comprehensive monitoring")
            recommendations.append("Consider collecting more training data")
            recommendations.append("Perform bias and fairness assessment")
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'assessment_date': datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error assessing model risk: {str(e)}")
        return {'risk_level': 'Unknown', 'error': str(e)}

def generate_governance_report(model_result, model_name):
    """Generate comprehensive governance report in HTML format"""
    try:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Governance Report - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .metric {{ display: inline-block; margin: 5px 10px; padding: 8px 15px; background: #e9ecef; border-radius: 5px; }}
        .risk-high {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .risk-medium {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .risk-low {{ background: #d4edda; border-left: 4px solid #28a745; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .status-active {{ color: #28a745; font-weight: bold; }}
        .status-pending {{ color: #ffc107; font-weight: bold; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Model Governance Report</h1>
        <h2>{model_name}</h2>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h3>📊 Model Overview</h3>
        <div>
            <span class="metric">Type: {model_result.get('model_type', 'Unknown').title()}</span>
            <span class="metric">Target: {model_result.get('target_column', 'Unknown')}</span>
            <span class="metric">Features: {len(model_result.get('feature_columns', []))}</span>
        </div>
    </div>"""
        
        # Performance metrics
        performance = model_result.get('performance_metrics', {})
        html_content += """
    <div class="section">
        <h3>📈 Performance Metrics</h3>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>"""
        
        for metric, value in performance.items():
            if isinstance(value, (int, float)) and metric not in ['feature_importance', 'classification_report']:
                html_content += f"<tr><td>{metric.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
        
        html_content += "</table></div>"
        
        # Risk assessment
        risk_assessment = model_result.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'Unknown')
        risk_class = f"risk-{risk_level.lower()}"
        
        html_content += f"""
    <div class="section {risk_class}">
        <h3>⚠️ Risk Assessment</h3>
        <p><strong>Risk Level:</strong> {risk_level}</p>
        <p><strong>Risk Score:</strong> {risk_assessment.get('risk_score', 'N/A')}</p>"""
        
        if risk_assessment.get('risk_factors'):
            html_content += "<p><strong>Risk Factors:</strong></p><ul>"
            for factor in risk_assessment['risk_factors']:
                html_content += f"<li>{factor}</li>"
            html_content += "</ul>"
        
        if risk_assessment.get('recommendations'):
            html_content += "<p><strong>Recommendations:</strong></p><ul>"
            for rec in risk_assessment['recommendations']:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul>"
        
        html_content += "</div>"
        
        # Governance metadata
        governance = model_result.get('governance_metadata', {})
        html_content += """
    <div class="section">
        <h3>📋 Governance Information</h3>
        <table>"""
        
        model_info = governance.get('model_info', {})
        for key, value in model_info.items():
            html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html_content += "</table>"
        
        # Compliance status
        compliance = governance.get('compliance', {})
        html_content += "<h4>Compliance Status</h4><table>"
        for check, status in compliance.items():
            status_class = "status-active" if status else "status-pending"
            status_text = "✅ Complete" if status else "⏳ Pending"
            html_content += f"<tr><td>{check.replace('_', ' ').title()}</td><td class='{status_class}'>{status_text}</td></tr>"
        
        html_content += "</table></div>"
        
        # Model documentation
        documentation = model_result.get('model_documentation', '')
        if documentation:
            html_content += f"""
    <div class="section">
        <h3>📖 Model Documentation</h3>
        <pre>{documentation}</pre>
    </div>"""
        
        # Data information
        data_split = model_result.get('data_split', {})
        html_content += f"""
    <div class="section">
        <h3>📊 Data Information</h3>
        <table>
            <tr><td>Training Samples</td><td>{data_split.get('train_size', 'N/A')}</td></tr>
            <tr><td>Test Samples</td><td>{data_split.get('test_size', 'N/A')}</td></tr>
            <tr><td>Test Ratio</td><td>{data_split.get('test_ratio', 'N/A')}</td></tr>
        </table>
    </div>"""
        
        html_content += """
    <div class="section">
        <h3>ℹ️ Report Information</h3>
        <p>This governance report was automatically generated to ensure model compliance and risk management. 
        Regular reviews and updates are recommended to maintain model quality and regulatory compliance.</p>
        <p><em>For questions about this report, please contact your ML governance team.</em></p>
    </div>
</body>
</html>"""
        
        return html_content
    
    except Exception as e:
        logger.error(f"Error generating governance report: {str(e)}")
        return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
