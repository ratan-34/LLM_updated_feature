from flask import request, render_template, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from datetime import datetime
import traceback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

logger = logging.getLogger(__name__)

def add_automl_routes(app, data_store, client):
    """Add AutoML routes to the Flask app"""
    
    @app.route('/automl')
    def automl():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"AutoML route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            # Set session for this tab
            session['session_id'] = session_id
            logger.info(f"Session set for AutoML: {session_id}")
            return render_template('automl.html')
        except Exception as e:
            logger.error(f"Error in automl route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/automl/dataset-info', methods=['GET'])
    def api_automl_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Sample data for faster processing (max 1000 rows)
            if len(df) > 1000:
                df_sample = df.sample(n=1000, random_state=42)
                logger.info(f"Sampling {len(df_sample)} rows from {len(df)} total rows for faster processing")
            else:
                df_sample = df
            
            # Analyze columns
            columns_info = []
            for col in df_sample.columns:
                col_type = str(df_sample[col].dtype)
                missing = df_sample[col].isna().sum()
                missing_pct = (missing / len(df_sample)) * 100
                unique_count = df_sample[col].nunique()
                
                # Determine if column is suitable for target
                is_target_suitable = False
                target_type = None
                
                if pd.api.types.is_numeric_dtype(df_sample[col]):
                    if unique_count <= 20 and unique_count >= 2:
                        is_target_suitable = True
                        target_type = 'classification'
                    elif unique_count > 20:
                        is_target_suitable = True
                        target_type = 'regression'
                elif pd.api.types.is_object_dtype(df_sample[col]):
                    if unique_count <= 50 and unique_count >= 2:
                        is_target_suitable = True
                        target_type = 'classification'
                
                # Get sample values safely
                sample_values = []
                try:
                    non_null_values = df_sample[col].dropna()
                    if len(non_null_values) > 0:
                        sample_count = min(3, len(non_null_values))
                        sample_values = non_null_values.head(sample_count).astype(str).tolist()
                except Exception as e:
                    logger.warning(f"Error getting sample values for column {col}: {str(e)}")
                    sample_values = ["N/A"]
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.2f}%",
                    'unique_count': int(unique_count),
                    'is_target_suitable': is_target_suitable,
                    'target_type': target_type,
                    'sample_values': sample_values
                })
            
            # Create a serializable version of algorithms without the actual model classes
            algorithms_info = {
                'classification': {
                    'quick': {
                        'Logistic Regression': {
                            'category': 'Linear Models',
                            'description': 'Fast linear classifier for binary and multiclass problems'
                        },
                        'Random Forest': {
                            'category': 'Ensemble',
                            'description': 'Robust ensemble method with high accuracy'
                        }
                    }
                },
                'regression': {
                    'quick': {
                        'Linear Regression': {
                            'category': 'Linear Models',
                            'description': 'Simple linear relationship modeling'
                        },
                        'Random Forest': {
                            'category': 'Ensemble',
                            'description': 'Robust ensemble method for regression'
                        }
                    }
                }
            }

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_info': columns_info,
                'algorithms': algorithms_info,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in api_automl_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/automl/analyze', methods=['POST'])
    def api_automl_analyze():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"Training requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            target_column = data.get('target_column')
            feature_columns = data.get('feature_columns', [])
            algorithm = data.get('algorithm')
            problem_type = data.get('problem_type')
            test_size = float(data.get('test_size', 0.2))
            
            if not target_column or not feature_columns or not algorithm:
                return jsonify({'error': 'Missing required parameters'}), 400
            
            df = data_store[session_id]['df']
            
            # Sample data for faster training (max 5000 rows)
            if len(df) > 5000:
                df = df.sample(n=5000, random_state=42)
                logger.info(f"Sampling {len(df)} rows for faster training")
            
            # Prepare data
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Remove rows with missing target values
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return jsonify({'error': 'No valid data after removing missing values'}), 400
            
            # Simple preprocessing for speed
            numeric_features = X.select_dtypes(include=['number']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            
            # Handle missing values quickly
            for col in numeric_features:
                X[col].fillna(X[col].median(), inplace=True)
            
            for col in categorical_features:
                X[col].fillna('Unknown', inplace=True)
            
            # Simple encoding for categorical variables
            if categorical_features:
                for col in categorical_features:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
            
            # Encode target if classification
            if problem_type == 'classification' and pd.api.types.is_object_dtype(y):
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                y_encoded = y
                label_encoder = None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, 
                stratify=y_encoded if problem_type == 'classification' else None
            )
            
            # Create and train model (simplified for speed)
            if problem_type == 'classification':
                if algorithm == 'Logistic Regression':
                    model = LogisticRegression(max_iter=50, random_state=42)
                else:  # Random Forest
                    model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
            else:  # regression
                if algorithm == 'Linear Regression':
                    model = LinearRegression()
                else:  # Random Forest
                    model = RandomForestRegressor(n_estimators=20, random_state=42, max_depth=5)
            
            # Train model
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                
                # Create simple confusion matrix plot
                try:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                    plt.title('Confusion Matrix')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    plt.tight_layout()
                    
                    # Save plot
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
                    buffer.seek(0)
                    cm_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                except Exception as e:
                    logger.warning(f"Error creating confusion matrix: {str(e)}")
                    cm_plot = None
                
                metrics = {
                    'accuracy': float(accuracy),
                    'confusion_matrix': cm_plot
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2_score': float(r2)
                }
            
            # Feature importance (if available)
            feature_importance = None
            try:
                if hasattr(model, 'feature_importances_'):
                    feature_names = X.columns.tolist()
                    importances = model.feature_importances_
                    feature_importance = list(zip(feature_names, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    feature_importance = feature_importance[:10]  # Top 10 features
            except Exception as e:
                logger.warning(f"Could not extract feature importance: {str(e)}")
            
            # Store model for download
            model_id = str(uuid.uuid4())
            model_data = {
                'model': model,
                'label_encoder': label_encoder,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'algorithm': algorithm,
                'problem_type': problem_type,
                'metrics': metrics,
                'training_time': training_time
            }
            data_store[f"model_{model_id}"] = model_data
            
            # Generate AI insights (simplified for speed)
            ai_insights = generate_automl_insights_fast(metrics, feature_importance, algorithm, problem_type)
            
            return jsonify({
                'model_id': model_id,
                'algorithm': algorithm,
                'problem_type': problem_type,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_time': float(training_time),
                'ai_insights': ai_insights,
                'data_shape': {
                    'train': X_train.shape,
                    'test': X_test.shape
                }
            })
        
        except Exception as e:
            logger.error(f"Error in api_automl_analyze: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_automl_insights_fast(metrics, feature_importance, algorithm, problem_type):
    """Generate fast insights without AI for speed"""
    if problem_type == 'classification':
        accuracy = metrics['accuracy']
        if accuracy > 0.9:
            performance = "Excellent performance"
            recommendation = "Model is ready for deployment"
        elif accuracy > 0.8:
            performance = "Good performance"
            recommendation = "Consider minor tuning for improvement"
        elif accuracy > 0.7:
            performance = "Moderate performance"
            recommendation = "Try feature engineering or different algorithms"
        else:
            performance = "Poor performance"
            recommendation = "Needs significant improvement - check data quality"
    else:
        r2 = metrics['r2_score']
        if r2 > 0.9:
            performance = "Excellent fit"
            recommendation = "Model explains data very well"
        elif r2 > 0.7:
            performance = "Good fit"
            recommendation = "Model performs well with room for improvement"
        elif r2 > 0.5:
            performance = "Moderate fit"
            recommendation = "Consider feature engineering or more complex models"
        else:
            performance = "Poor fit"
            recommendation = "Model needs significant improvement"
    
    feature_insight = "No feature importance available"
    if feature_importance and len(feature_importance) > 0:
        top_feature = feature_importance[0][0]
        feature_insight = f"'{top_feature}' is the most important feature for prediction"
    
    return {
        "performance_assessment": f"{performance} achieved with {algorithm}",
        "feature_insights": feature_insight,
        "improvement_recommendations": [
            recommendation,
            "Consider collecting more training data",
            "Try different preprocessing techniques"
        ],
        "etl_suggestions": [
            "Check for data quality issues",
            "Validate data consistency",
            "Consider feature scaling if needed"
        ],
        "deployment_readiness": f"Model training completed in under 5 seconds. {recommendation}"
    }