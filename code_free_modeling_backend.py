from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
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
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastCodeFreeModelingEngine:
    """
    Optimized Code-Free Modeling Engine for fast results (5-10 seconds)
    """
    
    def __init__(self, azure_client=None):
        self.client = azure_client
        # Simplified, fast models only
        self.models = {
            'classification': {
                'Logistic Regression': {
                    'model': LogisticRegression,
                    'params': {'max_iter': 100, 'random_state': 42, 'solver': 'liblinear'},
                    'description': 'Fast linear model for classification',
                    'pros': 'Very fast, interpretable, good baseline',
                    'cons': 'Limited to linear relationships'
                },
                'Random Forest': {
                    'model': RandomForestClassifier,
                    'params': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
                    'description': 'Fast ensemble method with good accuracy',
                    'pros': 'Handles non-linear data, feature importance',
                    'cons': 'Less interpretable than single trees'
                },
                'Decision Tree': {
                    'model': DecisionTreeClassifier,
                    'params': {'max_depth': 10, 'random_state': 42},
                    'description': 'Simple, fast, and interpretable',
                    'pros': 'Highly interpretable, fast training',
                    'cons': 'Can overfit, unstable'
                },
                'Naive Bayes': {
                    'model': GaussianNB,
                    'params': {},
                    'description': 'Very fast probabilistic classifier',
                    'pros': 'Extremely fast, works with small data',
                    'cons': 'Assumes feature independence'
                }
            },
            'regression': {
                'Linear Regression': {
                    'model': LinearRegression,
                    'params': {},
                    'description': 'Fast linear regression model',
                    'pros': 'Very fast, interpretable, good baseline',
                    'cons': 'Limited to linear relationships'
                },
                'Random Forest': {
                    'model': RandomForestRegressor,
                    'params': {'n_estimators': 50, 'max_depth': 10, 'random_state': 42, 'n_jobs': -1},
                    'description': 'Fast ensemble method for regression',
                    'pros': 'Handles non-linear data, robust',
                    'cons': 'Less interpretable'
                },
                'Decision Tree': {
                    'model': DecisionTreeRegressor,
                    'params': {'max_depth': 10, 'random_state': 42},
                    'description': 'Simple and fast regression tree',
                    'pros': 'Interpretable, fast training',
                    'cons': 'Can overfit'
                },
                'K-Nearest Neighbors': {
                    'model': KNeighborsRegressor,
                    'params': {'n_neighbors': 5},
                    'description': 'Simple distance-based regression',
                    'pros': 'Simple, no training phase',
                    'cons': 'Slow for large datasets'
                }
            }
        }
    
    def analyze_dataset(self, df):
        """Fast dataset analysis"""
        # Limit analysis to first 1000 rows for speed
        sample_df = df.head(1000) if len(df) > 1000 else df
        
        analysis = {
            'shape': df.shape,
            'columns': [],
            'data_quality': {},
            'recommendations': []
        }
        
        # Quick column analysis
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100),
                'unique_count': int(df[col].nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_categorical': pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]),
                'suitable_for_target': False,
                'suitable_for_feature': True
            }
            
            # Quick target suitability check
            if col_info['is_numeric']:
                if 2 <= col_info['unique_count'] <= 20:
                    col_info['suitable_for_target'] = True
                    col_info['problem_type'] = 'classification'
                elif col_info['unique_count'] > 20:
                    col_info['suitable_for_target'] = True
                    col_info['problem_type'] = 'regression'
            elif col_info['is_categorical']:
                if 2 <= col_info['unique_count'] <= 10:
                    col_info['suitable_for_target'] = True
                    col_info['problem_type'] = 'classification'
            
            # Quick sample values
            try:
                sample_values = sample_df[col].dropna().head(3).tolist()
                col_info['sample_values'] = [str(val)[:50] for val in sample_values]  # Limit length
            except:
                col_info['sample_values'] = []
            
            # Basic stats for numeric columns
            if col_info['is_numeric']:
                try:
                    col_info['min'] = float(df[col].min())
                    col_info['max'] = float(df[col].max())
                    col_info['mean'] = float(df[col].mean())
                except:
                    pass
            
            # Value counts for categorical (limited)
            if col_info['is_categorical'] and col_info['unique_count'] <= 20:
                try:
                    value_counts = df[col].value_counts().head(5).to_dict()
                    col_info['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
                except:
                    col_info['value_counts'] = {}
            
            analysis['columns'].append(col_info)
        
        # Quick data quality assessment
        analysis['data_quality'] = {
            'total_missing': int(df.isnull().sum().sum()),
            'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            'duplicate_rows': int(df.duplicated().sum()),
            'duplicate_percentage': float((df.duplicated().sum() / len(df)) * 100)
        }
        
        # Simple recommendations
        analysis['recommendations'] = self._generate_fast_recommendations(analysis)
        
        return analysis
    
    def _generate_fast_recommendations(self, analysis):
        """Generate quick recommendations"""
        recommendations = []
        
        if analysis['data_quality']['missing_percentage'] > 10:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'title': 'Handle Missing Values',
                'description': f"Dataset has {analysis['data_quality']['missing_percentage']:.1f}% missing values."
            })
        
        suitable_targets = [col for col in analysis['columns'] if col['suitable_for_target']]
        if suitable_targets:
            recommendations.append({
                'type': 'modeling',
                'priority': 'high',
                'title': 'Target Variables Available',
                'description': f"Found {len(suitable_targets)} potential target variables."
            })
        
        return recommendations
    
    def fast_preprocess_data(self, df, target_column, feature_columns):
        """Ultra-fast preprocessing"""
        # Sample data if too large (for speed)
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)
            logger.info(f"Sampled dataset to 5000 rows for faster processing")
        
        # Create feature matrix and target vector
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Fast missing value handling
        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Simple imputation
        if len(numeric_features) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
        
        if len(categorical_features) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        
        # Fast encoding for categorical variables
        label_encoders = {}
        for col in categorical_features:
            if X[col].nunique() <= 50:  # Only encode if not too many categories
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            else:
                # Drop high cardinality categorical columns
                X = X.drop(col, axis=1)
        
        # Encode target if categorical
        target_encoder = None
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
        else:
            y_encoded = y
        
        # Simple scaling for numeric features only
        scaler = None
        if len(numeric_features) > 0:
            scaler = StandardScaler()
            X[numeric_features] = scaler.fit_transform(X[numeric_features])
        
        preprocessing_info = {
            'numeric_imputer': numeric_imputer if len(numeric_features) > 0 else None,
            'categorical_imputer': categorical_imputer if len(categorical_features) > 0 else None,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'scaler': scaler,
            'feature_names': X.columns.tolist()
        }
        
        return X, y_encoded, preprocessing_info
    
    def train_single_model(self, model_name, model_config, X_train, X_test, y_train, y_test, problem_type):
        """Train a single model quickly"""
        try:
            start_time = time.time()
            
            # Create and train model
            model = model_config['model'](**model_config['params'])
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate basic metrics
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                try:
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                except:
                    precision = recall = f1 = 0.0
                
                metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
                
                # Confusion matrix
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    metrics['confusion_matrix'] = cm.tolist()
                except:
                    metrics['confusion_matrix'] = []
                
            else:  # regression
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = np.mean(np.abs(y_test - y_pred))
                
                metrics = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'mae': float(mae)
                }
            
            # Feature importance (if available)
            feature_importance = []
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                    feature_importance = [
                        {'feature': name, 'importance': float(imp)}
                        for name, imp in zip(feature_names, importances)
                    ]
                    feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
            except:
                pass
            
            training_time = time.time() - start_time
            
            return {
                'model': model,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_time': training_time,
                'predictions': {
                    'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                    'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
                }
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def fast_train_models(self, X, y, problem_type, selected_models):
        """Train multiple models in parallel for speed"""
        start_time = time.time()
        
        # Quick train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if problem_type == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        results = {}
        
        # Train models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_model = {}
            
            for model_name in selected_models:
                if model_name not in self.models[problem_type]:
                    continue
                
                model_config = self.models[problem_type][model_name]
                future = executor.submit(
                    self.train_single_model,
                    model_name, model_config, X_train, X_test, y_train, y_test, problem_type
                )
                future_to_model[future] = model_name
            
            # Collect results
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results[model_name] = result
                except Exception as e:
                    logger.error(f"Error getting result for {model_name}: {str(e)}")
                    results[model_name] = {'error': str(e)}
        
        total_time = time.time() - start_time
        logger.info(f"Trained {len(results)} models in {total_time:.2f} seconds")
        
        return results, X_test, y_test
    
    def create_fast_visualizations(self, results, problem_type):
        """Create simple, fast visualizations"""
        visualizations = {}
        
        try:
            # Model comparison chart
            model_names = []
            scores = []
            
            for name, result in results.items():
                if 'metrics' in result:
                    model_names.append(name)
                    if problem_type == 'classification':
                        scores.append(result['metrics']['accuracy'])
                    else:
                        scores.append(result['metrics']['r2_score'])
            
            if scores:
                plt.figure(figsize=(10, 6))
                bars = plt.bar(model_names, scores, color=['#3366FF', '#6366F1', '#8B5CF6', '#EC4899'])
                plt.title(f'Model Comparison - {"Accuracy" if problem_type == "classification" else "R² Score"}', 
                         fontsize=14, fontweight='bold')
                plt.ylabel('Score')
                plt.xticks(rotation=45, ha='right')
                
                # Add value labels
                for bar, score in zip(bars, scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                visualizations['model_comparison'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
            
            # Feature importance for best model
            if model_names and scores:
                best_model_name = model_names[scores.index(max(scores))]
                if 'feature_importance' in results[best_model_name] and results[best_model_name]['feature_importance']:
                    feature_imp = results[best_model_name]['feature_importance'][:8]  # Top 8 features
                    
                    plt.figure(figsize=(8, 6))
                    features = [f['feature'] for f in feature_imp]
                    importances = [f['importance'] for f in feature_imp]
                    
                    plt.barh(features, importances, color='#3366FF')
                    plt.title(f'Top Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
                    plt.xlabel('Importance')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    visualizations['feature_importance'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
        
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
        
        return visualizations
    
    def generate_fast_insights(self, results, problem_type):
        """Generate quick insights without AI"""
        insights = []
        
        try:
            # Find best model
            best_model = None
            best_score = -1
            
            for model_name, result in results.items():
                if 'metrics' in result:
                    score = result['metrics']['accuracy'] if problem_type == 'classification' else result['metrics']['r2_score']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            if best_model:
                # Performance insight
                if best_score > 0.9:
                    insights.append({
                        'title': 'Excellent Model Performance',
                        'description': f'{best_model} achieved outstanding performance with {best_score:.1%} {"accuracy" if problem_type == "classification" else "R² score"}.',
                        'category': 'Performance',
                        'priority': 'High',
                        'actionable': True
                    })
                elif best_score > 0.7:
                    insights.append({
                        'title': 'Good Model Performance',
                        'description': f'{best_model} shows good performance with {best_score:.1%} {"accuracy" if problem_type == "classification" else "R² score"}.',
                        'category': 'Performance',
                        'priority': 'Medium',
                        'actionable': True
                    })
                else:
                    insights.append({
                        'title': 'Model Performance Needs Improvement',
                        'description': f'Best model achieved {best_score:.1%}. Consider feature engineering or more data.',
                        'category': 'Improvement',
                        'priority': 'High',
                        'actionable': True
                    })
                
                # Model recommendation
                insights.append({
                    'title': f'Recommended Model: {best_model}',
                    'description': f'{best_model} is the top performer and ready for deployment.',
                    'category': 'Recommendation',
                    'priority': 'High',
                    'actionable': True
                })
                
                # Quick deployment insight
                insights.append({
                    'title': 'Ready for Deployment',
                    'description': 'Your model is trained and ready for deployment as an API endpoint.',
                    'category': 'Deployment',
                    'priority': 'Medium',
                    'actionable': True
                })
                
                # Training time insight
                total_time = sum(r.get('training_time', 0) for r in results.values() if 'training_time' in r)
                insights.append({
                    'title': 'Fast Training Completed',
                    'description': f'All models trained in {total_time:.1f} seconds using optimized algorithms.',
                    'category': 'Performance',
                    'priority': 'Low',
                    'actionable': False
                })
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights.append({
                'title': 'Analysis Complete',
                'description': 'Model training completed successfully.',
                'category': 'Performance',
                'priority': 'Medium',
                'actionable': True
            })
        
        return insights

# Flask route integration
def add_code_free_modeling_routes(app, data_store, client):
    """Add optimized Code-Free Modeling routes to the Flask app"""
    
    modeling_engine = FastCodeFreeModelingEngine(client)
    
    @app.route('/code-free-modeling')
    def code_free_modeling():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            if not session_id or session_id not in data_store:
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            return render_template('code-free-modeling.html')
        except Exception as e:
            logger.error(f"Error in code_free_modeling route: {str(e)}")
            return redirect(url_for('index'))
    
    @app.route('/api/code-free-modeling/analyze-dataset', methods=['GET'])
    def analyze_dataset_for_modeling():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Fast analysis
            analysis = modeling_engine.analyze_dataset(df)
            analysis['filename'] = filename
            
            return jsonify(analysis)
        
        except Exception as e:
            logger.error(f"Error in analyze_dataset_for_modeling: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/train', methods=['POST'])
    def train_code_free_models():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No dataset found. Please upload a dataset first.'}), 400
            
            df = data_store[session_id]['df']
            
            # Extract parameters
            target_column = data.get('target_column')
            feature_columns = data.get('feature_columns', [])
            problem_type = data.get('problem_type')
            selected_models = data.get('selected_models', [])
            
            if not target_column or not feature_columns or not problem_type:
                return jsonify({'error': 'Missing required parameters'}), 400
            
            # Limit to 4 models max for speed
            if len(selected_models) > 4:
                selected_models = selected_models[:4]
            
            start_time = time.time()
            
            # Fast preprocessing
            X, y, preprocessing_info = modeling_engine.fast_preprocess_data(
                df, target_column, feature_columns
            )
            
            # Fast model training
            results, X_test, y_test = modeling_engine.fast_train_models(
                X, y, problem_type, selected_models
            )
            
            # Fast visualizations
            visualizations = modeling_engine.create_fast_visualizations(results, problem_type)
            
            # Fast insights
            insights = modeling_engine.generate_fast_insights(results, problem_type)
            
            processing_time = round(time.time() - start_time, 2)
            
            # Store results
            modeling_id = str(uuid.uuid4())
            data_store[f"modeling_{modeling_id}"] = {
                'results': results,
                'preprocessing_info': preprocessing_info,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Prepare response (exclude model objects for JSON serialization)
            response_results = {}
            for name, result in results.items():
                response_results[name] = {k: v for k, v in result.items() if k != 'model'}
            
            response = {
                'modeling_id': modeling_id,
                'results': response_results,
                'visualizations': visualizations,
                'insights': insights,
                'processing_time': processing_time,
                'dataset_info': {
                    'shape': df.shape,
                    'processed_shape': X.shape
                }
            }
            
            logger.info(f"Fast training completed in {processing_time} seconds")
            return jsonify(response)
        
        except Exception as e:
            logger.error(f"Error in train_code_free_models: {str(e)}")
            return jsonify({'error': f'Training failed: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/model-info', methods=['GET'])
    def get_model_info():
        try:
            problem_type = request.args.get('problem_type')
            
            if not problem_type or problem_type not in ['classification', 'regression']:
                return jsonify({'error': 'Invalid problem type'}), 400
            
            # Get model information
            model_info = {}
            for model_name, model_config in modeling_engine.models[problem_type].items():
                model_info[model_name] = {
                    'description': model_config.get('description', ''),
                    'pros': model_config.get('pros', ''),
                    'cons': model_config.get('cons', '')
                }
            
            return jsonify({'models': model_info})
        
        except Exception as e:
            logger.error(f"Error in get_model_info: {str(e)}")
            return jsonify({'error': f'Failed to get model information: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/download', methods=['POST'])
    def download_modeling_results():
        try:
            data = request.json
            modeling_id = data.get('modeling_id')
            
            if not modeling_id:
                return jsonify({'error': 'Missing modeling_id'}), 400
            
            modeling_key = f"modeling_{modeling_id}"
            if modeling_key not in data_store:
                return jsonify({'error': 'Modeling results not found'}), 404
            
            # Create results summary
            modeling_data = data_store[modeling_key]
            session_id = modeling_data['session_id']
            
            # Create summary DataFrame
            results_summary = []
            for model_name, result in modeling_data['results'].items():
                if 'metrics' in result:
                    row = {'Model': model_name}
                    row.update(result['metrics'])
                    results_summary.append(row)
            
            results_df = pd.DataFrame(results_summary)
            
            # Create temporary file
            temp_filename = f"fast_modeling_results_{modeling_id[:8]}.csv"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            # Save to CSV
            results_df.to_csv(temp_path, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in download_modeling_results: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/deploy', methods=['POST'])
    def deploy_model():
        try:
            data = request.json
            modeling_id = data.get('modeling_id')
            model_name = data.get('model_name')
            
            if not modeling_id or not model_name:
                return jsonify({'error': 'Missing modeling_id or model_name'}), 400
            
            modeling_key = f"modeling_{modeling_id}"
            if modeling_key not in data_store:
                return jsonify({'error': 'Modeling results not found'}), 404
            
            modeling_data = data_store[modeling_key]
            results = modeling_data['results']
            
            if model_name not in results or 'model' not in results[model_name]:
                return jsonify({'error': f'Model {model_name} not found in results'}), 404
            
            # Create deployment ID
            deployment_id = str(uuid.uuid4())
            
            # Store deployment information
            data_store[f"deployment_{deployment_id}"] = {
                'model': results[model_name]['model'],
                'preprocessing_info': modeling_data['preprocessing_info'],
                'model_name': model_name,
                'modeling_id': modeling_id,
                'session_id': modeling_data['session_id'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'metrics': results[model_name]['metrics']
            }
            
            # Create API endpoint URL
            api_endpoint = f"/api/code-free-modeling/predict/{deployment_id}"
            
            return jsonify({
                'deployment_id': deployment_id,
                'model_name': model_name,
                'api_endpoint': api_endpoint,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        except Exception as e:
            logger.error(f"Error in deploy_model: {str(e)}")
            return jsonify({'error': f'Deployment failed: {str(e)}'}), 500
    
    @app.route('/api/code-free-modeling/predict/<deployment_id>', methods=['POST'])
    def predict(deployment_id):
        try:
            deployment_key = f"deployment_{deployment_id}"
            if deployment_key not in data_store:
                return jsonify({'error': 'Deployment not found'}), 404
            
            deployment_data = data_store[deployment_key]
            model = deployment_data['model']
            preprocessing_info = deployment_data['preprocessing_info']
            
            # Get input data
            data = request.json
            if not data:
                return jsonify({'error': 'No input data provided'}), 400
            
            # Convert input to DataFrame
            input_df = pd.DataFrame([data])
            
            # Apply same preprocessing
            feature_names = preprocessing_info['feature_names']
            
            # Ensure all required features are present
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Default value
            
            # Select only the features used in training
            input_df = input_df[feature_names]
            
            # Apply preprocessing steps
            if preprocessing_info['scaler']:
                numeric_features = input_df.select_dtypes(include=['number']).columns
                if len(numeric_features) > 0:
                    input_df[numeric_features] = preprocessing_info['scaler'].transform(input_df[numeric_features])
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(input_df)
                    probabilities = probabilities.tolist()
                except:
                    pass
            
            return jsonify({
                'prediction': prediction.tolist(),
                'probabilities': probabilities,
                'model_name': deployment_data['model_name'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500