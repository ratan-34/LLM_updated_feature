import os
import pandas as pd
import numpy as np
import json
import logging
import uuid
import time
from flask import Blueprint, request, jsonify, render_template, session
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
try:
    import umap
except ImportError:
    umap = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_embedded_ml_clustering_routes(app, data_store, client):
    @app.route('/embedded-ml-clustering')
    def embedded_ml_clustering():
        """Render the embedded ML clustering page"""
        try:
            return render_template('embedded_ml_clustering.html')
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            # Return a simple HTML page if template is not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Embedded ML Clustering</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
            </head>
            <body>
                <div class="container mt-5">
                    <div class="alert alert-warning">
                        <h4>Template Loading Issue</h4>
                        <p>The embedded ML clustering template could not be loaded. Please ensure the template file is in the correct location.</p>
                        <p>Error: """ + str(e) + """</p>
                    </div>
                </div>
            </body>
            </html>
            """
    
    @app.route('/api/clustering/get-columns', methods=['POST'])
    def get_dataset_columns():
        """Get columns from the dataset"""
        try:
            data = request.json
            session_id = data.get('session_id')
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'Invalid session ID'}), 400
            
            df = data_store[session_id]['data']
            
            # Get column information
            columns = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                is_categorical = pd.api.types.is_categorical_dtype(df[col]) or (df[col].nunique() < 10 and not is_numeric)
                is_text = pd.api.types.is_string_dtype(df[col]) and df[col].nunique() > 10
                is_datetime = pd.api.types.is_datetime64_dtype(df[col])
                
                columns.append({
                    'name': col,
                    'dtype': dtype,
                    'is_numeric': is_numeric,
                    'is_categorical': is_categorical,
                    'is_text': is_text,
                    'is_datetime': is_datetime,
                    'unique_values': int(df[col].nunique()),
                    'missing_values': int(df[col].isna().sum())
                })
            
            return jsonify({
                'columns': columns,
                'row_count': len(df)
            })
            
        except Exception as e:
            logger.error(f"Error getting columns: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/clustering/analyze', methods=['POST'])
    def analyze_clustering():
        """Perform clustering analysis"""
        try:
            data = request.json
            session_id = data.get('session_id')
            selected_columns = data.get('columns', [])
            algorithm = data.get('algorithm', 'kmeans')
            n_clusters = int(data.get('n_clusters', 3))
            embedding_method = data.get('embedding_method', 'pca')
            sample_size = int(data.get('sample_size', 1000))
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'Invalid session ID'}), 400
            
            if not selected_columns:
                return jsonify({'error': 'No columns selected'}), 400
            
            # Get the dataset
            df = data_store[session_id]['data']
            
            # Sample the dataset if it's too large
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
            
            # Select only the columns we need
            try:
                df_selected = df[selected_columns].copy()
            except KeyError as e:
                return jsonify({'error': f'Column not found: {str(e)}'}), 400
            
            # Handle missing values
            for col in df_selected.columns:
                if pd.api.types.is_numeric_dtype(df_selected[col]):
                    df_selected[col] = df_selected[col].fillna(df_selected[col].mean())
                else:
                    df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0] if len(df_selected[col].mode()) > 0 else 'Unknown')
            
            # Identify numeric and categorical columns
            numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create preprocessing pipeline
            preprocessor = None
            if numeric_cols and categorical_cols:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ])
            elif numeric_cols:
                preprocessor = ColumnTransformer(
                    transformers=[('num', StandardScaler(), numeric_cols)])
            elif categorical_cols:
                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
            
            if preprocessor is None:
                return jsonify({'error': 'No valid columns for clustering'}), 400
            
            # Transform the data
            X = preprocessor.fit_transform(df_selected)
            
            # Apply clustering algorithm
            labels = None
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X)
            elif algorithm == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                labels = model.fit_predict(X)
            elif algorithm == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)
            
            # Apply dimensionality reduction for visualization
            if embedding_method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif embedding_method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            elif embedding_method == 'umap' and umap is not None:
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                # Fallback to PCA if UMAP is not available
                reducer = PCA(n_components=2, random_state=42)
            
            # Get 2D embeddings
            X_dense = X.toarray() if hasattr(X, 'toarray') else X
            embeddings = reducer.fit_transform(X_dense)
            
            # Calculate silhouette score if applicable
            silhouette = None
            if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
                try:
                    silhouette = silhouette_score(X, labels)
                except:
                    silhouette = None
            
            # Generate cluster visualization
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f'Cluster Visualization using {algorithm.upper()} and {embedding_method.upper()}')
            plt.xlabel(f'{embedding_method.upper()} Component 1')
            plt.ylabel(f'{embedding_method.upper()} Component 2')
            
            # Save plot to a base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            # Generate cluster statistics
            cluster_stats = []
            df_with_clusters = df_selected.copy()
            df_with_clusters['cluster'] = labels
            
            for cluster_id in sorted(np.unique(labels)):
                cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
                stats = {
                    'cluster_id': int(cluster_id),
                    'size': len(cluster_df),
                    'percentage': round(len(cluster_df) / len(df_with_clusters) * 100, 2),
                    'features': {}
                }
                
                # Calculate statistics for each feature in the cluster
                for col in selected_columns:
                    if pd.api.types.is_numeric_dtype(df_selected[col]):
                        stats['features'][col] = {
                            'mean': round(float(cluster_df[col].mean()), 2),
                            'std': round(float(cluster_df[col].std()), 2),
                            'min': round(float(cluster_df[col].min()), 2),
                            'max': round(float(cluster_df[col].max()), 2)
                        }
                    else:
                        # For categorical features, get the most common value
                        if len(cluster_df[col]) > 0:
                            top_value = cluster_df[col].value_counts().index[0]
                            top_count = cluster_df[col].value_counts().iloc[0]
                            stats['features'][col] = {
                                'top_value': str(top_value),
                                'top_count': int(top_count),
                                'top_percentage': round(top_count / len(cluster_df) * 100, 2)
                            }
                
                cluster_stats.append(stats)
            
            # Generate LLM explanation if client is available
            explanation = None
            if client:
                try:
                    # Prepare prompt for LLM
                    prompt = f"""
                    I have performed clustering on a dataset with the following columns: {', '.join(selected_columns)}.
                    I used the {algorithm} algorithm and found {len(np.unique(labels))} clusters.
                    
                    Here are the statistics for each cluster:
                    {json.dumps(cluster_stats, indent=2)}
                    
                    Please provide a concise explanation of what these clusters might represent and what insights can be drawn from them.
                    Focus on the key differences between clusters and potential business implications.
                    Keep your explanation under 300 words and make it accessible to non-technical stakeholders.
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a data science expert specializing in cluster analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=500
                    )
                    
                    explanation = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error generating LLM explanation: {str(e)}")
                    explanation = "Unable to generate explanation due to an error."
            
            # Return the results
            return jsonify({
                'success': True,
                'cluster_count': len(np.unique(labels)),
                'silhouette_score': silhouette,
                'plot': plot_data,
                'cluster_stats': cluster_stats,
                'explanation': explanation,
                'sample_size': len(df_selected)
            })
            
        except Exception as e:
            logger.error(f"Error in clustering analysis: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/clustering/export-results', methods=['POST'])
    def export_clustering_results():
        """Export clustering results back to the dataset"""
        try:
            data = request.json
            session_id = data.get('session_id')
            selected_columns = data.get('columns', [])
            algorithm = data.get('algorithm', 'kmeans')
            n_clusters = int(data.get('n_clusters', 3))
            
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'Invalid session ID'}), 400
            
            # Get the dataset
            df = data_store[session_id]['data']
            
            # Select columns
            df_selected = df[selected_columns].copy()
            
            # Handle missing values
            for col in df_selected.columns:
                if pd.api.types.is_numeric_dtype(df_selected[col]):
                    df_selected[col] = df_selected[col].fillna(df_selected[col].mean())
                else:
                    df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0] if len(df_selected[col].mode()) > 0 else 'Unknown')
            
            # Preprocess data
            numeric_cols = df_selected.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()
            
            preprocessor = None
            if numeric_cols and categorical_cols:
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                    ])
            elif numeric_cols:
                preprocessor = ColumnTransformer(
                    transformers=[('num', StandardScaler(), numeric_cols)])
            elif categorical_cols:
                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])
            
            X = preprocessor.fit_transform(df_selected)
            
            # Apply clustering
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = model.fit_predict(X)
            elif algorithm == 'dbscan':
                model = DBSCAN(eps=0.5, min_samples=5)
                labels = model.fit_predict(X)
            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X)
            
            # Add cluster labels to the original dataset
            cluster_column_name = f"cluster_{algorithm}_{time.strftime('%Y%m%d_%H%M%S')}"
            data_store[session_id]['data'][cluster_column_name] = labels
            
            return jsonify({
                'success': True,
                'message': f'Cluster labels added as column "{cluster_column_name}"',
                'column_name': cluster_column_name
            })
            
        except Exception as e:
            logger.error(f"Error exporting clustering results: {str(e)}")
            return jsonify({'error': str(e)}), 500
