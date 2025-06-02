from flask import request, render_template, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

def add_llm_mesh_routes(app, data_store, client):
    """Add LLM Mesh routes to the Flask app"""
    
    @app.route('/llm_mesh')
    def llm_mesh():
        try:
            session_id = session.get('session_id')
            if not session_id or session_id not in data_store:
                return redirect(url_for('index'))
            
            return render_template('llm_mesh.html')
        except Exception as e:
            logger.error(f"Error in llm_mesh route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/llm_mesh/analyze', methods=['POST'])
    def api_llm_mesh_analyze():
        try:
            session_id = session.get('session_id')
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Simulate LLM Mesh analysis with comprehensive data profiling
            analysis_result = perform_llm_mesh_analysis(df, filename, client)
            
            return jsonify(analysis_result)
        
        except Exception as e:
            logger.error(f"Error in api_llm_mesh_analyze: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_llm_mesh_analysis(df, filename, client):
    """
    Perform comprehensive LLM Mesh analysis simulating multiple LLMs working together
    """
    try:
        # Basic profiling
        profiling = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': int(df.isnull().sum().sum()),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'quality_score': calculate_data_quality_score(df)
        }
        
        # Anomaly detection
        anomalies = detect_anomalies(df)
        
        # Column analysis
        column_analysis = analyze_columns_with_llm(df)
        
        # Generate insights using Azure OpenAI
        insights = generate_llm_mesh_insights(df, filename, client)
        
        return {
            'dataset_name': filename,
            'profiling': profiling,
            'anomalies': anomalies,
            'column_analysis': column_analysis,
            'insights': insights,
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        logger.error(f"Error in perform_llm_mesh_analysis: {str(e)}")
        return {
            'error': str(e),
            'dataset_name': filename,
            'profiling': {'total_rows': 0, 'total_columns': 0},
            'anomalies': [],
            'column_analysis': [],
            'insights': []
        }

def calculate_data_quality_score(df):
    """Calculate a data quality score based on various factors"""
    try:
        # Factors for quality score
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = df.duplicated().sum() / len(df)
        
        # Calculate score (0-100)
        quality_score = 100 - (missing_ratio * 50) - (duplicate_ratio * 30)
        quality_score = max(0, min(100, quality_score))
        
        return f"{quality_score:.1f}%"
    except:
        return "N/A"

def detect_anomalies(df):
    """Detect various types of anomalies in the dataset"""
    anomalies = []
    
    try:
        # Check for high missing values
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                anomalies.append({
                    'type': f'High Missing Values in {col}',
                    'description': f'Column {col} has {missing_pct:.1f}% missing values',
                    'confidence': 95
                })
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(df)) * 100
            anomalies.append({
                'type': 'Duplicate Rows Detected',
                'description': f'Found {duplicate_count} duplicate rows ({duplicate_pct:.1f}% of dataset)',
                'confidence': 100
            })
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
                
                if len(outliers) > len(df) * 0.05:  # More than 5% outliers
                    anomalies.append({
                        'type': f'Statistical Outliers in {col}',
                        'description': f'Column {col} contains {len(outliers)} potential outliers',
                        'confidence': 80
                    })
        
        # Check for constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                anomalies.append({
                    'type': f'Constant Column: {col}',
                    'description': f'Column {col} has only one unique value',
                    'confidence': 100
                })
    
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
    
    return anomalies

def analyze_columns_with_llm(df):
    """Analyze each column and provide AI-powered insights"""
    column_analysis = []
    
    try:
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            unique_count = df[col].nunique()
            
            # Generate summary based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                summary = f"Numeric column with {unique_count} unique values. Range: {df[col].min():.2f} to {df[col].max():.2f}"
                if missing_pct < 5:
                    recommendation = "Good quality numeric data. Consider for statistical analysis."
                elif missing_pct < 20:
                    recommendation = "Some missing values. Consider for imputation strategies."
                else:
                    recommendation = "High missing values. Investigate data collection process."
            
            elif pd.api.types.is_object_dtype(df[col]):
                summary = f"Categorical column with {unique_count} unique values"
                if unique_count / len(df) > 0.8:
                    recommendation = "High cardinality. Consider grouping or encoding strategies."
                elif unique_count < 10:
                    recommendation = "Low cardinality. Good for categorical analysis."
                else:
                    recommendation = "Medium cardinality. Suitable for most analyses."
            
            else:
                summary = f"Column of type {col_type} with {unique_count} unique values"
                recommendation = "Review data type and consider appropriate preprocessing."
            
            column_analysis.append({
                'name': col,
                'type': col_type,
                'summary': summary,
                'recommendation': recommendation
            })
    
    except Exception as e:
        logger.error(f"Error in column analysis: {str(e)}")
    
    return column_analysis

def generate_llm_mesh_insights(df, filename, client):
    """Generate strategic insights using Azure OpenAI (simulating LLM Mesh)"""
    insights = []
    
    try:
        if client:
            # Create a comprehensive prompt for LLM Mesh analysis
            df_summary = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'missing_summary': df.isnull().sum().to_dict(),
                'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
            prompt = f"""
            You are part of an advanced LLM Mesh system analyzing the dataset "{filename}". 
            Multiple specialized AI models have contributed to this analysis. Provide strategic insights and recommendations.
            
            Dataset Summary:
            {json.dumps(df_summary, indent=2, default=str)}
            
            As part of the LLM Mesh, provide 5-7 strategic insights covering:
            1. Data quality and reliability assessment
            2. Business value and potential use cases
            3. Recommended analytical approaches
            4. Data preparation suggestions
            5. Risk factors and limitations
            6. Opportunities for further analysis
            7. Integration recommendations
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed strategic recommendation"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an advanced LLM Mesh system providing strategic data insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights if Azure OpenAI is not available
            insights = [
                {
                    "title": "Data Quality Assessment",
                    "description": f"The dataset contains {len(df)} rows and {len(df.columns)} columns with varying data quality. Consider implementing data validation rules."
                },
                {
                    "title": "Analysis Readiness",
                    "description": "The dataset appears suitable for exploratory data analysis. Recommend starting with descriptive statistics and visualization."
                },
                {
                    "title": "Missing Data Strategy",
                    "description": f"Missing values detected in {df.isnull().any().sum()} columns. Develop appropriate imputation or exclusion strategies."
                },
                {
                    "title": "Feature Engineering Opportunities",
                    "description": "Consider creating derived features from existing columns to enhance analytical value."
                },
                {
                    "title": "Scalability Considerations",
                    "description": "Evaluate computational requirements for larger datasets and consider optimization strategies."
                }
            ]
    
    except Exception as e:
        logger.error(f"Error generating LLM Mesh insights: {str(e)}")
        insights = [
            {
                "title": "Analysis Error",
                "description": f"Unable to generate comprehensive insights due to: {str(e)}"
            }
        ]
    
    return insights