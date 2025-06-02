from flask import request, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def add_genai_docs_routes(app, data_store, client):
    """Add GenAI for Documentation routes to the Flask app"""
    
    @app.route('/genai-docs')
    def genai_docs():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"GenAI Docs route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            logger.info(f"Session set for GenAI Docs: {session_id}")
            return app.send_static_file('templates/GenAI-for-Documentation.html')
        except Exception as e:
            logger.error(f"Error in genai_docs route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/genai-docs/dataset-info', methods=['GET'])
    def api_genai_docs_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"GenAI Docs dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Quick analysis for speed (limit to first 1000 rows)
            sample_df = df.head(1000) if len(df) > 1000 else df
            
            # Analyze columns for documentation generation (optimized)
            columns_info = []
            for col in sample_df.columns:
                col_type = str(sample_df[col].dtype)
                missing = sample_df[col].isna().sum()
                missing_pct = (missing / len(sample_df)) * 100
                unique_count = sample_df[col].nunique()
                
                # Get sample values quickly
                sample_values = []
                try:
                    non_null_values = sample_df[col].dropna()
                    if len(non_null_values) > 0:
                        sample_count = min(3, len(non_null_values))
                        sample_values = non_null_values.head(sample_count).astype(str).tolist()
                except Exception as e:
                    sample_values = ["N/A"]
                
                # Quick data quality assessment
                quality_score = 100 - missing_pct
                if pd.api.types.is_numeric_dtype(sample_df[col]):
                    try:
                        data_characteristics = f"Numeric: {sample_df[col].min():.2f} to {sample_df[col].max():.2f}"
                    except:
                        data_characteristics = f"Numeric data"
                elif pd.api.types.is_object_dtype(sample_df[col]):
                    data_characteristics = f"Text/Categorical: {unique_count} unique values"
                else:
                    data_characteristics = f"Data type: {col_type}"
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.1f}%",
                    'unique_count': int(unique_count),
                    'sample_values': sample_values,
                    'quality_score': f"{quality_score:.1f}%",
                    'characteristics': data_characteristics
                })

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_info': columns_info,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in api_genai_docs_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/genai-docs/generate', methods=['POST'])
    def api_genai_docs_generate():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"GenAI documentation generation requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            selected_columns = data.get('selected_columns', [])
            model = data.get('model', 'gpt-4o')
            doc_type = data.get('doc_type', 'comprehensive')
            additional_instructions = data.get('additional_instructions', '')
            
            if not selected_columns:
                return jsonify({'error': 'No columns selected for documentation'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Generate FAST documentation (under 5 seconds)
            start_time = time.time()
            documentation_result = generate_fast_etl_documentation(
                df, selected_columns, filename, model, doc_type, additional_instructions, client
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store documentation for download
            doc_id = str(uuid.uuid4())
            data_store[f"documentation_{doc_id}"] = {
                'documentation': documentation_result['documentation'],
                'documentation_html': documentation_result['documentation_html'],
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': filename,
                'columns': selected_columns
            }
            
            # Prepare data preview (limited for speed)
            data_preview = {
                'columns': df[selected_columns].columns.tolist(),
                'data': df[selected_columns].head(5).to_dict(orient='records'),
                'shape': df[selected_columns].shape
            }
            
            return jsonify({
                'doc_id': doc_id,
                'documentation': documentation_result['documentation'],
                'documentation_html': documentation_result['documentation_html'],
                'data_preview': data_preview,
                'processing_time': processing_time,
                'insights': documentation_result.get('insights', []),
                'etl_recommendations': documentation_result.get('etl_recommendations', [])
            })
        
        except Exception as e:
            logger.error(f"Error in api_genai_docs_generate: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/genai-docs/download', methods=['POST'])
    def api_genai_docs_download():
        try:
            data = request.json
            session_id = data.get('session_id')
            doc_id = data.get('doc_id')
            
            if not session_id or not doc_id:
                return jsonify({'error': 'Missing session_id or doc_id'}), 400
            
            doc_key = f"documentation_{doc_id}"
            if doc_key not in data_store:
                return jsonify({'error': 'Documentation not found'}), 404
            
            doc_data = data_store[doc_key]
            
            # Create simple HTML file for download
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ETL Documentation - {doc_data['filename']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #3366FF; color: white; padding: 20px; border-radius: 5px; }}
        .content {{ margin-top: 20px; }}
        h3, h4, h5 {{ color: #333; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ETL Documentation</h1>
        <p>Dataset: {doc_data['filename']} | Generated: {doc_data['timestamp']}</p>
    </div>
    <div class="content">
        {doc_data['documentation_html']}
    </div>
</body>
</html>"""
            
            # Create temporary file
            temp_filename = f"ETL_Docs_{doc_data['filename']}_{doc_id[:8]}.html"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in api_genai_docs_download: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def generate_fast_etl_documentation(df, selected_columns, filename, model, doc_type, additional_instructions, client):
    """Generate FAST ETL documentation (under 5 seconds)"""
    try:
        # Quick data analysis (optimized for speed)
        selected_df = df[selected_columns].head(500)
        data_analysis = analyze_data_fast(selected_df, selected_columns)
        
        # Generate documentation using optimized AI or fallback
        if client:
            try:
                documentation = generate_fast_ai_documentation(
                    data_analysis, filename, model, doc_type, additional_instructions, client
                )
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {str(e)}")
                documentation = generate_fast_fallback_documentation(
                    data_analysis, filename, doc_type
                )
        else:
            documentation = generate_fast_fallback_documentation(
                data_analysis, filename, doc_type
            )
        
        # Convert to HTML format (simplified)
        documentation_html = convert_to_fast_html(documentation)
        
        # Generate quick insights
        insights = generate_fast_insights(data_analysis)
        etl_recommendations = generate_fast_recommendations()
        
        return {
            'documentation': documentation,
            'documentation_html': documentation_html,
            'insights': insights,
            'etl_recommendations': etl_recommendations
        }
    
    except Exception as e:
        logger.error(f"Error in generate_fast_etl_documentation: {str(e)}")
        return {
            'documentation': f"# ETL Documentation: {filename}\n\nBasic documentation generated for {len(selected_columns)} columns.",
            'documentation_html': f"<h3>ETL Documentation: {filename}</h3><p>Basic documentation generated for {len(selected_columns)} columns.</p>",
            'insights': [{'title': 'Documentation Generated', 'description': 'Basic documentation created successfully.'}],
            'etl_recommendations': []
        }

def analyze_data_fast(df, selected_columns):
    """Fast data analysis (optimized for speed)"""
    try:
        analysis = {
            'basic_info': {
                'total_rows': len(df),
                'total_columns': len(selected_columns),
                'missing_values_total': int(df.isnull().sum().sum()),
                'duplicate_rows': int(df.duplicated().sum())
            },
            'column_details': []
        }
        
        # Quick column analysis
        for col in selected_columns[:10]:
            if col in df.columns:
                col_analysis = {
                    'name': col,
                    'data_type': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
                    'unique_count': int(df[col].nunique())
                }
                
                # Quick type-specific analysis
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        col_analysis.update({
                            'min_value': float(df[col].min()),
                            'max_value': float(df[col].max()),
                            'mean_value': float(df[col].mean())
                        })
                    except:
                        pass
                elif pd.api.types.is_object_dtype(df[col]):
                    try:
                        top_values = df[col].value_counts().head(3).to_dict()
                        col_analysis['top_values'] = top_values
                    except:
                        pass
                
                analysis['column_details'].append(col_analysis)
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in analyze_data_fast: {str(e)}")
        return {
            'basic_info': {'total_rows': len(df), 'total_columns': len(selected_columns)},
            'column_details': []
        }

def generate_fast_ai_documentation(data_analysis, filename, model, doc_type, additional_instructions, client):
    """Generate documentation using Azure OpenAI (optimized for speed)"""
    try:
        basic_info = data_analysis['basic_info']
        column_details = data_analysis['column_details'][:5]
        
        prompt = f"""
        Generate concise ETL documentation for dataset "{filename}".
        
        Dataset: {basic_info['total_rows']} rows, {basic_info['total_columns']} columns
        Missing values: {basic_info['missing_values_total']}
        
        Key columns: {[col['name'] for col in column_details]}
        
        Create a {doc_type} documentation with:
        1. Executive Summary
        2. Data Overview
        3. Column Descriptions
        4. ETL Recommendations
        5. Data Quality Notes
        
        Keep it concise and professional. Focus on practical ETL insights.
        Additional instructions: {additional_instructions or 'None'}
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an ETL documentation expert. Create concise, professional documentation quickly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
            timeout=10
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Error in fast AI documentation: {str(e)}")
        raise

def generate_fast_fallback_documentation(data_analysis, filename, doc_type):
    """Generate fast fallback documentation"""
    basic_info = data_analysis['basic_info']
    column_details = data_analysis['column_details']
    
    documentation = f"""# ETL Documentation: {filename}

## Executive Summary
This {doc_type} documentation covers the dataset "{filename}" with {basic_info['total_rows']:,} rows and {basic_info['total_columns']} columns.

## Data Overview
- **Total Records**: {basic_info['total_rows']:,}
- **Total Columns**: {basic_info['total_columns']}
- **Missing Values**: {basic_info['missing_values_total']:,}
- **Duplicate Records**: {basic_info.get('duplicate_rows', 0):,}

## Column Analysis
"""
    
    for col in column_details:
        documentation += f"""
### {col['name']}
- **Type**: {col['data_type']}
- **Missing**: {col['missing_count']} ({col['missing_percentage']})
- **Unique Values**: {col['unique_count']}
"""
        
        if col.get('min_value') is not None:
            documentation += f"- **Range**: {col['min_value']:.2f} to {col['max_value']:.2f}\n"
        
        if col.get('top_values'):
            top_vals = list(col['top_values'].keys())[:3]
            documentation += f"- **Top Values**: {', '.join(map(str, top_vals))}\n"
    
    documentation += """
## ETL Recommendations
1. **Data Quality**: Implement validation checks for missing values
2. **Performance**: Consider indexing on key columns
3. **Monitoring**: Set up data quality alerts
4. **Documentation**: Keep this documentation updated

## Implementation Notes
- Generated automatically for quick ETL planning
- Review and customize for specific requirements
- Consider data volume for processing strategies
"""
    
    return documentation

def convert_to_fast_html(documentation):
    """Fast HTML conversion"""
    try:
        html = documentation.replace('\n', '<br>')
        html = html.replace('# ', '<h3 style="color: #3366FF; margin: 20px 0 10px 0;">')
        html = html.replace('## ', '<h4 style="color: #6366F1; margin: 15px 0 8px 0;">')
        html = html.replace('### ', '<h5 style="color: #8B5CF6; margin: 10px 0 5px 0;">')
        html = html.replace('**', '<strong>')
        html = html.replace('**', '</strong>')
        html = html.replace('- ', '<li>')
        
        return f'<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">{html}</div>'
    
    except Exception as e:
        logger.error(f"Error in fast HTML conversion: {str(e)}")
        return f'<pre>{documentation}</pre>'

def generate_fast_insights(data_analysis):
    """Generate quick insights"""
    try:
        insights = []
        basic_info = data_analysis['basic_info']
        
        if basic_info['missing_values_total'] > 0:
            insights.append({
                'title': 'Data Quality Check',
                'description': f'Found {basic_info["missing_values_total"]} missing values. Consider data cleaning strategies.'
            })
        
        if basic_info['total_rows'] > 100000:
            insights.append({
                'title': 'Large Dataset',
                'description': 'Large dataset detected. Consider partitioning and incremental processing.'
            })
        
        insights.append({
            'title': 'ETL Ready',
            'description': 'Dataset analysis complete. Ready for ETL pipeline design.'
        })
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating fast insights: {str(e)}")
        return [{'title': 'Analysis Complete', 'description': 'Data analysis completed successfully.'}]

def generate_fast_recommendations():
    """Generate quick ETL recommendations"""
    return [
        {
            'category': 'Performance',
            'title': 'Optimize Processing',
            'description': 'Use parallel processing and chunking for large datasets.',
            'priority': 'High'
        },
        {
            'category': 'Quality',
            'title': 'Data Validation',
            'description': 'Implement automated data quality checks.',
            'priority': 'High'
        },
        {
            'category': 'Monitoring',
            'title': 'Pipeline Monitoring',
            'description': 'Set up monitoring and alerting for ETL jobs.',
            'priority': 'Medium'
        }
    ]
