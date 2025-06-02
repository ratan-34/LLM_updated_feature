from flask import request, render_template, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import traceback

logger = logging.getLogger(__name__)

def add_data_profiling_routes(app, data_store, client):
    """Add Data Profiling routes to the Flask app"""
    
    @app.route('/dataprofiling')
    def data_profiling():
        try:
            session_id = session.get('session_id')
            logger.info(f"Data profiling requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            return render_template('dataprofiling.html')
        
        except Exception as e:
            logger.error(f"Error in data_profiling route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/dataprofiling', methods=['GET'])
    def api_data_profiling():
        try:
            session_id = session.get('session_id')
            logger.info(f"API data profiling requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            
            # Generate profiling data
            profile = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB",
                    'duplicated_rows': int(df.duplicated().sum())
                },
                'columns': []
            }
            
            # Process each column
            for col in df.columns:
                try:
                    col_type = str(df[col].dtype)
                    missing = int(df[col].isna().sum())
                    missing_pct = (missing / len(df)) * 100
                    
                    col_info = {
                        'name': col,
                        'type': col_type,
                        'missing': missing,
                        'missing_pct': f"{missing_pct:.2f}%",
                    }
                    
                    # Add numeric statistics
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            col_info.update({
                                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                                'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                                'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                                'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                            })
                            
                            # Generate histogram
                            plt.figure(figsize=(8, 4))
                            sns.histplot(df[col].dropna(), kde=True)
                            plt.title(f'Distribution of {col}')
                            plt.tight_layout()
                            
                            # Save plot to buffer
                            buffer = io.BytesIO()
                            plt.savefig(buffer, format='png')
                            buffer.seek(0)
                            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            plt.close()
                            
                            col_info['histogram'] = plot_data
                        except Exception as e:
                            logger.warning(f"Error generating numeric stats for column {col}: {str(e)}")
                        
                    # Add categorical statistics
                    elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                        try:
                            value_counts = df[col].value_counts().head(10).to_dict()
                            col_info.update({
                                'unique_values': int(df[col].nunique()),
                                'top_values': value_counts
                            })
                            
                            # Generate bar plot for top categories
                            if df[col].nunique() <= 20:  # Only for columns with reasonable number of categories
                                plt.figure(figsize=(10, 5))
                                top_categories = df[col].value_counts().head(10)
                                sns.barplot(x=top_categories.index, y=top_categories.values)
                                plt.title(f'Top Categories in {col}')
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                
                                # Save plot to buffer
                                buffer = io.BytesIO()
                                plt.savefig(buffer, format='png')
                                buffer.seek(0)
                                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                                plt.close()
                                
                                col_info['barplot'] = plot_data
                        except Exception as e:
                            logger.warning(f"Error generating categorical stats for column {col}: {str(e)}")
                    
                    # Add datetime statistics
                    elif pd.api.types.is_datetime64_dtype(df[col]):
                        try:
                            col_info.update({
                                'min': df[col].min().strftime('%Y-%m-%d %H:%M:%S'),
                                'max': df[col].max().strftime('%Y-%m-%d %H:%M:%S'),
                                'range': f"{(df[col].max() - df[col].min()).days} days"
                            })
                        except Exception as e:
                            logger.warning(f"Error generating datetime stats for column {col}: {str(e)}")
                    
                    profile['columns'].append(col_info)
                
                except Exception as e:
                    logger.warning(f"Error processing column {col}: {str(e)}")
                    # Add minimal column info to avoid breaking the UI
                    profile['columns'].append({
                        'name': col,
                        'type': str(df[col].dtype),
                        'missing': int(df[col].isna().sum()),
                        'missing_pct': f"{(df[col].isna().sum() / len(df)) * 100:.2f}%",
                        'error': str(e)
                    })
            
            # Generate correlation matrix for numeric columns
            try:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) > 1:
                    plt.figure(figsize=(10, 8))
                    corr_matrix = df[numeric_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                    plt.title('Correlation Matrix')
                    plt.tight_layout()
                    
                    # Save plot to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    corr_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    profile['correlation_matrix'] = corr_plot
            except Exception as e:
                logger.warning(f"Error generating correlation matrix: {str(e)}")
            
            # Use Azure OpenAI to generate insights
            try:
                if client:
                    df_sample = df.head(100).to_string()
                    prompt = f"""
                    You are a data analysis expert. Based on the following dataset sample, provide 3-5 key insights and potential analysis directions.
                    
                    Dataset Sample:
                    {df_sample}
                    
                    Please provide insights about:
                    1. Data quality issues you notice
                    2. Interesting patterns or relationships
                    3. Suggested analyses or visualizations
                    4. Potential business questions this data could answer
                    
                    Format your response as JSON with the following structure:
                    {{
                        "insights": [
                            {{
                                "title": "Insight title",
                                "description": "Detailed explanation"
                            }}
                        ],
                        "suggested_analyses": [
                            {{
                                "title": "Analysis title",
                                "description": "What to analyze and why it's valuable"
                            }}
                        ]
                    }}
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a data analysis expert providing insights in JSON format only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=1000,
                        response_format={"type": "json_object"}
                    )
                    
                    ai_insights = json.loads(response.choices[0].message.content)
                    profile['ai_insights'] = ai_insights
                else:
                    # Fallback if Azure OpenAI is not available
                    profile['ai_insights'] = {
                        "insights": [
                            {
                                "title": "Basic Data Overview",
                                "description": f"This dataset contains {len(df)} rows and {len(df.columns)} columns with various data types."
                            }
                        ],
                        "suggested_analyses": [
                            {
                                "title": "Exploratory Data Analysis",
                                "description": "Perform basic statistical analysis and visualization to understand the distribution and relationships in the data."
                            }
                        ]
                    }
            except Exception as e:
                logger.error(f"Error generating AI insights: {str(e)}")
                profile['ai_insights'] = {
                    "error": "Could not generate AI insights. Please try again later.",
                    "insights": [
                        {
                            "title": "Basic Data Overview",
                            "description": f"This dataset contains {len(df)} rows and {len(df.columns)} columns."
                        }
                    ],
                    "suggested_analyses": [
                        {
                            "title": "Exploratory Data Analysis",
                            "description": "Perform basic statistical analysis to understand your data."
                        }
                    ]
                }
            
            return jsonify(profile)
        
        except Exception as e:
            logger.error(f"Error in api_data_profiling: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'An error occurred while profiling the data.',
                'details': str(e),
                'basic_info': {
                    'rows': 0,
                    'columns': 0,
                    'memory_usage': '0 MB',
                    'duplicated_rows': 0
                },
                'columns': []
            }), 500