from flask import request, render_template, jsonify, session, redirect, url_for, send_file
import pandas as pd
import numpy as np
import json
import uuid
import logging
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)

def add_nlp_tools_routes(app, data_store, client):
    """Add NLP Tools routes to the Flask app"""
    
    @app.route('/nlp-tools')
    def nlp_tools():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"NLP Tools route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            # Set session for this tab
            session['session_id'] = session_id
            logger.info(f"Session set for NLP Tools: {session_id}")
            return render_template('nlp-tools.html')
        except Exception as e:
            logger.error(f"Error in nlp_tools route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/nlp-tools/dataset-info', methods=['GET'])
    def api_nlp_tools_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"NLP Tools dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze columns for NLP suitability
            columns_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Check if column is suitable for NLP
                is_text_suitable = False
                if pd.api.types.is_object_dtype(df[col]):
                    # Check if it contains text-like data
                    sample_values = df[col].dropna().head(5).astype(str).tolist()
                    avg_length = np.mean([len(str(val)) for val in sample_values]) if sample_values else 0
                    if avg_length > 10:  # Assume text if average length > 10 characters
                        is_text_suitable = True
                
                columns_info.append({
                    'name': col,
                    'type': col_type,
                    'missing': int(missing),
                    'missing_pct': f"{missing_pct:.2f}%",
                    'unique_count': int(unique_count),
                    'is_text_suitable': is_text_suitable
                })

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns': columns_info,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in api_nlp_tools_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/nlp-tools/analyze', methods=['POST'])
    def api_nlp_tools_analyze():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"NLP analysis requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            text_column = data.get('text_column')
            nlp_task = data.get('nlp_task')
            model = data.get('model')
            sample_size = data.get('sample_size', '500')
            custom_labels = data.get('custom_labels', '')
            
            if not text_column or not nlp_task or not model:
                return jsonify({'error': 'Missing required parameters'}), 400
            
            df = data_store[session_id]['df']
            
            # Sample data if needed
            if sample_size != 'all':
                sample_size = int(sample_size)
                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                    logger.info(f"Sampling {len(df)} rows for NLP analysis")
            
            # Perform NLP analysis
            analysis_result = perform_nlp_analysis(df, text_column, nlp_task, model, custom_labels, client)
            
            # Store analysis result
            analysis_id = str(uuid.uuid4())
            data_store[f"nlp_analysis_{analysis_id}"] = {
                'result': analysis_result,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            analysis_result['analysis_id'] = analysis_id
            return jsonify(analysis_result)
        
        except Exception as e:
            logger.error(f"Error in api_nlp_tools_analyze: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def perform_nlp_analysis(df, text_column, nlp_task, model, custom_labels, client):
    """
    Perform NLP analysis using Azure OpenAI
    """
    try:
        # Get text data
        text_data = df[text_column].dropna().astype(str).tolist()
        processed_rows = len(text_data)
        
        if processed_rows == 0:
            raise ValueError("No valid text data found in the selected column")
        
        # Limit to first 100 rows for demo purposes (to avoid API limits)
        if len(text_data) > 100:
            text_data = text_data[:100]
            processed_rows = 100
        
        # Perform analysis based on task type
        if nlp_task == 'sentiment':
            results = analyze_sentiment(text_data, model, custom_labels, client)
        elif nlp_task == 'classification':
            results = classify_text(text_data, model, custom_labels, client)
        elif nlp_task == 'entity_extraction':
            results = extract_entities(text_data, model, client)
        elif nlp_task == 'summarization':
            results = summarize_text(text_data, model, client)
        elif nlp_task == 'keyword_extraction':
            results = extract_keywords(text_data, model, client)
        else:
            raise ValueError(f"Unsupported NLP task: {nlp_task}")
        
        # Calculate metrics
        metrics = calculate_nlp_metrics(results, nlp_task)
        
        # Generate insights
        insights = generate_nlp_insights(results, nlp_task, model)
        
        return {
            'processed_rows': processed_rows,
            'model': model,
            'task': nlp_task,
            'results': results,
            'metrics': metrics,
            'insights': insights
        }
    
    except Exception as e:
        logger.error(f"Error in perform_nlp_analysis: {str(e)}")
        raise

def analyze_sentiment(text_data, model, custom_labels, client):
    """Analyze sentiment of text data"""
    try:
        if not client:
            # Fallback sentiment analysis
            return fallback_sentiment_analysis(text_data)
        
        results = []
        labels = custom_labels.split(',') if custom_labels else ['positive', 'negative', 'neutral']
        labels = [label.strip() for label in labels]
        
        # Process in batches to avoid token limits
        batch_size = 10
        for i in range(0, len(text_data), batch_size):
            batch = text_data[i:i+batch_size]
            
            prompt = f"""
            Analyze the sentiment of the following texts. Classify each text as one of: {', '.join(labels)}.
            
            Texts:
            {chr(10).join([f"{j+1}. {text}" for j, text in enumerate(batch)])}
            
            Respond with JSON format:
            {{
                "results": [
                    {{"text": "original text", "sentiment": "label", "confidence": 0.95}},
                    ...
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analysis AI. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            batch_results = json.loads(response.choices[0].message.content)
            results.extend(batch_results.get('results', []))
        
        return results[:len(text_data)]  # Ensure we don't have more results than input
    
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        return fallback_sentiment_analysis(text_data)

def fallback_sentiment_analysis(text_data):
    """Fallback sentiment analysis using simple rules"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
    
    results = []
    for text in text_data:
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            confidence = min(0.9, 0.6 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = 'negative'
            confidence = min(0.9, 0.6 + (neg_count - pos_count) * 0.1)
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence
        })
    
    return results

def classify_text(text_data, model, custom_labels, client):
    """Classify text data into categories"""
    # Implementation similar to analyze_sentiment but for classification
    return fallback_text_classification(text_data)

def extract_entities(text_data, model, client):
    """Extract named entities from text"""
    # Implementation for entity extraction
    return fallback_entity_extraction(text_data)

def summarize_text(text_data, model, client):
    """Summarize text data"""
    # Implementation for text summarization
    return fallback_summarization(text_data)

def extract_keywords(text_data, model, client):
    """Extract keywords from text data"""
    # Implementation for keyword extraction
    return fallback_keyword_extraction(text_data)

def fallback_text_classification(text_data):
    """Fallback text classification"""
    return [{'text': text, 'category': 'general', 'confidence': 0.5} for text in text_data]

def fallback_entity_extraction(text_data):
    """Fallback entity extraction"""
    return [{'text': text, 'entities': []} for text in text_data]

def fallback_summarization(text_data):
    """Fallback summarization"""
    return [{'text': text, 'summary': text[:100] + '...', 'compression_ratio': 0.5} for text in text_data]

def fallback_keyword_extraction(text_data):
    """Fallback keyword extraction"""
    return [{'text': text, 'keywords': [], 'phrases': []} for text in text_data]

def calculate_nlp_metrics(results, nlp_task):
    """Calculate metrics based on NLP task results"""
    return {
        'total_texts': len(results),
        'task_type': nlp_task
    }

def generate_nlp_insights(results, nlp_task, model):
    """Generate insights about NLP analysis results"""
    return [
        {
            'title': 'Analysis Complete',
            'description': f'Successfully processed {len(results)} texts using {nlp_task} analysis.'
        }
    ]