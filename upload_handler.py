from flask import request, jsonify, session
import pandas as pd
import uuid
import os
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

logger = logging.getLogger(__name__)

def add_upload_routes(app):
    """Add file upload routes to the Flask app"""
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            from data_storage import data_store
            
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            if file and (file.filename.endswith(('.csv', '.xlsx', '.xls'))):
                # Generate a unique ID for this session
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
                logger.info(f"New session created: {session_id}")
                
                # Save the file
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
                file.save(file_path)
                logger.info(f"File saved: {file_path}")
                
                # Read the file
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    logger.info(f"File read successfully: {filename}, shape: {df.shape}")
                except Exception as e:
                    logger.error(f"Error reading file: {str(e)}")
                    return jsonify({'error': f'Error reading file: {str(e)}'}), 400
                
                # Store the dataframe in our data store
                data_store[session_id] = {
                    'df': df,
                    'filename': filename,
                    'file_path': file_path,
                    'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Get basic info about the dataframe
                info = {
                    'filename': filename,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                    'preview': df.head(5).to_dict(orient='records'),
                    'session_id': session_id
                }
                
                return jsonify(info)
            
            return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400
        
        except Exception as e:
            logger.error(f"Error in upload_file: {str(e)}")
            return jsonify({'error': f'Server error: {str(e)}'}), 500