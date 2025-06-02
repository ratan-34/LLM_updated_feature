from flask import request, render_template, jsonify, session, redirect, url_for
import pandas as pd
import json
import logging
import sys
import io
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import base64

logger = logging.getLogger(__name__)

def add_llm_code_generation_routes(app, data_store, client):
    """Add LLM Code Generation routes to the Flask app"""
    
    @app.route('/llmcodegeneration')
    def llm_code_generation():
        try:
            session_id = session.get('session_id')
            if not session_id or session_id not in data_store:
                return redirect(url_for('index'))
            
            return render_template('llmcodegeneration.html')
        except Exception as e:
            logger.error(f"Error in llm_code_generation route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/llmcodegeneration/generate', methods=['POST'])
    def api_llmcodegeneration_generate():
        try:
            session_id = session.get('session_id')
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            # Get data from request
            data = request.json
            task_description = data.get('task_description')
            code_type = data.get('code_type', 'analysis')
            complexity = data.get('complexity', 3)
            
            if not task_description:
                return jsonify({'error': 'No task description provided'}), 400
            
            # Get dataframe info
            df = data_store[session_id]['df']
            df_info = {
                'columns': df.columns.tolist(),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'shape': df.shape,
                'sample': df.head(5).to_dict(orient='records')
            }
            
            # Use Azure OpenAI to generate code
            try:
                if client:
                    df_sample = df.head(10).to_string()
                    df_info_str = json.dumps(df_info, indent=2)
                    
                    complexity_levels = {
                        1: "basic (simple operations, no advanced techniques)",
                        2: "simple (common data operations, minimal complexity)",
                        3: "medium (some advanced techniques, good balance)",
                        4: "advanced (complex operations, efficient code)",
                        5: "expert (cutting-edge techniques, highly optimized)"
                    }
                    
                    complexity_desc = complexity_levels.get(complexity, "medium")
                    
                    prompt = f"""
                    You are a Python data science expert. Generate code for the following task:
                    
                    Task: {task_description}
                    
                    Code Type: {code_type}
                    Complexity Level: {complexity} ({complexity_desc})
                    
                    Dataset Information:
                    {df_info_str}
                    
                    Dataset Sample:
                    {df_sample}
                    
                    Please provide:
                    1. A clear explanation of what the code does
                    2. Well-commented Python code that accomplishes the task
                    3. A list of required packages
                    
                    The code should assume that the dataframe is already loaded as 'df'.
                    The code should be appropriate for the specified complexity level.
                    
                    Format your response as JSON with the following structure:
                    {{
                        "explanation": "Explanation of what the code does",
                        "code": "Python code",
                        "requirements": ["package1", "package2"]
                    }}
                    """
                    
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a Python data science expert providing code in JSON format only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=2000,
                        response_format={"type": "json_object"}
                    )
                    
                    result = json.loads(response.choices[0].message.content)
                else:
                    # Fallback if Azure OpenAI is not available
                    result = {
                        "explanation": "This code performs basic data analysis on the dataset.",
                        "code": "# Basic data analysis\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\n# Display basic statistics\nprint(df.describe())\n\n# Check for missing values\nprint('\\nMissing values:')\nprint(df.isnull().sum())\n\n# Plot a histogram for numeric columns\ndf.select_dtypes(include=['number']).hist(figsize=(10, 8))\nplt.tight_layout()\nplt.show()",
                        "requirements": ["pandas", "matplotlib"]
                    }
                
                # Store the generated code
                data_store[session_id]['generated_code'] = result['code']
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error generating code: {str(e)}")
                return jsonify({
                    "explanation": "Basic data analysis code (fallback due to error).",
                    "code": "# Basic data analysis\nimport pandas as pd\n\n# Display basic statistics\nprint(df.describe())\n\n# Check for missing values\nprint('\\nMissing values:')\nprint(df.isnull().sum())",
                    "requirements": ["pandas"],
                    "error": str(e)
                })
        
        except Exception as e:
            logger.error(f"Error in api_llmcodegeneration_generate: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/llmcodegeneration/execute', methods=['POST'])
    def api_llmcodegeneration_execute():
        try:
            session_id = session.get('session_id')
            if not session_id or session_id not in data_store:
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            # Get code from request
            data = request.json
            code = data.get('code', '')
            
            if not code:
                return jsonify({'error': 'No code provided'}), 400
            
            # Get dataframe
            df = data_store[session_id]['df'].copy()
            
            # Execute code
            try:
                # Redirect stdout to capture print statements
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()
                
                # Create a local namespace
                local_vars = {'df': df, 'plt': plt, 'np': np, 'pd': pd, 'sns': sns}
                
                # Execute the code
                exec(code, globals(), local_vars)
                
                # Get the output
                output = mystdout.getvalue()
                
                # Reset stdout
                sys.stdout = old_stdout
                
                # Check if there are any figures
                figures = []
                if plt.get_fignums():
                    for i in plt.get_fignums():
                        fig = plt.figure(i)
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format='png')
                        buffer.seek(0)
                        figures.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
                    
                    plt.close('all')
                
                return jsonify({
                    'output': output,
                    'figures': figures
                })
            
            except Exception as e:
                logger.error(f"Error executing code: {str(e)}")
                return jsonify({'error': str(e)}), 400
        
        except Exception as e:
            logger.error(f"Error in api_llmcodegeneration_execute: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500