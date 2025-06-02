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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os

logger = logging.getLogger(__name__)

def add_time_series_forecasting_routes(app, data_store, client):
    """Add Time Series Forecasting routes to the Flask app"""
    
    @app.route('/time-series-forecasting')
    def time_series_forecasting():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Time Series Forecasting route accessed with session_id: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))
            
            session['session_id'] = session_id
            logger.info(f"Session set for Time Series Forecasting: {session_id}")
            return app.send_static_file('templates/time-series-forecasting.html')
        except Exception as e:
            logger.error(f"Error in time_series_forecasting route: {str(e)}")
            return redirect(url_for('index'))

    @app.route('/api/time-series/dataset-info', methods=['GET'])
    def api_time_series_dataset_info():
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Time series dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Analyze columns for time series suitability
            columns_info = []
            datetime_columns = []
            numeric_columns = []
            
            for col in df.columns:
                col_type = str(df[col].dtype)
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                unique_count = df[col].nunique()
                
                # Check if column could be datetime
                is_datetime = False
                is_numeric_suitable = False
                
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    is_datetime = True
                    datetime_columns.append(col)
                elif df[col].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        sample_values = df[col].dropna().head(100)
                        pd.to_datetime(sample_values, errors='raise')
                        is_datetime = True
                        datetime_columns.append(col)
                    except:
                        pass
                
                # Check if numeric column is suitable for forecasting
                if pd.api.types.is_numeric_dtype(df[col]) and missing_pct < 50:
                    is_numeric_suitable = True
                    numeric_columns.append(col)
                
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
                    'is_datetime': is_datetime,
                    'is_numeric_suitable': is_numeric_suitable
                })

            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'columns_info': columns_info,
                'datetime_columns': datetime_columns,
                'numeric_columns': numeric_columns,
                'session_id': session_id
            })
        
        except Exception as e:
            logger.error(f"Error in api_time_series_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/time-series/forecast', methods=['POST'])
    def api_time_series_forecast():
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"Time series forecasting requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            datetime_column = data.get('datetime_column')
            target_column = data.get('target_column')
            forecast_periods = data.get('forecast_periods', 30)
            model_type = data.get('model_type', 'linear_trend')
            model = data.get('model', 'gpt-4o')
            
            if not datetime_column or not target_column:
                return jsonify({'error': 'Both datetime and target columns must be selected'}), 400
            
            df = data_store[session_id]['df']
            
            # Perform time series forecasting
            start_time = time.time()
            forecast_result = perform_time_series_forecasting(
                df, datetime_column, target_column, forecast_periods, model_type, model, client
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store forecast result
            forecast_id = str(uuid.uuid4())
            data_store[f"forecast_{forecast_id}"] = {
                'result': forecast_result,
                'original_df': df,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            forecast_result['forecast_id'] = forecast_id
            forecast_result['processing_time'] = processing_time
            
            return jsonify(forecast_result)
        
        except Exception as e:
            logger.error(f"Error in api_time_series_forecast: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/time-series/download', methods=['POST'])
    def api_time_series_download():
        try:
            data = request.json
            session_id = data.get('session_id')
            forecast_id = data.get('forecast_id')
            
            if not session_id or not forecast_id:
                return jsonify({'error': 'Missing session_id or forecast_id'}), 400
            
            forecast_key = f"forecast_{forecast_id}"
            if forecast_key not in data_store:
                return jsonify({'error': 'Forecast result not found'}), 404
            
            forecast_data = data_store[forecast_key]
            forecast_df = forecast_data['result']['forecast_data']
            
            # Create temporary file
            temp_filename = f"time_series_forecast_{forecast_id}.csv"
            temp_path = os.path.join('static/temp', temp_filename)
            
            # Ensure temp directory exists
            os.makedirs('static/temp', exist_ok=True)
            
            # Save to CSV
            forecast_df.to_csv(temp_path, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in api_time_series_download: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

def perform_time_series_forecasting(df, datetime_column, target_column, forecast_periods, model_type, model, client):
    """Perform time series forecasting analysis"""
    try:
        # Prepare time series data
        ts_df = prepare_time_series_data(df, datetime_column, target_column)
        
        if ts_df.empty:
            raise ValueError("No valid time series data after preprocessing")
        
        # Analyze time series characteristics
        ts_analysis = analyze_time_series(ts_df, target_column)
        
        # Generate forecast
        forecast_result = generate_forecast(ts_df, target_column, forecast_periods, model_type)
        
        # Create visualizations
        visualizations = create_time_series_visualizations(ts_df, forecast_result, target_column)
        
        # Calculate forecast metrics
        metrics = calculate_forecast_metrics(ts_df, forecast_result, target_column)
        
        # Generate AI insights
        insights = generate_time_series_insights(ts_analysis, forecast_result, metrics, model, client)
        
        # Combine historical and forecast data
        forecast_data = combine_historical_and_forecast(ts_df, forecast_result, datetime_column, target_column)
        
        return {
            'model_type': model_type,
            'forecast_periods': forecast_periods,
            'forecast_data': forecast_data,
            'ts_analysis': ts_analysis,
            'metrics': metrics,
            'visualizations': visualizations,
            'insights': insights,
            'forecast_summary': {
                'last_actual_value': float(ts_df[target_column].iloc[-1]),
                'first_forecast_value': float(forecast_result['forecast'].iloc[0]),
                'last_forecast_value': float(forecast_result['forecast'].iloc[-1]),
                'forecast_trend': 'increasing' if forecast_result['forecast'].iloc[-1] > forecast_result['forecast'].iloc[0] else 'decreasing'
            }
        }
    
    except Exception as e:
        logger.error(f"Error in perform_time_series_forecasting: {str(e)}")
        raise

def prepare_time_series_data(df, datetime_column, target_column):
    """Prepare and clean time series data"""
    try:
        ts_df = df[[datetime_column, target_column]].copy()
        
        # Convert datetime column
        if not pd.api.types.is_datetime64_any_dtype(ts_df[datetime_column]):
            ts_df[datetime_column] = pd.to_datetime(ts_df[datetime_column], errors='coerce')
        
        # Remove rows with invalid dates or target values
        ts_df = ts_df.dropna()
        
        # Sort by datetime
        ts_df = ts_df.sort_values(datetime_column).reset_index(drop=True)
        
        # Remove duplicates (keep last)
        ts_df = ts_df.drop_duplicates(subset=[datetime_column], keep='last')
        
        # Ensure target column is numeric
        ts_df[target_column] = pd.to_numeric(ts_df[target_column], errors='coerce')
        ts_df = ts_df.dropna()
        
        return ts_df
    
    except Exception as e:
        logger.error(f"Error in prepare_time_series_data: {str(e)}")
        raise

def analyze_time_series(ts_df, target_column):
    """Analyze time series characteristics"""
    try:
        analysis = {
            'data_points': len(ts_df),
            'date_range': {
                'start': ts_df.iloc[0, 0].strftime('%Y-%m-%d'),
                'end': ts_df.iloc[-1, 0].strftime('%Y-%m-%d')
            },
            'target_stats': {
                'mean': float(ts_df[target_column].mean()),
                'std': float(ts_df[target_column].std()),
                'min': float(ts_df[target_column].min()),
                'max': float(ts_df[target_column].max()),
                'trend': 'increasing' if ts_df[target_column].iloc[-1] > ts_df[target_column].iloc[0] else 'decreasing'
            }
        }
        
        # Calculate basic trend
        x = np.arange(len(ts_df))
        y = ts_df[target_column].values
        trend_slope = np.polyfit(x, y, 1)[0]
        analysis['trend_slope'] = float(trend_slope)
        
        # Detect seasonality (simple approach)
        if len(ts_df) > 12:
            # Check for weekly pattern (if daily data)
            try:
                ts_df['day_of_week'] = ts_df.iloc[:, 0].dt.dayofweek
                weekly_pattern = ts_df.groupby('day_of_week')[target_column].mean().std()
                analysis['weekly_seasonality'] = float(weekly_pattern)
            except:
                analysis['weekly_seasonality'] = 0.0
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in analyze_time_series: {str(e)}")
        return {'data_points': len(ts_df), 'error': str(e)}

def generate_forecast(ts_df, target_column, forecast_periods, model_type):
    """Generate forecast using specified model"""
    try:
        if model_type == 'linear_trend':
            return linear_trend_forecast(ts_df, target_column, forecast_periods)
        elif model_type == 'moving_average':
            return moving_average_forecast(ts_df, target_column, forecast_periods)
        elif model_type == 'exponential_smoothing':
            return exponential_smoothing_forecast(ts_df, target_column, forecast_periods)
        else:
            # Default to linear trend
            return linear_trend_forecast(ts_df, target_column, forecast_periods)
    
    except Exception as e:
        logger.error(f"Error in generate_forecast: {str(e)}")
        raise

def linear_trend_forecast(ts_df, target_column, forecast_periods):
    """Generate forecast using linear trend"""
    try:
        # Fit linear trend
        x = np.arange(len(ts_df))
        y = ts_df[target_column].values
        
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Generate forecast
        forecast_x = np.arange(len(ts_df), len(ts_df) + forecast_periods)
        forecast_values = slope * forecast_x + intercept
        
        # Generate forecast dates
        last_date = ts_df.iloc[-1, 0]
        date_freq = infer_frequency(ts_df.iloc[:, 0])
        forecast_dates = pd.date_range(start=last_date + date_freq, periods=forecast_periods, freq=date_freq)
        
        return {
            'forecast': pd.Series(forecast_values, index=forecast_dates),
            'model_params': {'slope': slope, 'intercept': intercept},
            'confidence_intervals': generate_confidence_intervals(forecast_values, len(ts_df))
        }
    
    except Exception as e:
        logger.error(f"Error in linear_trend_forecast: {str(e)}")
        raise

def moving_average_forecast(ts_df, target_column, forecast_periods, window=7):
    """Generate forecast using moving average"""
    try:
        # Calculate moving average
        ma_values = ts_df[target_column].rolling(window=window).mean()
        last_ma = ma_values.iloc[-1]
        
        # Simple forecast: repeat last moving average
        forecast_values = np.full(forecast_periods, last_ma)
        
        # Generate forecast dates
        last_date = ts_df.iloc[-1, 0]
        date_freq = infer_frequency(ts_df.iloc[:, 0])
        forecast_dates = pd.date_range(start=last_date + date_freq, periods=forecast_periods, freq=date_freq)
        
        return {
            'forecast': pd.Series(forecast_values, index=forecast_dates),
            'model_params': {'window': window, 'last_ma': last_ma},
            'confidence_intervals': generate_confidence_intervals(forecast_values, len(ts_df))
        }
    
    except Exception as e:
        logger.error(f"Error in moving_average_forecast: {str(e)}")
        raise

def exponential_smoothing_forecast(ts_df, target_column, forecast_periods, alpha=0.3):
    """Generate forecast using exponential smoothing"""
    try:
        # Simple exponential smoothing
        values = ts_df[target_column].values
        smoothed = [values[0]]
        
        for i in range(1, len(values)):
            smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast: repeat last smoothed value
        last_smoothed = smoothed[-1]
        forecast_values = np.full(forecast_periods, last_smoothed)
        
        # Generate forecast dates
        last_date = ts_df.iloc[-1, 0]
        date_freq = infer_frequency(ts_df.iloc[:, 0])
        forecast_dates = pd.date_range(start=last_date + date_freq, periods=forecast_periods, freq=date_freq)
        
        return {
            'forecast': pd.Series(forecast_values, index=forecast_dates),
            'model_params': {'alpha': alpha, 'last_smoothed': last_smoothed},
            'confidence_intervals': generate_confidence_intervals(forecast_values, len(ts_df))
        }
    
    except Exception as e:
        logger.error(f"Error in exponential_smoothing_forecast: {str(e)}")
        raise

def infer_frequency(datetime_series):
    """Infer frequency from datetime series"""
    try:
        if len(datetime_series) < 2:
            return pd.Timedelta(days=1)
        
        # Calculate most common difference
        diffs = datetime_series.diff().dropna()
        most_common_diff = diffs.mode().iloc[0] if len(diffs.mode()) > 0 else pd.Timedelta(days=1)
        
        return most_common_diff
    
    except Exception as e:
        logger.error(f"Error inferring frequency: {str(e)}")
        return pd.Timedelta(days=1)

def generate_confidence_intervals(forecast_values, historical_length, confidence=0.95):
    """Generate simple confidence intervals"""
    try:
        # Simple approach: use standard error based on historical data length
        std_error = np.std(forecast_values) / np.sqrt(historical_length)
        z_score = 1.96  # 95% confidence
        
        margin = z_score * std_error
        
        return {
            'lower': forecast_values - margin,
            'upper': forecast_values + margin
        }
    
    except Exception as e:
        logger.error(f"Error generating confidence intervals: {str(e)}")
        return {'lower': forecast_values, 'upper': forecast_values}

def create_time_series_visualizations(ts_df, forecast_result, target_column):
    """Create time series visualizations"""
    visualizations = []
    
    try:
        # Historical data and forecast plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(ts_df.iloc[:, 0], ts_df[target_column], label='Historical Data', color='blue', linewidth=2)
        
        # Plot forecast
        forecast_dates = forecast_result['forecast'].index
        forecast_values = forecast_result['forecast'].values
        plt.plot(forecast_dates, forecast_values, label='Forecast', color='red', linewidth=2, linestyle='--')
        
        # Plot confidence intervals if available
        if 'confidence_intervals' in forecast_result:
            ci = forecast_result['confidence_intervals']
            plt.fill_between(forecast_dates, ci['lower'], ci['upper'], alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title('Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        visualizations.append({
            'type': 'forecast_plot',
            'title': 'Time Series Forecast',
            'data': plot_data
        })
        
        # Residuals plot (for historical data)
        if len(ts_df) > 10:
            plt.figure(figsize=(10, 6))
            
            # Calculate simple trend residuals
            x = np.arange(len(ts_df))
            y = ts_df[target_column].values
            trend_line = np.polyfit(x, y, 1)
            trend_values = np.polyval(trend_line, x)
            residuals = y - trend_values
            
            plt.subplot(2, 1, 1)
            plt.plot(ts_df.iloc[:, 0], residuals, color='green')
            plt.title('Residuals (Actual - Trend)')
            plt.ylabel('Residuals')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.hist(residuals, bins=20, alpha=0.7, color='green')
            plt.title('Residuals Distribution')
            plt.xlabel('Residual Value')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            visualizations.append({
                'type': 'residuals_plot',
                'title': 'Model Residuals Analysis',
                'data': plot_data
            })
    
    except Exception as e:
        logger.error(f"Error creating time series visualizations: {str(e)}")
    
    return visualizations

def calculate_forecast_metrics(ts_df, forecast_result, target_column):
    """Calculate forecast accuracy metrics using holdout validation"""
    try:
        if len(ts_df) < 10:
            return {'note': 'Insufficient data for validation metrics'}
        
        # Use last 20% of data for validation
        split_point = int(len(ts_df) * 0.8)
        train_data = ts_df.iloc[:split_point]
        test_data = ts_df.iloc[split_point:]
        
        if len(test_data) == 0:
            return {'note': 'Insufficient data for validation'}
        
        # Generate forecast for test period (simplified)
        test_periods = len(test_data)
        x_train = np.arange(len(train_data))
        y_train = train_data[target_column].values
        
        # Fit model on training data
        slope, intercept = np.polyfit(x_train, y_train, 1)
        
        # Predict test period
        x_test = np.arange(len(train_data), len(train_data) + test_periods)
        y_pred = slope * x_test + intercept
        y_true = test_data[target_column].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'validation_points': len(test_data)
        }
    
    except Exception as e:
        logger.error(f"Error calculating forecast metrics: {str(e)}")
        return {'error': str(e)}

def combine_historical_and_forecast(ts_df, forecast_result, datetime_column, target_column):
    """Combine historical and forecast data into single DataFrame"""
    try:
        # Historical data
        historical = ts_df[[datetime_column, target_column]].copy()
        historical['type'] = 'historical'
        historical['lower_ci'] = historical[target_column]
        historical['upper_ci'] = historical[target_column]
        
        # Forecast data
        forecast_df = pd.DataFrame({
            datetime_column: forecast_result['forecast'].index,
            target_column: forecast_result['forecast'].values,
            'type': 'forecast'
        })
        
        # Add confidence intervals if available
        if 'confidence_intervals' in forecast_result:
            ci = forecast_result['confidence_intervals']
            forecast_df['lower_ci'] = ci['lower']
            forecast_df['upper_ci'] = ci['upper']
        else:
            forecast_df['lower_ci'] = forecast_df[target_column]
            forecast_df['upper_ci'] = forecast_df[target_column]
        
        # Combine
        combined_df = pd.concat([historical, forecast_df], ignore_index=True)
        
        return combined_df
    
    except Exception as e:
        logger.error(f"Error combining historical and forecast data: {str(e)}")
        return ts_df

def generate_time_series_insights(ts_analysis, forecast_result, metrics, model, client):
    """Generate AI insights about time series forecast"""
    try:
        insights = []
        
        if client:
            # Prepare summary for AI
            summary = {
                'data_points': ts_analysis.get('data_points', 0),
                'trend_slope': ts_analysis.get('trend_slope', 0),
                'forecast_trend': forecast_result.get('forecast_summary', {}).get('forecast_trend', 'unknown'),
                'metrics': metrics,
                'model_type': forecast_result.get('model_type', 'unknown')
            }
            
            prompt = f"""
            Analyze the time series forecasting results and provide insights.
            
            Time Series Analysis:
            {json.dumps(summary, indent=2)}
            
            Provide 3-5 strategic insights about:
            1. Quality and reliability of the forecast
            2. Trends and patterns identified
            3. Business implications
            4. Recommendations for improvement
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Insight title",
                        "description": "Detailed explanation and recommendation"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a time series forecasting expert providing insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Fallback insights
            data_points = ts_analysis.get('data_points', 0)
            trend_slope = ts_analysis.get('trend_slope', 0)
            
            insights = [
                {
                    "title": "Forecast Generated",
                    "description": f"Successfully generated forecast based on {data_points} historical data points."
                },
                {
                    "title": "Trend Analysis",
                    "description": f"Data shows {'positive' if trend_slope > 0 else 'negative' if trend_slope < 0 else 'neutral'} trend with slope of {trend_slope:.4f}."
                },
                {
                    "title": "Model Performance",
                    "description": f"Forecast accuracy metrics: MAPE = {metrics.get('mape', 'N/A'):.2f}%" if 'mape' in metrics else "Model validation completed."
                }
            ]
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating time series insights: {str(e)}")
        return [{"title": "Forecast Complete", "description": "Time series forecasting completed successfully."}]
