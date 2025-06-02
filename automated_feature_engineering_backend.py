"""
Automated Feature Engineering Backend
Enterprise-grade Flask backend for AI-powered feature engineering platform
"""

from flask import Flask, request, jsonify, session, redirect, url_for, send_file, render_template_string
import pandas as pd
import numpy as np
import json
import uuid
import time
import logging
import os
import re
import io
import base64
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import tempfile
import shutil
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle

# Scientific computing and ML libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform

# Visualization libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineeringEngine:
    """
    Advanced Feature Engineering Engine with AI-powered capabilities
    """
    
    def __init__(self, openai_client=None):
        self.client = openai_client
        self.feature_generators = {
            'statistical': self._create_statistical_features,
            'temporal': self._create_temporal_features,
            'categorical': self._create_categorical_features,
            'interaction': self._create_interaction_features,
            'text': self._create_text_features,
            'aggregation': self._create_aggregation_features,
            'polynomial': self._create_polynomial_features,
            'binning': self._create_binning_features,
            'clustering': self._create_clustering_features,
            'frequency': self._create_frequency_features
        }
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset analysis for feature engineering recommendations"""
        try:
            analysis = {
                'basic_info': self._get_basic_info(df),
                'column_analysis': self._analyze_columns(df),
                'data_quality': self._assess_data_quality(df),
                'feature_recommendations': self._generate_recommendations(df),
                'correlation_analysis': self._analyze_correlations(df),
                'distribution_analysis': self._analyze_distributions(df)
            }
            return analysis
        except Exception as e:
            logger.error(f"Error in dataset analysis: {str(e)}")
            raise
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
    
    def _analyze_columns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each column for feature engineering potential"""
        columns_info = []
        
        for col in df.columns:
            try:
                col_info = {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': float(df[col].isnull().sum() / len(df) * 100),
                    'unique_count': int(df[col].nunique()),
                    'unique_percentage': float(df[col].nunique() / len(df) * 100),
                    'fe_potential': self._assess_fe_potential(df[col]),
                    'recommended_techniques': self._recommend_techniques(df[col]),
                    'data_type_category': self._categorize_data_type(df[col])
                }
                
                # Add type-specific analysis
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_info.update(self._analyze_numeric_column(df[col]))
                elif pd.api.types.is_object_dtype(df[col]):
                    col_info.update(self._analyze_text_column(df[col]))
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    col_info.update(self._analyze_datetime_column(df[col]))
                
                columns_info.append(col_info)
                
            except Exception as e:
                logger.warning(f"Error analyzing column {col}: {str(e)}")
                continue
        
        return columns_info
    
    def _assess_fe_potential(self, series: pd.Series) -> str:
        """Assess feature engineering potential of a column"""
        try:
            missing_pct = series.isnull().sum() / len(series) * 100
            unique_count = series.nunique()
            
            if missing_pct > 70:
                return "Low"
            
            if pd.api.types.is_numeric_dtype(series):
                if unique_count < 5:
                    return "Medium"
                elif missing_pct > 50:
                    return "Low"
                elif series.std() == 0:
                    return "Low"
                else:
                    return "High"
            elif pd.api.types.is_object_dtype(series):
                if unique_count > len(series) * 0.95:
                    return "Medium"  # Likely text data
                elif unique_count < 20:
                    return "High"  # Good for categorical encoding
                else:
                    return "Medium"
            elif pd.api.types.is_datetime64_any_dtype(series):
                return "High"  # Always good for temporal features
            else:
                return "Medium"
        except:
            return "Low"
    
    def _recommend_techniques(self, series: pd.Series) -> List[str]:
        """Recommend feature engineering techniques for a column"""
        techniques = []
        
        try:
            if pd.api.types.is_numeric_dtype(series):
                techniques.extend(['statistical', 'interaction', 'polynomial', 'binning'])
                if series.nunique() > 10:
                    techniques.append('clustering')
            
            elif pd.api.types.is_object_dtype(series):
                techniques.extend(['categorical', 'frequency'])
                # Check if it might be text
                sample_text = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
                if len(sample_text.split()) > 3:  # Likely text
                    techniques.append('text')
                # Check if it might be datetime
                try:
                    pd.to_datetime(series.dropna().head(10))
                    techniques.append('temporal')
                except:
                    pass
            
            elif pd.api.types.is_datetime64_any_dtype(series):
                techniques.append('temporal')
            
            # Always consider aggregation if there are other columns
            techniques.append('aggregation')
            
        except Exception as e:
            logger.warning(f"Error recommending techniques: {str(e)}")
        
        return list(set(techniques))
    
    def _categorize_data_type(self, series: pd.Series) -> str:
        """Categorize the data type for better understanding"""
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 10:
                return "categorical_numeric"
            elif series.dtype in ['int64', 'int32']:
                return "integer"
            else:
                return "continuous"
        elif pd.api.types.is_object_dtype(series):
            # Try to determine if it's categorical or text
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1:
                return "categorical"
            else:
                return "text"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        else:
            return "other"
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column statistics"""
        try:
            return {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'outliers_count': int(self._count_outliers(series)),
                'zero_count': int((series == 0).sum()),
                'negative_count': int((series < 0).sum())
            }
        except:
            return {}
    
    def _analyze_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text column characteristics"""
        try:
            text_series = series.astype(str)
            return {
                'avg_length': float(text_series.str.len().mean()),
                'max_length': int(text_series.str.len().max()),
                'min_length': int(text_series.str.len().min()),
                'avg_words': float(text_series.str.split().str.len().mean()),
                'contains_numbers': int(text_series.str.contains(r'\d').sum()),
                'contains_special_chars': int(text_series.str.contains(r'[^a-zA-Z0-9\s]').sum()),
                'is_likely_categorical': series.nunique() / len(series) < 0.1
            }
        except:
            return {}
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column characteristics"""
        try:
            return {
                'min_date': str(series.min()),
                'max_date': str(series.max()),
                'date_range_days': int((series.max() - series.min()).days),
                'has_time_component': bool(series.dt.hour.nunique() > 1),
                'frequency': self._infer_frequency(series)
            }
        except:
            return {}
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return ((series < lower_bound) | (series > upper_bound)).sum()
        except:
            return 0
    
    def _infer_frequency(self, series: pd.Series) -> str:
        """Infer the frequency of datetime series"""
        try:
            if len(series.dropna()) < 2:
                return "unknown"
            
            diff = series.dropna().diff().mode()
            if len(diff) > 0:
                days = diff.iloc[0].days
                if days == 1:
                    return "daily"
                elif days == 7:
                    return "weekly"
                elif 28 <= days <= 31:
                    return "monthly"
                elif 365 <= days <= 366:
                    return "yearly"
                else:
                    return f"{days}_days"
            return "irregular"
        except:
            return "unknown"
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        try:
            return {
                'completeness_score': float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
                'consistency_score': self._calculate_consistency_score(df),
                'validity_score': self._calculate_validity_score(df),
                'uniqueness_score': self._calculate_uniqueness_score(df),
                'overall_quality': self._calculate_overall_quality(df)
            }
        except Exception as e:
            logger.warning(f"Error assessing data quality: {str(e)}")
            return {}
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        try:
            # Simple consistency check based on data type consistency
            consistency_issues = 0
            total_checks = 0
            
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    # Check for mixed types in object columns
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        types = set(type(x).__name__ for x in sample)
                        if len(types) > 1:
                            consistency_issues += 1
                        total_checks += 1
            
            if total_checks == 0:
                return 100.0
            
            return max(0, (1 - consistency_issues / total_checks) * 100)
        except:
            return 85.0  # Default score
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate data validity score"""
        try:
            validity_issues = 0
            total_checks = 0
            
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check for infinite values
                    if np.isinf(df[col]).any():
                        validity_issues += 1
                    total_checks += 1
                elif pd.api.types.is_object_dtype(df[col]):
                    # Check for empty strings
                    if (df[col] == '').any():
                        validity_issues += 1
                    total_checks += 1
            
            if total_checks == 0:
                return 100.0
            
            return max(0, (1 - validity_issues / total_checks) * 100)
        except:
            return 90.0  # Default score
    
    def _calculate_uniqueness_score(self, df: pd.DataFrame) -> float:
        """Calculate data uniqueness score"""
        try:
            duplicate_ratio = df.duplicated().sum() / len(df)
            return max(0, (1 - duplicate_ratio) * 100)
        except:
            return 95.0  # Default score
    
    def _calculate_overall_quality(self, df: pd.DataFrame) -> str:
        """Calculate overall data quality rating"""
        try:
            quality_metrics = self._assess_data_quality(df)
            avg_score = np.mean([
                quality_metrics.get('completeness_score', 85),
                quality_metrics.get('consistency_score', 85),
                quality_metrics.get('validity_score', 90),
                quality_metrics.get('uniqueness_score', 95)
            ])
            
            if avg_score >= 90:
                return "Excellent"
            elif avg_score >= 80:
                return "Good"
            elif avg_score >= 70:
                return "Fair"
            else:
                return "Poor"
        except:
            return "Good"
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Generate AI-powered recommendations for feature engineering"""
        recommendations = []
        
        try:
            # Analyze data characteristics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            
            # Generate recommendations based on data types
            if len(numeric_cols) > 1:
                recommendations.append({
                    'type': 'interaction',
                    'title': 'Feature Interactions Recommended',
                    'description': f'With {len(numeric_cols)} numeric columns, creating interaction features could reveal hidden patterns.',
                    'priority': 'high'
                })
            
            if len(categorical_cols) > 0:
                high_cardinality = [col for col in categorical_cols if df[col].nunique() > 20]
                if high_cardinality:
                    recommendations.append({
                        'type': 'categorical',
                        'title': 'High Cardinality Encoding Needed',
                        'description': f'Columns {high_cardinality[:3]} have high cardinality. Consider target encoding or embedding.',
                        'priority': 'medium'
                    })
            
            if len(datetime_cols) > 0:
                recommendations.append({
                    'type': 'temporal',
                    'title': 'Temporal Features Available',
                    'description': f'Extract time-based features from {len(datetime_cols)} datetime column(s) for better insights.',
                    'priority': 'high'
                })
            
            # Check for skewed distributions
            skewed_cols = []
            for col in numeric_cols:
                if abs(df[col].skew()) > 1:
                    skewed_cols.append(col)
            
            if skewed_cols:
                recommendations.append({
                    'type': 'statistical',
                    'title': 'Skewed Distributions Detected',
                    'description': f'Apply log transformation or power transforms to {len(skewed_cols)} skewed column(s).',
                    'priority': 'medium'
                })
            
            # Check for potential text data
            text_cols = []
            for col in categorical_cols:
                if df[col].astype(str).str.len().mean() > 20:
                    text_cols.append(col)
            
            if text_cols:
                recommendations.append({
                    'type': 'text',
                    'title': 'Text Features Detected',
                    'description': f'Extract text features from {len(text_cols)} column(s) with long text content.',
                    'priority': 'medium'
                })
            
            # Missing value recommendations
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                recommendations.append({
                    'type': 'preprocessing',
                    'title': 'Missing Value Treatment',
                    'description': f'Handle missing values in {len(missing_cols)} column(s) before feature engineering.',
                    'priority': 'high'
                })
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) < 2:
                return {'correlation_matrix': {}, 'high_correlations': []}
            
            corr_matrix = numeric_df.corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        high_correlations.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_correlations
            }
        except Exception as e:
            logger.warning(f"Error analyzing correlations: {str(e)}")
            return {'correlation_matrix': {}, 'high_correlations': []}
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            distributions = {}
            
            for col in numeric_cols:
                try:
                    # Test for normality
                    _, p_value = stats.normaltest(df[col].dropna())
                    is_normal = p_value > 0.05
                    
                    distributions[col] = {
                        'is_normal': bool(is_normal),
                        'skewness': float(df[col].skew()),
                        'kurtosis': float(df[col].kurtosis()),
                        'distribution_type': self._identify_distribution(df[col])
                    }
                except:
                    continue
            
            return distributions
        except Exception as e:
            logger.warning(f"Error analyzing distributions: {str(e)}")
            return {}
    
    def _identify_distribution(self, series: pd.Series) -> str:
        """Identify the likely distribution type"""
        try:
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return "normal"
            elif skewness > 1:
                return "right_skewed"
            elif skewness < -1:
                return "left_skewed"
            elif kurtosis > 3:
                return "heavy_tailed"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def generate_features(self, df: pd.DataFrame, selected_columns: List[str], 
                         feature_types: List[str], processing_mode: str = 'intelligent') -> Tuple[pd.DataFrame, List[Dict]]:
        """Generate features using selected techniques"""
        try:
            enhanced_df = df.copy()
            feature_info = []
            
            # Filter to selected columns
            selected_df = df[selected_columns].copy()
            
            # Process each feature type
            for feature_type in feature_types:
                if feature_type in self.feature_generators:
                    try:
                        logger.info(f"Generating {feature_type} features...")
                        features, info = self.feature_generators[feature_type](selected_df)
                        
                        if not features.empty:
                            enhanced_df = pd.concat([enhanced_df, features], axis=1)
                            feature_info.extend(info)
                            logger.info(f"Generated {len(features.columns)} {feature_type} features")
                        
                    except Exception as e:
                        logger.error(f"Error generating {feature_type} features: {str(e)}")
                        continue
            
            # Clean the enhanced dataframe
            enhanced_df = self._clean_enhanced_dataframe(enhanced_df)
            
            return enhanced_df, feature_info
            
        except Exception as e:
            logger.error(f"Error in feature generation: {str(e)}")
            raise
    
    def _create_statistical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create comprehensive statistical features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                try:
                    # Rolling statistics with different windows
                    for window in [3, 5, 7]:
                        features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                        features[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                        features[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window, min_periods=1).median()
                        features[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                        features[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                        
                        feature_info.extend([
                            {'name': f'{col}_rolling_mean_{window}', 'type': 'statistical', 'description': f'{window}-period rolling mean of {col}'},
                            {'name': f'{col}_rolling_std_{window}', 'type': 'statistical', 'description': f'{window}-period rolling std of {col}'},
                            {'name': f'{col}_rolling_median_{window}', 'type': 'statistical', 'description': f'{window}-period rolling median of {col}'},
                            {'name': f'{col}_rolling_min_{window}', 'type': 'statistical', 'description': f'{window}-period rolling min of {col}'},
                            {'name': f'{col}_rolling_max_{window}', 'type': 'statistical', 'description': f'{window}-period rolling max of {col}'}
                        ])
                    
                    # Lag features
                    for lag in [1, 2, 3, 5]:
                        features[f'{col}_lag_{lag}'] = df[col].shift(lag)
                        feature_info.append({'name': f'{col}_lag_{lag}', 'type': 'statistical', 'description': f'{lag}-period lag of {col}'})
                    
                    # Cumulative features
                    features[f'{col}_cumsum'] = df[col].cumsum()
                    features[f'{col}_cummax'] = df[col].cummax()
                    features[f'{col}_cummin'] = df[col].cummin()
                    features[f'{col}_cumprod'] = df[col].cumprod()
                    
                    # Percentage change
                    features[f'{col}_pct_change'] = df[col].pct_change().fillna(0)
                    features[f'{col}_pct_change_abs'] = df[col].pct_change().abs().fillna(0)
                    
                    # Normalization features
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val > 0:
                        features[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                    
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        features[f'{col}_minmax'] = (df[col] - min_val) / (max_val - min_val)
                    
                    # Rank and quantile features
                    features[f'{col}_rank'] = df[col].rank(pct=True)
                    features[f'{col}_quantile'] = pd.qcut(df[col], q=10, labels=False, duplicates='drop')
                    
                    # Statistical transformations
                    if (df[col] > 0).all():
                        features[f'{col}_log'] = np.log1p(df[col])
                        features[f'{col}_sqrt'] = np.sqrt(df[col])
                    
                    features[f'{col}_squared'] = df[col] ** 2
                    features[f'{col}_cubed'] = df[col] ** 3
                    
                    # Outlier indicators
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    features[f'{col}_is_outlier'] = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).astype(int)
                    
                    feature_info.extend([
                        {'name': f'{col}_cumsum', 'type': 'statistical', 'description': f'Cumulative sum of {col}'},
                        {'name': f'{col}_cummax', 'type': 'statistical', 'description': f'Cumulative maximum of {col}'},
                        {'name': f'{col}_cummin', 'type': 'statistical', 'description': f'Cumulative minimum of {col}'},
                        {'name': f'{col}_pct_change', 'type': 'statistical', 'description': f'Percentage change of {col}'},
                        {'name': f'{col}_zscore', 'type': 'statistical', 'description': f'Z-score normalized {col}'},
                        {'name': f'{col}_minmax', 'type': 'statistical', 'description': f'Min-max normalized {col}'},
                        {'name': f'{col}_rank', 'type': 'statistical', 'description': f'Percentile rank of {col}'},
                        {'name': f'{col}_is_outlier', 'type': 'statistical', 'description': f'Outlier indicator for {col}'}
                    ])
                    
                except Exception as e:
                    logger.warning(f"Error creating statistical features for {col}: {str(e)}")
                    continue
        
        return features, feature_info
    
    def _create_temporal_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create comprehensive temporal features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        # Identify datetime columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    sample_data = df[col].dropna().head(100)
                    if len(sample_data) > 0:
                        pd.to_datetime(sample_data, errors='raise')
                        datetime_cols.append(col)
                except:
                    continue
        
        for col in datetime_cols:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    dt_series = pd.to_datetime(df[col], errors='coerce')
                else:
                    dt_series = df[col]
                
                # Basic temporal features
                features[f'{col}_year'] = dt_series.dt.year
                features[f'{col}_month'] = dt_series.dt.month
                features[f'{col}_day'] = dt_series.dt.day
                features[f'{col}_dayofweek'] = dt_series.dt.dayofweek
                features[f'{col}_dayofyear'] = dt_series.dt.dayofyear
                features[f'{col}_week'] = dt_series.dt.isocalendar().week
                features[f'{col}_quarter'] = dt_series.dt.quarter
                features[f'{col}_hour'] = dt_series.dt.hour
                features[f'{col}_minute'] = dt_series.dt.minute
                features[f'{col}_second'] = dt_series.dt.second
                
                # Boolean temporal features
                features[f'{col}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
                features[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
                features[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
                features[f'{col}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
                features[f'{col}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
                features[f'{col}_is_year_start'] = dt_series.dt.is_year_start.astype(int)
                features[f'{col}_is_year_end'] = dt_series.dt.is_year_end.astype(int)
                
                # Cyclical features (sine/cosine encoding)
                features[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                features[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                features[f'{col}_day_sin'] = np.sin(2 * np.pi * dt_series.dt.day / 31)
                features[f'{col}_day_cos'] = np.cos(2 * np.pi * dt_series.dt.day / 31)
                features[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                features[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
                
                # Time since features
                features[f'{col}_days_since_epoch'] = (dt_series - pd.Timestamp('1970-01-01')).dt.days
                features[f'{col}_seconds_since_midnight'] = dt_series.dt.hour * 3600 + dt_series.dt.minute * 60 + dt_series.dt.second
                
                # Business day features
                features[f'{col}_is_business_day'] = (dt_series.dt.dayofweek < 5).astype(int)
                
                # Season features
                def get_season(month):
                    if month in [12, 1, 2]:
                        return 0  # Winter
                    elif month in [3, 4, 5]:
                        return 1  # Spring
                    elif month in [6, 7, 8]:
                        return 2  # Summer
                    else:
                        return 3  # Fall
                
                features[f'{col}_season'] = dt_series.dt.month.apply(get_season)
                
                # Time differences (if multiple datetime columns)
                if len(datetime_cols) > 1:
                    features[f'{col}_timestamp'] = dt_series.astype(np.int64) // 10**9
                
                feature_info.extend([
                    {'name': f'{col}_year', 'type': 'temporal', 'description': f'Year from {col}'},
                    {'name': f'{col}_month', 'type': 'temporal', 'description': f'Month from {col}'},
                    {'name': f'{col}_day', 'type': 'temporal', 'description': f'Day from {col}'},
                    {'name': f'{col}_dayofweek', 'type': 'temporal', 'description': f'Day of week from {col}'},
                    {'name': f'{col}_quarter', 'type': 'temporal', 'description': f'Quarter from {col}'},
                    {'name': f'{col}_hour', 'type': 'temporal', 'description': f'Hour from {col}'},
                    {'name': f'{col}_is_weekend', 'type': 'temporal', 'description': f'Weekend indicator from {col}'},
                    {'name': f'{col}_is_business_day', 'type': 'temporal', 'description': f'Business day indicator from {col}'},
                    {'name': f'{col}_season', 'type': 'temporal', 'description': f'Season from {col}'},
                    {'name': f'{col}_month_sin', 'type': 'temporal', 'description': f'Cyclical month (sine) from {col}'},
                    {'name': f'{col}_month_cos', 'type': 'temporal', 'description': f'Cyclical month (cosine) from {col}'}
                ])
                
            except Exception as e:
                logger.warning(f"Error creating temporal features for {col}: {str(e)}")
                continue
        
        return features, feature_info
    
    def _create_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create comprehensive categorical encoding features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].notna().sum() > 0:
                try:
                    unique_count = df[col].nunique()
                    
                    # One-hot encoding for low cardinality
                    if unique_count <= 15:
                        dummies = pd.get_dummies(df[col], prefix=f'{col}_onehot', dummy_na=True)
                        features = pd.concat([features, dummies], axis=1)
                        
                        for dummy_col in dummies.columns:
                            feature_info.append({
                                'name': dummy_col,
                                'type': 'categorical',
                                'description': f'One-hot encoding for {col}'
                            })
                    
                    # Label encoding
                    le = LabelEncoder()
                    features[f'{col}_label_encoded'] = le.fit_transform(df[col].fillna('missing'))
                    
                    # Frequency encoding
                    freq_map = df[col].value_counts().to_dict()
                    features[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
                    
                    # Frequency percentage
                    total_count = len(df)
                    features[f'{col}_frequency_pct'] = features[f'{col}_frequency'] / total_count
                    
                    # Rare category indicator
                    rare_threshold = 0.01  # 1% threshold
                    rare_categories = freq_map.keys() if len(freq_map) == 0 else [k for k, v in freq_map.items() if v / total_count < rare_threshold]
                    features[f'{col}_is_rare'] = df[col].isin(rare_categories).astype(int)
                    
                    # Target encoding (using mean of label encoded values as proxy)
                    if unique_count > 2:
                        target_map = df.groupby(col)[f'{col}_label_encoded'].mean().to_dict()
                        features[f'{col}_target_encoded'] = df[col].map(target_map).fillna(0)
                        
                        feature_info.append({
                            'name': f'{col}_target_encoded',
                            'type': 'categorical',
                            'description': f'Target encoding of {col}'
                        })
                    
                    # Binary encoding for high cardinality
                    if unique_count > 15:
                        label_encoded = le.fit_transform(df[col].fillna('missing'))
                        binary_features = self._create_binary_encoding(label_encoded, f'{col}_binary')
                        features = pd.concat([features, binary_features], axis=1)
                        
                        for binary_col in binary_features.columns:
                            feature_info.append({
                                'name': binary_col,
                                'type': 'categorical',
                                'description': f'Binary encoding for {col}'
                            })
                    
                    # Count encoding (number of occurrences)
                    count_map = df[col].value_counts().to_dict()
                    features[f'{col}_count'] = df[col].map(count_map).fillna(0)
                    
                    # Rank encoding
                    rank_map = df[col].value_counts().rank(ascending=False).to_dict()
                    features[f'{col}_rank'] = df[col].map(rank_map).fillna(0)
                    
                    feature_info.extend([
                        {'name': f'{col}_label_encoded', 'type': 'categorical', 'description': f'Label encoding of {col}'},
                        {'name': f'{col}_frequency', 'type': 'categorical', 'description': f'Frequency encoding of {col}'},
                        {'name': f'{col}_frequency_pct', 'type': 'categorical', 'description': f'Frequency percentage of {col}'},
                        {'name': f'{col}_is_rare', 'type': 'categorical', 'description': f'Rare category indicator for {col}'},
                        {'name': f'{col}_count', 'type': 'categorical', 'description': f'Count encoding of {col}'},
                        {'name': f'{col}_rank', 'type': 'categorical', 'description': f'Rank encoding of {col}'}
                    ])
                    
                except Exception as e:
                    logger.warning(f"Error creating categorical features for {col}: {str(e)}")
                    continue
        
        return features, feature_info
    
    def _create_binary_encoding(self, label_encoded: np.ndarray, prefix: str) -> pd.DataFrame:
        """Create binary encoding for high cardinality categorical variables"""
        max_val = max(label_encoded) if len(label_encoded) > 0 else 0
        n_bits = int(np.ceil(np.log2(max_val + 1))) if max_val > 0 else 1
        
        binary_features = pd.DataFrame()
        for i in range(n_bits):
            binary_features[f'{prefix}_{i}'] = (label_encoded >> i) & 1
        
        return binary_features
    
    def _create_interaction_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create comprehensive feature interaction features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Limit to prevent feature explosion
        numeric_cols = numeric_cols[:6]
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if df[col1].notna().sum() > 0 and df[col2].notna().sum() > 0:
                    try:
                        # Basic arithmetic operations
                        features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                        features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                        features[f'{col2}_minus_{col1}'] = df[col2] - df[col1]
                        
                        # Division (with protection against division by zero)
                        denominator1 = df[col2].replace(0, np.nan)
                        denominator2 = df[col1].replace(0, np.nan)
                        features[f'{col1}_div_{col2}'] = (df[col1] / denominator1).fillna(0)
                        features[f'{col2}_div_{col1}'] = (df[col2] / denominator2).fillna(0)
                        
                        # Power operations
                        features[f'{col1}_pow_{col2}'] = np.power(np.abs(df[col1]), np.abs(df[col2]) % 3)  # Limit power to prevent overflow
                        
                        # Distance metrics
                        features[f'{col1}_{col2}_euclidean'] = np.sqrt(df[col1]**2 + df[col2]**2)
                        features[f'{col1}_{col2}_manhattan'] = np.abs(df[col1]) + np.abs(df[col2])
                        features[f'{col1}_{col2}_chebyshev'] = np.maximum(np.abs(df[col1]), np.abs(df[col2]))
                        
                        # Statistical interactions
                        features[f'{col1}_{col2}_mean'] = (df[col1] + df[col2]) / 2
                        features[f'{col1}_{col2}_max'] = np.maximum(df[col1], df[col2])
                        features[f'{col1}_{col2}_min'] = np.minimum(df[col1], df[col2])
                        features[f'{col1}_{col2}_range'] = features[f'{col1}_{col2}_max'] - features[f'{col1}_{col2}_min']
                        
                        # Comparison features
                        features[f'{col1}_gt_{col2}'] = (df[col1] > df[col2]).astype(int)
                        features[f'{col1}_eq_{col2}'] = (df[col1] == df[col2]).astype(int)
                        
                        # Ratio features
                        sum_cols = df[col1] + df[col2]
                        features[f'{col1}_ratio_sum'] = df[col1] / (sum_cols + 1e-8)
                        features[f'{col2}_ratio_sum'] = df[col2] / (sum_cols + 1e-8)
                        
                        feature_info.extend([
                            {'name': f'{col1}_x_{col2}', 'type': 'interaction', 'description': f'Product of {col1} and {col2}'},
                            {'name': f'{col1}_plus_{col2}', 'type': 'interaction', 'description': f'Sum of {col1} and {col2}'},
                            {'name': f'{col1}_minus_{col2}', 'type': 'interaction', 'description': f'Difference {col1} - {col2}'},
                            {'name': f'{col1}_div_{col2}', 'type': 'interaction', 'description': f'Ratio {col1} / {col2}'},
                            {'name': f'{col1}_{col2}_euclidean', 'type': 'interaction', 'description': f'Euclidean distance between {col1} and {col2}'},
                            {'name': f'{col1}_{col2}_manhattan', 'type': 'interaction', 'description': f'Manhattan distance between {col1} and {col2}'},
                            {'name': f'{col1}_gt_{col2}', 'type': 'interaction', 'description': f'Comparison: {col1} > {col2}'},
                            {'name': f'{col1}_ratio_sum', 'type': 'interaction', 'description': f'Ratio of {col1} to sum of {col1} and {col2}'}
                        ])
                        
                    except Exception as e:
                        logger.warning(f"Error creating interaction features for {col1} and {col2}: {str(e)}")
                        continue
        
        # Polynomial features for individual columns
        for col in numeric_cols:
            try:
                features[f'{col}_squared'] = df[col] ** 2
                features[f'{col}_cubed'] = df[col] ** 3
                features[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                
                feature_info.extend([
                    {'name': f'{col}_squared', 'type': 'interaction', 'description': f'Square of {col}'},
                    {'name': f'{col}_cubed', 'type': 'interaction', 'description': f'Cube of {col}'},
                    {'name': f'{col}_sqrt', 'type': 'interaction', 'description': f'Square root of {col}'}
                ])
                
            except Exception as e:
                logger.warning(f"Error creating polynomial features for {col}: {str(e)}")
                continue
        
        return features, feature_info
    
    def _create_text_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create comprehensive text-based features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            if df[col].notna().sum() > 0:
                try:
                    text_series = df[col].astype(str)
                    
                    # Basic text statistics
                    features[f'{col}_word_count'] = text_series.str.split().str.len()
                    features[f'{col}_char_count'] = text_series.str.len()
                    features[f'{col}_unique_words'] = text_series.apply(lambda x: len(set(x.split())))
                    features[f'{col}_avg_word_length'] = text_series.apply(
                        lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0
                    )
                    
                    # Character analysis
                    features[f'{col}_uppercase_count'] = text_series.str.count('[A-Z]')
                    features[f'{col}_lowercase_count'] = text_series.str.count('[a-z]')
                    features[f'{col}_digit_count'] = text_series.str.count(r'\d')
                    features[f'{col}_punctuation_count'] = text_series.str.count(r'[^\w\s]')
                    features[f'{col}_whitespace_count'] = text_series.str.count(r'\s')
                    features[f'{col}_special_char_count'] = text_series.str.count(r'[!@#$%^&*(),.?":{}|<>]')
                    
                    # Specific punctuation
                    features[f'{col}_exclamation_count'] = text_series.str.count('!')
                    features[f'{col}_question_count'] = text_series.str.count(r'\?')
                    features[f'{col}_period_count'] = text_series.str.count(r'\.')
                    features[f'{col}_comma_count'] = text_series.str.count(',')
                    
                    # Ratios
                    char_count = features[f'{col}_char_count']
                    features[f'{col}_uppercase_ratio'] = features[f'{col}_uppercase_count'] / (char_count + 1)
                    features[f'{col}_digit_ratio'] = features[f'{col}_digit_count'] / (char_count + 1)
                    features[f'{col}_punctuation_ratio'] = features[f'{col}_punctuation_count'] / (char_count + 1)
                    
                    # Text complexity
                    features[f'{col}_sentence_count'] = text_series.str.count(r'[.!?]+')
                    features[f'{col}_avg_sentence_length'] = features[f'{col}_word_count'] / (features[f'{col}_sentence_count'] + 1)
                    
                    # Pattern detection
                    features[f'{col}_has_url'] = text_series.str.contains(
                        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$$\$$,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        regex=True, na=False
                    ).astype(int)
                    
                    features[f'{col}_has_email'] = text_series.str.contains(
                        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                        regex=True, na=False
                    ).astype(int)
                    
                    features[f'{col}_has_phone'] = text_series.str.contains(
                        r'(\+\d{1,3}[-.\s]?)?$$?\d{3}$$?[-.\s]?\d{3}[-.\s]?\d{4}',
                        regex=True, na=False
                    ).astype(int)
                    
                    features[f'{col}_has_hashtag'] = text_series.str.contains(r'#\w+', regex=True, na=False).astype(int)
                    features[f'{col}_has_mention'] = text_series.str.contains(r'@\w+', regex=True, na=False).astype(int)
                    
                    # Language patterns
                    features[f'{col}_starts_with_capital'] = text_series.str.match(r'^[A-Z]').astype(int)
                    features[f'{col}_ends_with_punctuation'] = text_series.str.match(r'.*[.!?]$').astype(int)
                    features[f'{col}_all_caps'] = text_series.str.isupper().astype(int)
                    features[f'{col}_all_lower'] = text_series.str.islower().astype(int)
                    
                    # Sentiment proxies
                    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'best', 'awesome', 'perfect']
                    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'dislike', 'poor', 'disappointing', 'annoying']
                    
                    features[f'{col}_positive_words'] = text_series.apply(
                        lambda x: sum(1 for word in positive_words if word in x.lower())
                    )
                    features[f'{col}_negative_words'] = text_series.apply(
                        lambda x: sum(1 for word in negative_words if word in x.lower())
                    )
                    features[f'{col}_sentiment_score'] = features[f'{col}_positive_words'] - features[f'{col}_negative_words']
                    
                    # Readability proxies
                    features[f'{col}_avg_syllables'] = text_series.apply(self._estimate_syllables)
                    features[f'{col}_flesch_score'] = self._calculate_flesch_score(
                        features[f'{col}_avg_sentence_length'],
                        features[f'{col}_avg_syllables']
                    )
                    
                    feature_info.extend([
                        {'name': f'{col}_word_count', 'type': 'text', 'description': f'Word count in {col}'},
                        {'name': f'{col}_char_count', 'type': 'text', 'description': f'Character count in {col}'},
                        {'name': f'{col}_unique_words', 'type': 'text', 'description': f'Unique word count in {col}'},
                        {'name': f'{col}_avg_word_length', 'type': 'text', 'description': f'Average word length in {col}'},
                        {'name': f'{col}_uppercase_ratio', 'type': 'text', 'description': f'Uppercase character ratio in {col}'},
                        {'name': f'{col}_digit_ratio', 'type': 'text', 'description': f'Digit character ratio in {col}'},
                        {'name': f'{col}_punctuation_ratio', 'type': 'text', 'description': f'Punctuation ratio in {col}'},
                        {'name': f'{col}_sentence_count', 'type': 'text', 'description': f'Sentence count in {col}'},
                        {'name': f'{col}_has_url', 'type': 'text', 'description': f'URL presence in {col}'},
                        {'name': f'{col}_has_email', 'type': 'text', 'description': f'Email presence in {col}'},
                        {'name': f'{col}_sentiment_score', 'type': 'text', 'description': f'Sentiment score of {col}'},
                        {'name': f'{col}_flesch_score', 'type': 'text', 'description': f'Readability score of {col}'}
                    ])
                    
                except Exception as e:
                    logger.warning(f"Error creating text features for {col}: {str(e)}")
                    continue
        
        return features, feature_info
    
    def _estimate_syllables(self, text: str) -> float:
        """Estimate syllable count for readability calculation"""
        try:
            words = text.lower().split()
            if not words:
                return 0
            
            syllable_count = 0
            for word in words:
                word = re.sub(r'[^a-z]', '', word)
                if not word:
                    continue
                
                # Count vowel groups
                vowels = 'aeiouy'
                prev_was_vowel = False
                count = 0
                
                for char in word:
                    is_vowel = char in vowels
                    if is_vowel and not prev_was_vowel:
                        count += 1
                    prev_was_vowel = is_vowel
                
                # Handle silent e
                if word.endswith('e'):
                    count -= 1
                
                # Every word has at least one syllable
                syllable_count += max(1, count)
            
            return syllable_count / len(words)
        except:
            return 1.0
    
    def _calculate_flesch_score(self, avg_sentence_length: pd.Series, avg_syllables: pd.Series) -> pd.Series:
        """Calculate Flesch readability score"""
        try:
            return 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        except:
            return pd.Series([50] * len(avg_sentence_length))
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create comprehensive aggregation features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Group-by aggregations
        for cat_col in categorical_cols[:3]:  # Limit to prevent explosion
            if df[cat_col].notna().sum() > 0:
                for num_col in numeric_cols[:4]:  # Limit numeric columns
                    if df[num_col].notna().sum() > 0:
                        try:
                            # Multiple aggregation functions
                            agg_functions = ['mean', 'std', 'min', 'max', 'count', 'median', 'sum']
                            group_stats = df.groupby(cat_col)[num_col].agg(agg_functions)
                            
                            for agg_func in agg_functions:
                                feature_name = f'{cat_col}_{num_col}_{agg_func}'
                                mapped_values = df[cat_col].map(group_stats[agg_func]).fillna(0)
                                features[feature_name] = mapped_values
                                
                                feature_info.append({
                                    'name': feature_name,
                                    'type': 'aggregation',
                                    'description': f'{agg_func.title()} of {num_col} grouped by {cat_col}'
                                })
                            
                            # Deviation features
                            group_means = df[cat_col].map(group_stats['mean']).fillna(0)
                            features[f'{cat_col}_{num_col}_deviation'] = df[num_col] - group_means
                            features[f'{cat_col}_{num_col}_deviation_abs'] = np.abs(features[f'{cat_col}_{num_col}_deviation'])
                            
                            # Normalized features
                            group_stds = df[cat_col].map(group_stats['std']).fillna(1)
                            features[f'{cat_col}_{num_col}_normalized'] = features[f'{cat_col}_{num_col}_deviation'] / (group_stds + 1e-8)
                            
                            # Percentile within group
                            features[f'{cat_col}_{num_col}_percentile'] = df.groupby(cat_col)[num_col].rank(pct=True)
                            
                            feature_info.extend([
                                {'name': f'{cat_col}_{num_col}_deviation', 'type': 'aggregation', 'description': f'Deviation from group mean of {num_col} by {cat_col}'},
                                {'name': f'{cat_col}_{num_col}_normalized', 'type': 'aggregation', 'description': f'Normalized {num_col} within {cat_col} groups'},
                                {'name': f'{cat_col}_{num_col}_percentile', 'type': 'aggregation', 'description': f'Percentile rank of {num_col} within {cat_col} groups'}
                            ])
                            
                        except Exception as e:
                            logger.warning(f"Error creating aggregation features for {cat_col} and {num_col}: {str(e)}")
                            continue
        
        # Cross-column aggregations
        if len(numeric_cols) >= 2:
            try:
                numeric_subset = df[numeric_cols[:6]]  # Limit columns
                
                # Row-wise statistics
                features['row_mean'] = numeric_subset.mean(axis=1)
                features['row_std'] = numeric_subset.std(axis=1).fillna(0)
                features['row_min'] = numeric_subset.min(axis=1)
                features['row_max'] = numeric_subset.max(axis=1)
                features['row_range'] = features['row_max'] - features['row_min']
                features['row_median'] = numeric_subset.median(axis=1)
                features['row_sum'] = numeric_subset.sum(axis=1)
                features['row_skew'] = numeric_subset.skew(axis=1).fillna(0)
                features['row_kurtosis'] = numeric_subset.kurtosis(axis=1).fillna(0)
                
                # Count of non-zero values
                features['row_nonzero_count'] = (numeric_subset != 0).sum(axis=1)
                features['row_positive_count'] = (numeric_subset > 0).sum(axis=1)
                features['row_negative_count'] = (numeric_subset < 0).sum(axis=1)
                
                feature_info.extend([
                    {'name': 'row_mean', 'type': 'aggregation', 'description': 'Mean across numeric columns'},
                    {'name': 'row_std', 'type': 'aggregation', 'description': 'Standard deviation across numeric columns'},
                    {'name': 'row_min', 'type': 'aggregation', 'description': 'Minimum across numeric columns'},
                    {'name': 'row_max', 'type': 'aggregation', 'description': 'Maximum across numeric columns'},
                    {'name': 'row_range', 'type': 'aggregation',  'description': 'Maximum across numeric columns'},
                    {'name': 'row_range', 'type': 'aggregation', 'description': 'Range across numeric columns'},
                    {'name': 'row_median', 'type': 'aggregation', 'description': 'Median across numeric columns'},
                    {'name': 'row_sum', 'type': 'aggregation', 'description': 'Sum across numeric columns'},
                    {'name': 'row_nonzero_count', 'type': 'aggregation', 'description': 'Count of non-zero values across numeric columns'},
                    {'name': 'row_positive_count', 'type': 'aggregation', 'description': 'Count of positive values across numeric columns'},
                    {'name': 'row_negative_count', 'type': 'aggregation', 'description': 'Count of negative values across numeric columns'}
                ])
                
            except Exception as e:
                logger.warning(f"Error creating cross-column aggregation features: {str(e)}")
        
        return features, feature_info
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create polynomial features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limit to prevent explosion
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                try:
                    # Polynomial degrees
                    for degree in [2, 3]:
                        features[f'{col}_poly_{degree}'] = df[col] ** degree
                        feature_info.append({
                            'name': f'{col}_poly_{degree}',
                            'type': 'polynomial',
                            'description': f'{col} raised to power {degree}'
                        })
                    
                    # Root transformations
                    if (df[col] >= 0).all():
                        features[f'{col}_sqrt'] = np.sqrt(df[col])
                        features[f'{col}_cbrt'] = np.cbrt(df[col])
                        
                        feature_info.extend([
                            {'name': f'{col}_sqrt', 'type': 'polynomial', 'description': f'Square root of {col}'},
                            {'name': f'{col}_cbrt', 'type': 'polynomial', 'description': f'Cube root of {col}'}
                        ])
                    
                    # Logarithmic transformations
                    if (df[col] > 0).all():
                        features[f'{col}_log'] = np.log(df[col])
                        features[f'{col}_log1p'] = np.log1p(df[col])
                        features[f'{col}_log10'] = np.log10(df[col])
                        
                        feature_info.extend([
                            {'name': f'{col}_log', 'type': 'polynomial', 'description': f'Natural logarithm of {col}'},
                            {'name': f'{col}_log1p', 'type': 'polynomial', 'description': f'Log(1 + {col})'},
                            {'name': f'{col}_log10', 'type': 'polynomial', 'description': f'Base-10 logarithm of {col}'}
                        ])
                    
                    # Exponential transformations (with clipping to prevent overflow)
                    clipped_col = np.clip(df[col], -10, 10)
                    features[f'{col}_exp'] = np.exp(clipped_col)
                    features[f'{col}_exp2'] = np.exp2(clipped_col)
                    
                    feature_info.extend([
                        {'name': f'{col}_exp', 'type': 'polynomial', 'description': f'Exponential of {col}'},
                        {'name': f'{col}_exp2', 'type': 'polynomial', 'description': f'2^{col}'}
                    ])
                    
                except Exception as e:
                    logger.warning(f"Error creating polynomial features for {col}: {str(e)}")
                    continue
        
        return features, feature_info
    
    def _create_binning_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create binning/discretization features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                try:
                    # Equal-width binning
                    for n_bins in [5, 10]:
                        features[f'{col}_bin_equal_{n_bins}'] = pd.cut(df[col], bins=n_bins, labels=False, duplicates='drop')
                        feature_info.append({
                            'name': f'{col}_bin_equal_{n_bins}',
                            'type': 'binning',
                            'description': f'{col} binned into {n_bins} equal-width bins'
                        })
                    
                    # Equal-frequency binning (quantiles)
                    for n_bins in [5, 10]:
                        try:
                            features[f'{col}_bin_quantile_{n_bins}'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
                            feature_info.append({
                                'name': f'{col}_bin_quantile_{n_bins}',
                                'type': 'binning',
                                'description': f'{col} binned into {n_bins} equal-frequency bins'
                            })
                        except:
                            continue
                    
                    # Custom binning based on statistics
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    
                    # Standard deviation based bins
                    bins = [df[col].min(), mean_val - std_val, mean_val, mean_val + std_val, df[col].max()]
                    features[f'{col}_bin_std'] = pd.cut(df[col], bins=bins, labels=False, duplicates='drop')
                    
                    # Percentile-based bins
                    percentiles = [0, 25, 50, 75, 100]
                    bins = [df[col].quantile(p/100) for p in percentiles]
                    features[f'{col}_bin_percentile'] = pd.cut(df[col], bins=bins, labels=False, duplicates='drop')
                    
                    feature_info.extend([
                        {'name': f'{col}_bin_std', 'type': 'binning', 'description': f'{col} binned by standard deviation'},
                        {'name': f'{col}_bin_percentile', 'type': 'binning', 'description': f'{col} binned by percentiles'}
                    ])
                    
                except Exception as e:
                    logger.warning(f"Error creating binning features for {col}: {str(e)}")
                    continue
        
        return features, feature_info
    
    def _create_clustering_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create clustering-based features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return features, feature_info
            
            # Use subset of numeric columns to prevent computational issues
            numeric_subset = df[numeric_cols[:5]].fillna(0)
            
            if len(numeric_subset) > 10:  # Only if we have enough data points
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler
                
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numeric_subset)
                
                # K-means clustering with different k values
                for k in [3, 5, 8]:
                    try:
                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans.fit_predict(scaled_data)
                        features[f'kmeans_cluster_{k}'] = cluster_labels
                        
                        # Distance to cluster centers
                        distances = kmeans.transform(scaled_data)
                        features[f'kmeans_distance_min_{k}'] = distances.min(axis=1)
                        features[f'kmeans_distance_mean_{k}'] = distances.mean(axis=1)
                        
                        feature_info.extend([
                            {'name': f'kmeans_cluster_{k}', 'type': 'clustering', 'description': f'K-means cluster assignment (k={k})'},
                            {'name': f'kmeans_distance_min_{k}', 'type': 'clustering', 'description': f'Distance to nearest cluster center (k={k})'},
                            {'name': f'kmeans_distance_mean_{k}', 'type': 'clustering', 'description': f'Mean distance to all cluster centers (k={k})'}
                        ])
                        
                    except Exception as e:
                        logger.warning(f"Error in K-means clustering with k={k}: {str(e)}")
                        continue
                
        except Exception as e:
            logger.warning(f"Error creating clustering features: {str(e)}")
        
        return features, feature_info
    
    def _create_frequency_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """Create frequency-based features"""
        features = pd.DataFrame(index=df.index)
        feature_info = []
        
        # For all columns, create frequency-based features
        for col in df.columns:
            if df[col].notna().sum() > 0:
                try:
                    # Value frequency
                    value_counts = df[col].value_counts()
                    features[f'{col}_frequency'] = df[col].map(value_counts).fillna(0)
                    
                    # Frequency rank
                    freq_rank = value_counts.rank(ascending=False, method='dense')
                    features[f'{col}_freq_rank'] = df[col].map(freq_rank).fillna(0)
                    
                    # Relative frequency
                    total_count = len(df)
                    features[f'{col}_rel_frequency'] = features[f'{col}_frequency'] / total_count
                    
                    # Frequency percentile
                    features[f'{col}_freq_percentile'] = features[f'{col}_frequency'].rank(pct=True)
                    
                    # Rare value indicator
                    rare_threshold = 0.01  # 1%
                    rare_values = value_counts[value_counts / total_count < rare_threshold].index
                    features[f'{col}_is_rare'] = df[col].isin(rare_values).astype(int)
                    
                    # Common value indicator
                    common_threshold = 0.1  # 10%
                    common_values = value_counts[value_counts / total_count > common_threshold].index
                    features[f'{col}_is_common'] = df[col].isin(common_values).astype(int)
                    
                    feature_info.extend([
                        {'name': f'{col}_frequency', 'type': 'frequency', 'description': f'Frequency of values in {col}'},
                        {'name': f'{col}_freq_rank', 'type': 'frequency', 'description': f'Frequency rank of values in {col}'},
                        {'name': f'{col}_rel_frequency', 'type': 'frequency', 'description': f'Relative frequency of values in {col}'},
                        {'name': f'{col}_is_rare', 'type': 'frequency', 'description': f'Rare value indicator for {col}'},
                        {'name': f'{col}_is_common', 'type': 'frequency', 'description': f'Common value indicator for {col}'}
                    ])
                    
                except Exception as e:
                    logger.warning(f"Error creating frequency features for {col}: {str(e)}")
                    continue
        
        return features, feature_info
    
    def _clean_enhanced_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the enhanced dataframe by handling infinite and extreme values"""
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # For numeric columns, handle NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    # Use median for imputation, fallback to 0
                    median_val = df[col].median()
                    if pd.isna(median_val):
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(median_val)
            
            # Cap extreme values (beyond 5 standard deviations)
            for col in numeric_cols:
                if df[col].std() > 0:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    lower_bound = mean_val - 5 * std_val
                    upper_bound = mean_val + 5 * std_val
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Handle object columns
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                df[col] = df[col].fillna('missing')
            
            return df
            
        except Exception as e:
            logger.warning(f"Error cleaning enhanced dataframe: {str(e)}")
            return df

def generate_feature_engineering_insights(original_df: pd.DataFrame, enhanced_df: pd.DataFrame, 
                                        feature_info: List[Dict], model: str, client) -> List[Dict]:
    """Generate comprehensive insights about the feature engineering process using Azure OpenAI"""
    try:
        insights = []
        
        if client:
            # Prepare detailed summary for LLM
            feature_types_count = {}
            for feature in feature_info:
                feature_type = feature['type']
                feature_types_count[feature_type] = feature_types_count.get(feature_type, 0) + 1
            
            # Analyze feature quality
            numeric_features = enhanced_df.select_dtypes(include=[np.number]).columns
            correlation_analysis = {}
            if len(numeric_features) > 1:
                corr_matrix = enhanced_df[numeric_features].corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': float(corr_val)
                            })
                correlation_analysis['high_correlations'] = high_corr_pairs[:5]  # Top 5
            
            summary = {
                'original_features': len(original_df.columns),
                'new_features': len(enhanced_df.columns) - len(original_df.columns),
                'total_features': len(enhanced_df.columns),
                'feature_types_count': feature_types_count,
                'data_shape': f"{original_df.shape[0]} rows × {original_df.shape[1]} columns → {enhanced_df.shape[0]} rows × {enhanced_df.shape[1]} columns",
                'feature_density': len(enhanced_df.columns) / len(original_df),
                'missing_values_original': original_df.isnull().sum().sum(),
                'missing_values_enhanced': enhanced_df.isnull().sum().sum(),
                'correlation_analysis': correlation_analysis,
                'sample_new_features': [f['name'] for f in feature_info[:10]]
            }
            
            prompt = f"""
            You are an expert data scientist analyzing automated feature engineering results. 
            
            Feature Engineering Analysis:
            {json.dumps(summary, indent=2)}
            
            Provide 5-7 strategic insights covering:
            1. Feature quality and potential ML impact
            2. Dimensionality and overfitting risks
            3. Feature selection recommendations
            4. Model training best practices
            5. Data preprocessing considerations
            6. Performance optimization suggestions
            7. Next steps in the ML pipeline
            
            Each insight should be actionable and specific to this dataset.
            
            Format as JSON:
            {{
                "insights": [
                    {{
                        "title": "Concise insight title (max 60 characters)",
                        "description": "Detailed actionable recommendation (max 250 characters)",
                        "priority": "high|medium|low",
                        "category": "quality|performance|methodology|risk"
                    }}
                ]
            }}
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist providing feature engineering insights in JSON format. Focus on actionable, specific recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            ai_response = json.loads(response.choices[0].message.content)
            insights = ai_response.get('insights', [])
        
        else:
            # Comprehensive fallback insights
            new_features_count = len(enhanced_df.columns) - len(original_df.columns)
            feature_types = list(set([f['type'] for f in feature_info]))
            feature_density = len(enhanced_df.columns) / len(original_df.columns)
            
            insights = [
                {
                    "title": "Feature Engineering Complete",
                    "description": f"Generated {new_features_count} new features ({feature_density:.1f}x expansion). Dataset now has {len(enhanced_df.columns)} total features for enhanced ML performance.",
                    "priority": "high",
                    "category": "quality"
                },
                {
                    "title": "Diverse Feature Portfolio",
                    "description": f"Created {len(feature_types)} feature types: {', '.join(feature_types[:4])}. This diversity should capture multiple data patterns and relationships.",
                    "priority": "high",
                    "category": "quality"
                },
                {
                    "title": "Feature Selection Critical",
                    "description": f"With {len(enhanced_df.columns)} features, implement feature selection (SelectKBest, LASSO, or tree-based) to prevent overfitting and improve interpretability.",
                    "priority": "high",
                    "category": "methodology"
                },
                {
                    "title": "Cross-Validation Strategy",
                    "description": "Use stratified k-fold CV with the enhanced dataset. Monitor for overfitting by comparing train/validation performance across different feature subsets.",
                    "priority": "medium",
                    "category": "methodology"
                },
                {
                    "title": "Computational Efficiency",
                    "description": f"Large feature space may impact training time. Consider dimensionality reduction (PCA, feature selection) for faster model iteration and deployment.",
                    "priority": "medium",
                    "category": "performance"
                },
                {
                    "title": "Feature Importance Analysis",
                    "description": "After model training, analyze feature importance to identify the most valuable engineered features and guide future feature engineering efforts.",
                    "priority": "medium",
                    "category": "methodology"
                },
                {
                    "title": "Data Leakage Validation",
                    "description": "Review temporal and aggregation features for potential data leakage. Ensure features don't contain future information in time-series scenarios.",
                    "priority": "high",
                    "category": "risk"
                }
            ]
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating feature engineering insights: {str(e)}")
        return [
            {
                "title": "Processing Complete", 
                "description": "Feature engineering completed successfully. Review new features and proceed with model training using proper validation techniques.",
                "priority": "medium",
                "category": "quality"
            }
        ]

def add_feature_engineering_routes(app, data_store, client):
    """
    Register routes for the Automated Feature Engineering feature
    """

    # Initialize feature engineering engine with Azure OpenAI client
    fe_engine = FeatureEngineeringEngine(client)

    @app.route('/automated-feature-engineering')
    def automated_feature_engineering():
        """Main feature engineering page"""
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Automated Feature Engineering route accessed with session_id: {session_id}")

            if not session_id or session_id not in data_store:
                logger.warning(f"No valid session found: {session_id}")
                return redirect(url_for('index'))

            session['session_id'] = session_id
            logger.info(f"Session set for Automated Feature Engineering: {session_id}")

            # Redirect to the HTML frontend
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Automated Feature Engineering</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
            </head>
            <body>
                <script>
                    window.location.href = '/static/automated-feature-engineering.html?session_id={{ session_id }}';
                </script>
            </body>
            </html>
            """, session_id=session_id)

        except Exception as e:
            logger.error(f"Error in automated_feature_engineering route: {str(e)}")
            return redirect(url_for('index'))
    @app.route('/api/feature-engineering/dataset-info', methods=['GET'])
    def api_feature_engineering_dataset_info():
        """Get comprehensive dataset information and analysis"""
        try:
            session_id = request.args.get('session_id') or session.get('session_id')
            logger.info(f"Feature Engineering dataset info requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            df = data_store[session_id]['df']
            filename = data_store[session_id]['filename']
            
            # Perform comprehensive dataset analysis
            analysis = fe_engine.analyze_dataset(df)
            
            return jsonify({
                'filename': filename,
                'rows': len(df),
                'columns': analysis['column_analysis'],
                'size': int(analysis['basic_info']['memory_usage']),
                'session_id': session_id,
                'data_quality': analysis['data_quality'],
                'recommendations': analysis['feature_recommendations'],
                'correlation_analysis': analysis['correlation_analysis'],
                'distribution_analysis': analysis['distribution_analysis']
            })
        
        except Exception as e:
            logger.error(f"Error in api_feature_engineering_dataset_info: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}'}), 500

    @app.route('/api/feature-engineering/generate', methods=['POST'])
    def api_feature_engineering_generate():
        """Generate features using the advanced feature engineering engine"""
        try:
            data = request.json
            session_id = data.get('session_id') or session.get('session_id')
            logger.info(f"Feature engineering generation requested for session: {session_id}")
            
            if not session_id or session_id not in data_store:
                logger.warning(f"No data found for session: {session_id}")
                return jsonify({'error': 'No data found. Please upload a file first.'}), 400
            
            selected_columns = data.get('selected_columns', [])
            feature_types = data.get('feature_types', [])
            model = data.get('model', 'gpt-4o')
            processing_mode = data.get('processing_mode', 'intelligent')
            
            if not selected_columns:
                return jsonify({'error': 'No columns selected for feature engineering'}), 400
            
            if not feature_types:
                return jsonify({'error': 'No feature engineering techniques selected'}), 400
            
            df = data_store[session_id]['df']
            
            # Perform feature engineering
            start_time = time.time()
            enhanced_df, feature_info = fe_engine.generate_features(
                df, selected_columns, feature_types, processing_mode
            )
            processing_time = round(time.time() - start_time, 2)
            
            # Store enhanced dataset
            processing_id = str(uuid.uuid4())
            data_store[f"enhanced_{processing_id}"] = {
                'enhanced_df': enhanced_df,
                'original_df': df,
                'feature_info': feature_info,
                'session_id': session_id,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processing_time': processing_time,
                'selected_columns': selected_columns,
                'feature_types': feature_types
            }
            
            # Generate insights using Azure OpenAI
            insights = generate_feature_engineering_insights(df, enhanced_df, feature_info, model, openai_client)
            
            # Prepare response data
            original_data = {
                'columns': df.columns.tolist(),
                'data': df.head(10).to_dict(orient='records')
            }
            
            enhanced_data = {
                'columns': enhanced_df.columns.tolist(),
                'data': enhanced_df.head(10).to_dict(orient='records')
            }
            
            new_features_count = len(enhanced_df.columns) - len(df.columns)
            
            # Calculate feature statistics
            feature_stats = {
                'total_features': len(enhanced_df.columns),
                'new_features': new_features_count,
                'feature_types': list(set([f['type'] for f in feature_info])),
                'feature_type_counts': {},
                'memory_usage_mb': enhanced_df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            for feature in feature_info:
                feature_type = feature['type']
                feature_stats['feature_type_counts'][feature_type] = feature_stats['feature_type_counts'].get(feature_type, 0) + 1
            
            return jsonify({
                'processing_id': processing_id,
                'original_data': original_data,
                'enhanced_data': enhanced_data,
                'new_features_count': new_features_count,
                'total_features_count': len(enhanced_df.columns),
                'processing_time': processing_time,
                'feature_info': feature_info,
                'feature_stats': feature_stats,
                'insights': insights,
                'success': True
            })
        
        except Exception as e:
            logger.error(f"Error in api_feature_engineering_generate: {str(e)}")
            return jsonify({'error': f'An error occurred: {str(e)}', 'success': False}), 500

    @app.route('/api/feature-engineering/download', methods=['POST'])
    def api_feature_engineering_download():
        """Download the enhanced dataset"""
        try:
            data = request.json
            session_id = data.get('session_id')
            processing_id = data.get('processing_id')
            
            if not session_id or not processing_id:
                return jsonify({'error': 'Missing session_id or processing_id'}), 400
            
            enhanced_key = f"enhanced_{processing_id}"
            if enhanced_key not in data_store:
                return jsonify({'error': 'Enhanced dataset not found'}), 404
            
            enhanced_data = data_store[enhanced_key]
            enhanced_df = enhanced_data['enhanced_df']
            
            # Create temporary file
            temp_filename = f"enhanced_dataset_{processing_id}_{int(time.time())}.csv"
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, temp_filename)
            
            # Save to CSV with metadata
            with open(temp_path, 'w', newline='', encoding='utf-8') as f:
                # Write metadata as comments
                f.write(f"# Enhanced Dataset - Generated on {enhanced_data['timestamp']}\n")
                f.write(f"# Processing Time: {enhanced_data.get('processing_time', 'N/A')} seconds\n")
                f.write(f"# Original Columns: {len(enhanced_data['original_df'].columns)}\n")
                f.write(f"# Enhanced Columns: {len(enhanced_df.columns)}\n")
                f.write(f"# New Features: {len(enhanced_df.columns) - len(enhanced_data['original_df'].columns)}\n")
                f.write(f"# Feature Types: {', '.join(enhanced_data.get('feature_types', []))}\n")
                f.write("#\n")
                
                # Write the actual data
                enhanced_df.to_csv(f, index=False)
            
            return send_file(temp_path, as_attachment=True, download_name=temp_filename)
        
        except Exception as e:
            logger.error(f"Error in api_feature_engineering_download: {str(e)}")
            return jsonify({'error': f'Download failed: {str(e)}'}), 500

    @app.route('/api/feature-engineering/feature-importance', methods=['POST'])
    def api_feature_importance():
        """Calculate feature importance using various methods"""
        try:
            data = request.json
            processing_id = data.get('processing_id')
            target_column = data.get('target_column')
            
            if not processing_id:
                return jsonify({'error': 'Missing processing_id'}), 400
            
            enhanced_key = f"enhanced_{processing_id}"
            if enhanced_key not in data_store:
                return jsonify({'error': 'Enhanced dataset not found'}), 404
            
            enhanced_df = data_store[enhanced_key]['enhanced_df']
            
            if target_column and target_column in enhanced_df.columns:
                # Calculate feature importance with target
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
                
                X = enhanced_df.drop(columns=[target_column]).select_dtypes(include=[np.number])
                y = enhanced_df[target_column]
                
                # Determine if classification or regression
                is_classification = y.dtype == 'object' or y.nunique() < 20
                
                if is_classification:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    mi_scores = mutual_info_classif(X, y)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    mi_scores = mutual_info_regression(X, y)
                
                model.fit(X, y)
                
                feature_importance = {
                    'random_forest': dict(zip(X.columns, model.feature_importances_)),
                    'mutual_info': dict(zip(X.columns, mi_scores))
                }
            else:
                # Calculate feature importance without target (variance-based)
                numeric_df = enhanced_df.select_dtypes(include=[np.number])
                variances = numeric_df.var()
                
                feature_importance = {
                    'variance': variances.to_dict()
                }
            
            return jsonify({
                'feature_importance': feature_importance,
                'success': True
            })
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return jsonify({'error': f'Feature importance calculation failed: {str(e)}'}), 500

    @app.route('/api/feature-engineering/visualizations', methods=['POST'])
    def api_feature_visualizations():
        """Generate feature visualization data"""
        try:
            data = request.json
            processing_id = data.get('processing_id')
            
            if not processing_id:
                return jsonify({'error': 'Missing processing_id'}), 400
            
            enhanced_key = f"enhanced_{processing_id}"
            if enhanced_key not in data_store:
                return jsonify({'error': 'Enhanced dataset not found'}), 404
            
            enhanced_df = data_store[enhanced_key]['enhanced_df']
            original_df = data_store[enhanced_key]['original_df']
            
            # Generate correlation heatmap data
            numeric_df = enhanced_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                correlation_matrix = numeric_df.corr()
                
                # Create correlation heatmap
                fig = px.imshow(
                    correlation_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                correlation_plot = json.dumps(fig, cls=PlotlyJSONEncoder)
            else:
                correlation_plot = None
            
            # Feature distribution comparison
            distribution_plots = {}
            for col in original_df.select_dtypes(include=[np.number]).columns[:5]:  # Limit to 5 columns
                if col in enhanced_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=original_df[col], name=f'Original {col}', opacity=0.7))
                    fig.add_trace(go.Histogram(x=enhanced_df[col], name=f'Enhanced {col}', opacity=0.7))
                    fig.update_layout(title=f'Distribution Comparison: {col}', barmode='overlay')
                    distribution_plots[col] = json.dumps(fig, cls=PlotlyJSONEncoder)
            
            return jsonify({
                'correlation_plot': correlation_plot,
                'distribution_plots': distribution_plots,
                'success': True
            })
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            return jsonify({'error': f'Visualization generation failed: {str(e)}'}), 500
