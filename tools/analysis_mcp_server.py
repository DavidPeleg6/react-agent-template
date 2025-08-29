"""
Analysis MCP (Model Context Protocol) tools integration for domain-agnostic data analysis.
Provides statistical analysis, pattern recognition, and data insights capabilities.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
import structlog
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

# Import from data directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import PostgreSQLConnectionConfig

# Import PostgreSQL client for data access
from .postgres_mcp_server import PostgreSQLMCPClient, emit_progress_event, ProgressEvent

logger = structlog.get_logger(__name__)


class AnalysisResult(BaseModel):
    """Result from statistical analysis operations."""
    analysis_type: str
    results: Dict[str, Any]
    confidence_level: Optional[float] = None
    statistical_significance: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class AnalysisMCPClient:
    """Client for performing domain-agnostic data analysis operations."""
    
    def __init__(self, postgres_config: PostgreSQLConnectionConfig):
        """
        Initialize Analysis MCP client.
        
        Args:
            postgres_config: PostgreSQL configuration for data access
        """
        self.postgres_client = PostgreSQLMCPClient(postgres_config)
        logger.info("🔧 Initializing AnalysisMCPClient")
        
    async def calculate_descriptive_statistics(self, data: List[float]) -> AnalysisResult:
        """Calculate descriptive statistics for a dataset."""
        try:
            if not data:
                return AnalysisResult(
                    analysis_type="descriptive_statistics",
                    results={"error": "No data provided"}
                )
            
            if stats is None:
                return AnalysisResult(
                    analysis_type="descriptive_statistics",
                    results={"error": "SciPy not available for statistical analysis"}
                )
            
            np_data = np.array(data)
            
            results = {
                "count": len(data),
                "mean": float(np.mean(np_data)),
                "median": float(np.median(np_data)),
                "std_dev": float(np.std(np_data)),
                "variance": float(np.var(np_data)),
                "min": float(np.min(np_data)),
                "max": float(np.max(np_data)),
                "quartiles": {
                    "q1": float(np.percentile(np_data, 25)),
                    "q2": float(np.percentile(np_data, 50)),
                    "q3": float(np.percentile(np_data, 75))
                },
                "skewness": float(stats.skew(np_data)),
                "kurtosis": float(stats.kurtosis(np_data))
            }
            
            return AnalysisResult(
                analysis_type="descriptive_statistics",
                results=results,
                metadata={"timestamp": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate descriptive statistics: {e}")
            return AnalysisResult(
                analysis_type="descriptive_statistics",
                results={"error": str(e)}
            )
    
    async def perform_correlation_analysis(self, data_x: List[float], data_y: List[float]) -> AnalysisResult:
        """Perform correlation analysis between two datasets."""
        try:
            if len(data_x) != len(data_y) or len(data_x) < 2:
                return AnalysisResult(
                    analysis_type="correlation_analysis",
                    results={"error": "Invalid data - datasets must have same length and at least 2 points"}
                )
            
            if stats is None:
                return AnalysisResult(
                    analysis_type="correlation_analysis",
                    results={"error": "SciPy not available for correlation analysis"}
                )
            
            np_x = np.array(data_x)
            np_y = np.array(data_y)
            
            # Pearson correlation
            pearson_corr, pearson_p = stats.pearsonr(np_x, np_y)
            
            # Spearman correlation
            spearman_corr, spearman_p = stats.spearmanr(np_x, np_y)
            
            results = {
                "pearson_correlation": float(pearson_corr),
                "pearson_p_value": float(pearson_p),
                "spearman_correlation": float(spearman_corr),
                "spearman_p_value": float(spearman_p),
                "sample_size": len(data_x),
                "significant_at_5pct": pearson_p < 0.05
            }
            
            return AnalysisResult(
                analysis_type="correlation_analysis",
                results=results,
                confidence_level=0.95,
                statistical_significance=pearson_p < 0.05,
                metadata={"timestamp": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to perform correlation analysis: {e}")
            return AnalysisResult(
                analysis_type="correlation_analysis",
                results={"error": str(e)}
            )
    
    async def detect_outliers(self, data: List[float], method: str = "iqr") -> AnalysisResult:
        """Detect outliers in a dataset using various methods."""
        try:
            if not data or len(data) < 4:
                return AnalysisResult(
                    analysis_type="outlier_detection",
                    results={"error": "Insufficient data for outlier detection"}
                )
            
            if stats is None and method == "zscore":
                return AnalysisResult(
                    analysis_type="outlier_detection",
                    results={"error": "SciPy not available for Z-score outlier detection. Use 'iqr' method instead."}
                )
            
            np_data = np.array(data)
            outliers = []
            
            if method == "iqr":
                q1 = np.percentile(np_data, 25)
                q3 = np.percentile(np_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = [float(x) for x in np_data if x < lower_bound or x > upper_bound]
            
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(np_data))
                outliers = [float(np_data[i]) for i, z in enumerate(z_scores) if z > 3]
            
            results = {
                "method": method,
                "outliers": outliers,
                "outlier_count": len(outliers),
                "outlier_percentage": (len(outliers) / len(data)) * 100,
                "data_points_analyzed": len(data)
            }
            
            if method == "iqr":
                results.update({
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "iqr": float(iqr)
                })
            
            return AnalysisResult(
                analysis_type="outlier_detection",
                results=results,
                metadata={"timestamp": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to detect outliers: {e}")
            return AnalysisResult(
                analysis_type="outlier_detection",
                results={"error": str(e)}
            )
    
    async def analyze_trends(self, data: List[Tuple[float, float]]) -> AnalysisResult:
        """Analyze trends in time series data (time, value pairs)."""
        try:
            if len(data) < 3:
                return AnalysisResult(
                    analysis_type="trend_analysis",
                    results={"error": "Insufficient data points for trend analysis"}
                )
            
            if stats is None:
                return AnalysisResult(
                    analysis_type="trend_analysis",
                    results={"error": "SciPy not available for trend analysis"}
                )
            
            times = np.array([point[0] for point in data])
            values = np.array([point[1] for point in data])
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(times, values)
            
            # Calculate trend direction
            if slope > 0:
                trend_direction = "increasing"
            elif slope < 0:
                trend_direction = "decreasing"
            else:
                trend_direction = "flat"
            
            # Calculate rate of change
            total_change = values[-1] - values[0]
            time_span = times[-1] - times[0] if times[-1] != times[0] else 1
            rate_of_change = total_change / time_span
            
            results = {
                "trend_direction": trend_direction,
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "standard_error": float(std_err),
                "rate_of_change": float(rate_of_change),
                "total_change": float(total_change),
                "time_span": float(time_span),
                "data_points": len(data),
                "significant_trend": p_value < 0.05
            }
            
            return AnalysisResult(
                analysis_type="trend_analysis",
                results=results,
                confidence_level=0.95,
                statistical_significance=p_value < 0.05,
                metadata={"timestamp": time.time()}
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            return AnalysisResult(
                analysis_type="trend_analysis",
                results={"error": str(e)}
            )
    
    async def close(self):
        """Clean up resources."""
        if self.postgres_client:
            await self.postgres_client.close()


# Create analysis tools with embedded PostgreSQL configuration
def create_analysis_tools_with_config(postgres_config: PostgreSQLConnectionConfig):
    """Create analysis tools with embedded PostgreSQL configuration."""
    
    @tool
    async def calculate_summary_statistics(
        query: str,
        column_name: str,
        config: RunnableConfig = None
    ) -> str:
        """
        Calculate descriptive statistics for numerical data from database query.
        Returns mean, median, standard deviation, quartiles, and other summary statistics.
        """
        try:
            await emit_progress_event(config, ProgressEvent(
                timestamp=time.time(),
                event_type="tool_call",
                message=f"🔢 Calculating summary statistics for column: {column_name}"
            ))
            
            client = AnalysisMCPClient(postgres_config)
            
            # Execute query to get data
            query_result = await client.postgres_client.execute_query(query)
            
            if not query_result.success or not query_result.data:
                return f"Error: Failed to retrieve data - {query_result.error}"
            
            # Extract numerical values from specified column
            numerical_data = []
            for row in query_result.data:
                if column_name in row and row[column_name] is not None:
                    try:
                        value = float(row[column_name])
                        numerical_data.append(value)
                    except (ValueError, TypeError):
                        continue
            
            if not numerical_data:
                return f"Error: No numerical data found in column '{column_name}'"
            
            # Calculate statistics
            result = await client.calculate_descriptive_statistics(numerical_data)
            await client.close()
            
            if "error" in result.results:
                return f"Analysis Error: {result.results['error']}"
            
            stats = result.results
            return f"""Summary Statistics for {column_name}:
Count: {stats['count']}
Mean: {stats['mean']:.4f}
Median: {stats['median']:.4f}
Standard Deviation: {stats['std_dev']:.4f}
Variance: {stats['variance']:.4f}
Min: {stats['min']:.4f}
Max: {stats['max']:.4f}
Quartiles: Q1={stats['quartiles']['q1']:.4f}, Q2={stats['quartiles']['q2']:.4f}, Q3={stats['quartiles']['q3']:.4f}
Skewness: {stats['skewness']:.4f}
Kurtosis: {stats['kurtosis']:.4f}"""
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {e}")
            return f"Error calculating summary statistics: {str(e)}"
    
    @tool
    async def analyze_correlation_between_columns(
        query: str,
        column_x: str,
        column_y: str,
        config: RunnableConfig = None
    ) -> str:
        """
        Analyze correlation between two numerical columns from database query.
        Returns Pearson and Spearman correlation coefficients with significance tests.
        """
        try:
            await emit_progress_event(config, ProgressEvent(
                timestamp=time.time(),
                event_type="tool_call",
                message=f"📊 Analyzing correlation between {column_x} and {column_y}"
            ))
            
            client = AnalysisMCPClient(postgres_config)
            
            # Execute query to get data
            query_result = await client.postgres_client.execute_query(query)
            
            if not query_result.success or not query_result.data:
                return f"Error: Failed to retrieve data - {query_result.error}"
            
            # Extract paired numerical values
            data_x, data_y = [], []
            for row in query_result.data:
                if (column_x in row and column_y in row and 
                    row[column_x] is not None and row[column_y] is not None):
                    try:
                        x_val = float(row[column_x])
                        y_val = float(row[column_y])
                        data_x.append(x_val)
                        data_y.append(y_val)
                    except (ValueError, TypeError):
                        continue
            
            if len(data_x) < 2:
                return f"Error: Insufficient paired data points for correlation analysis"
            
            # Perform correlation analysis
            result = await client.perform_correlation_analysis(data_x, data_y)
            await client.close()
            
            if "error" in result.results:
                return f"Analysis Error: {result.results['error']}"
            
            corr = result.results
            significance = "statistically significant" if corr['significant_at_5pct'] else "not statistically significant"
            
            return f"""Correlation Analysis between {column_x} and {column_y}:
Sample Size: {corr['sample_size']}
Pearson Correlation: {corr['pearson_correlation']:.4f} (p-value: {corr['pearson_p_value']:.6f})
Spearman Correlation: {corr['spearman_correlation']:.4f} (p-value: {corr['spearman_p_value']:.6f})
Statistical Significance: {significance} at 5% level
Interpretation: {'Strong' if abs(corr['pearson_correlation']) > 0.7 else 'Moderate' if abs(corr['pearson_correlation']) > 0.3 else 'Weak'} correlation"""
            
        except Exception as e:
            logger.error(f"Error analyzing correlation: {e}")
            return f"Error analyzing correlation: {str(e)}"
    
    @tool
    async def detect_data_outliers(
        query: str,
        column_name: str,
        method: str = "iqr",
        config: RunnableConfig = None
    ) -> str:
        """
        Detect outliers in numerical data using IQR or Z-score methods.
        Returns list of outlier values and statistical boundaries.
        """
        try:
            await emit_progress_event(config, ProgressEvent(
                timestamp=time.time(),
                event_type="tool_call",
                message=f"🔍 Detecting outliers in {column_name} using {method} method"
            ))
            
            client = AnalysisMCPClient(postgres_config)
            
            # Execute query to get data
            query_result = await client.postgres_client.execute_query(query)
            
            if not query_result.success or not query_result.data:
                return f"Error: Failed to retrieve data - {query_result.error}"
            
            # Extract numerical values
            numerical_data = []
            for row in query_result.data:
                if column_name in row and row[column_name] is not None:
                    try:
                        value = float(row[column_name])
                        numerical_data.append(value)
                    except (ValueError, TypeError):
                        continue
            
            if len(numerical_data) < 4:
                return f"Error: Insufficient data points for outlier detection (need at least 4)"
            
            # Detect outliers
            result = await client.detect_outliers(numerical_data, method)
            await client.close()
            
            if "error" in result.results:
                return f"Analysis Error: {result.results['error']}"
            
            outlier_data = result.results
            outlier_list = outlier_data['outliers'][:10]  # Limit to first 10 for readability
            
            response = f"""Outlier Detection for {column_name} (using {method} method):
Data Points Analyzed: {outlier_data['data_points_analyzed']}
Outliers Found: {outlier_data['outlier_count']} ({outlier_data['outlier_percentage']:.2f}% of data)"""
            
            if outlier_data['outlier_count'] > 0:
                response += f"\nFirst {len(outlier_list)} outlier values: {[f'{x:.4f}' for x in outlier_list]}"
                if outlier_data['outlier_count'] > 10:
                    response += f"\n... and {outlier_data['outlier_count'] - 10} more"
            
            if method == "iqr" and "lower_bound" in outlier_data:
                response += f"\nIQR Boundaries: Lower={outlier_data['lower_bound']:.4f}, Upper={outlier_data['upper_bound']:.4f}"
                response += f"\nIQR: {outlier_data['iqr']:.4f}"
            
            return response
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return f"Error detecting outliers: {str(e)}"
    
    @tool
    async def analyze_time_series_trends(
        query: str,
        time_column: str,
        value_column: str,
        config: RunnableConfig = None
    ) -> str:
        """
        Analyze trends in time series data using linear regression.
        Returns trend direction, slope, statistical significance, and rate of change.
        """
        try:
            await emit_progress_event(config, ProgressEvent(
                timestamp=time.time(),
                event_type="tool_call",
                message=f"📈 Analyzing time series trends for {value_column} over {time_column}"
            ))
            
            client = AnalysisMCPClient(postgres_config)
            
            # Execute query to get data
            query_result = await client.postgres_client.execute_query(query)
            
            if not query_result.success or not query_result.data:
                return f"Error: Failed to retrieve data - {query_result.error}"
            
            # Extract time-value pairs and convert to numerical
            time_value_pairs = []
            for row in query_result.data:
                if (time_column in row and value_column in row and 
                    row[time_column] is not None and row[value_column] is not None):
                    try:
                        # Convert time to timestamp if it's a datetime
                        time_val = row[time_column]
                        if isinstance(time_val, datetime):
                            time_val = time_val.timestamp()
                        else:
                            time_val = float(time_val)
                        
                        value_val = float(row[value_column])
                        time_value_pairs.append((time_val, value_val))
                    except (ValueError, TypeError):
                        continue
            
            if len(time_value_pairs) < 3:
                return f"Error: Insufficient data points for trend analysis (need at least 3)"
            
            # Sort by time
            time_value_pairs.sort(key=lambda x: x[0])
            
            # Analyze trends
            result = await client.analyze_trends(time_value_pairs)
            await client.close()
            
            if "error" in result.results:
                return f"Analysis Error: {result.results['error']}"
            
            trend = result.results
            significance = "statistically significant" if trend['significant_trend'] else "not statistically significant"
            
            return f"""Time Series Trend Analysis for {value_column}:
Data Points: {trend['data_points']}
Trend Direction: {trend['trend_direction'].upper()}
Slope: {trend['slope']:.6f}
R-squared: {trend['r_squared']:.4f}
Statistical Significance: {significance} (p-value: {trend['p_value']:.6f})
Rate of Change: {trend['rate_of_change']:.6f} per time unit
Total Change: {trend['total_change']:.4f}
Time Span: {trend['time_span']:.2f} time units
Standard Error: {trend['standard_error']:.6f}"""
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return f"Error analyzing trends: {str(e)}"
    
    return [
        calculate_summary_statistics,
        analyze_correlation_between_columns,
        detect_data_outliers,
        analyze_time_series_trends
    ]


# Export the tools list for compatibility
ANALYSIS_TOOLS = create_analysis_tools_with_config
