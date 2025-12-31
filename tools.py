"""
Custom Tools for Multi-Agent Data Science System
Implements CSV reading and statistical analysis tools
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from crewai.tools import BaseTool


class CSVReaderTool(BaseTool):
    """Tool for reading and inspecting CSV files"""
    
    name: str = "CSV Reader Tool"
    description: str = """Reads CSV files and provides basic information about the dataset.
    Returns column names, data types, shape, and a preview of the first few rows.
    Input should be a file path to a CSV file."""
    
    def _run(self, csv_path: str) -> str:
        """
        Read CSV file and return basic information
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            String containing dataset information
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Prepare output
            output = []
            output.append("=" * 70)
            output.append("CSV DATASET INFORMATION")
            output.append("=" * 70)
            
            # Basic info
            output.append(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            output.append(f"\n📁 File: {csv_path}")
            
            # Column information
            output.append("\n\n📋 COLUMNS AND DATA TYPES:")
            output.append("-" * 70)
            for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
                output.append(f"{i:2d}. {col:30s} | {str(dtype):15s}")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            output.append(f"\n💾 Memory Usage: {memory_mb:.2f} MB")
            
            # Missing values summary
            missing = df.isnull().sum()
            if missing.sum() > 0:
                output.append("\n\n⚠️  MISSING VALUES DETECTED:")
                output.append("-" * 70)
                for col, count in missing[missing > 0].items():
                    pct = (count / len(df)) * 100
                    output.append(f"{col:30s} | {count:5d} ({pct:5.2f}%)")
            else:
                output.append("\n\n✓ No missing values detected")
            
            # Preview
            output.append("\n\n👀 FIRST 5 ROWS PREVIEW:")
            output.append("-" * 70)
            output.append(df.head().to_string())
            
            # Last rows preview
            output.append("\n\n👀 LAST 3 ROWS PREVIEW:")
            output.append("-" * 70)
            output.append(df.tail(3).to_string())
            
            output.append("\n" + "=" * 70)
            
            return "\n".join(output)
            
        except FileNotFoundError:
            return f"❌ Error: File not found at path: {csv_path}"
        except pd.errors.EmptyDataError:
            return f"❌ Error: The CSV file is empty: {csv_path}"
        except Exception as e:
            return f"❌ Error reading CSV: {str(e)}"


class DataStatsTool(BaseTool):
    """Tool for computing statistical summaries of datasets"""
    
    name: str = "Data Statistics Tool"
    description: str = """Computes comprehensive statistical analysis of a CSV dataset.
    Provides descriptive statistics, distribution analysis, correlation matrix, 
    and identifies outliers. Input should be a file path to a CSV file."""
    
    def _run(self, csv_path: str) -> str:
        """
        Compute statistical analysis of the dataset
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            String containing statistical analysis
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            output = []
            output.append("=" * 70)
            output.append("STATISTICAL ANALYSIS REPORT")
            output.append("=" * 70)
            
            # Separate numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            
            output.append(f"\n📊 Numeric Columns: {len(numeric_cols)}")
            output.append(f"📊 Categorical Columns: {len(categorical_cols)}")
            
            # ================================================================
            # DESCRIPTIVE STATISTICS FOR NUMERIC COLUMNS
            # ================================================================
            if numeric_cols:
                output.append("\n\n" + "=" * 70)
                output.append("DESCRIPTIVE STATISTICS (Numeric Features)")
                output.append("=" * 70)
                
                desc = df[numeric_cols].describe().round(2)
                output.append("\n" + desc.to_string())
                
                # Additional statistics
                output.append("\n\n📈 ADDITIONAL STATISTICS:")
                output.append("-" * 70)
                
                for col in numeric_cols:
                    output.append(f"\n{col}:")
                    output.append(f"  • Median: {df[col].median():.2f}")
                    output.append(f"  • Mode: {df[col].mode().values[0]:.2f}")
                    output.append(f"  • Variance: {df[col].var():.2f}")
                    output.append(f"  • Skewness: {df[col].skew():.2f}")
                    output.append(f"  • Kurtosis: {df[col].kurtosis():.2f}")
                    output.append(f"  • Range: [{df[col].min():.2f}, {df[col].max():.2f}]")
            
            # ================================================================
            # CORRELATION ANALYSIS
            # ================================================================
            if len(numeric_cols) > 1:
                output.append("\n\n" + "=" * 70)
                output.append("CORRELATION MATRIX")
                output.append("=" * 70)
                
                corr_matrix = df[numeric_cols].corr().round(3)
                output.append("\n" + corr_matrix.to_string())
                
                # Find strong correlations
                output.append("\n\n🔗 STRONG CORRELATIONS (|r| > 0.5):")
                output.append("-" * 70)
                
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            col1 = corr_matrix.columns[i]
                            col2 = corr_matrix.columns[j]
                            strong_corr.append((col1, col2, corr_val))
                
                if strong_corr:
                    strong_corr.sort(key=lambda x: abs(x[2]), reverse=True)
                    for col1, col2, corr_val in strong_corr:
                        direction = "positive" if corr_val > 0 else "negative"
                        output.append(f"  • {col1} ↔ {col2}: {corr_val:.3f} ({direction})")
                else:
                    output.append("  No strong correlations found")
            
            # ================================================================
            # OUTLIER DETECTION (IQR Method)
            # ================================================================
            if numeric_cols:
                output.append("\n\n" + "=" * 70)
                output.append("OUTLIER ANALYSIS (IQR Method)")
                output.append("=" * 70)
                
                outlier_summary = []
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    n_outliers = len(outliers)
                    pct_outliers = (n_outliers / len(df)) * 100
                    
                    if n_outliers > 0:
                        outlier_summary.append({
                            'column': col,
                            'n_outliers': n_outliers,
                            'pct': pct_outliers,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        })
                
                if outlier_summary:
                    output.append("\n⚠️  Outliers detected in the following columns:")
                    output.append("-" * 70)
                    for item in outlier_summary:
                        output.append(f"\n{item['column']}:")
                        output.append(f"  • Count: {item['n_outliers']} ({item['pct']:.2f}%)")
                        output.append(f"  • Bounds: [{item['lower_bound']:.2f}, {item['upper_bound']:.2f}]")
                else:
                    output.append("\n✓ No significant outliers detected")
            
            # ================================================================
            # DISTRIBUTION INSIGHTS
            # ================================================================
            output.append("\n\n" + "=" * 70)
            output.append("DISTRIBUTION INSIGHTS")
            output.append("=" * 70)
            
            for col in numeric_cols:
                output.append(f"\n{col}:")
                
                # Skewness interpretation
                skew = df[col].skew()
                if abs(skew) < 0.5:
                    skew_interp = "approximately symmetric"
                elif skew > 0:
                    skew_interp = "right-skewed (positive skew)"
                else:
                    skew_interp = "left-skewed (negative skew)"
                output.append(f"  • Distribution: {skew_interp}")
                
                # Kurtosis interpretation
                kurt = df[col].kurtosis()
                if abs(kurt) < 0.5:
                    kurt_interp = "normal-like tails"
                elif kurt > 0:
                    kurt_interp = "heavy tails (more outliers)"
                else:
                    kurt_interp = "light tails (fewer outliers)"
                output.append(f"  • Tail behavior: {kurt_interp}")
            
            # ================================================================
            # CATEGORICAL COLUMNS ANALYSIS
            # ================================================================
            if categorical_cols:
                output.append("\n\n" + "=" * 70)
                output.append("CATEGORICAL FEATURES ANALYSIS")
                output.append("=" * 70)
                
                for col in categorical_cols:
                    output.append(f"\n{col}:")
                    output.append(f"  • Unique values: {df[col].nunique()}")
                    output.append(f"  • Most frequent: {df[col].mode().values[0]} (appears {df[col].value_counts().iloc[0]} times)")
                    
                    if df[col].nunique() <= 10:
                        output.append("  • Value counts:")
                        for val, count in df[col].value_counts().head(10).items():
                            pct = (count / len(df)) * 100
                            output.append(f"    - {val}: {count} ({pct:.1f}%)")
            
            # ================================================================
            # DATA QUALITY SUMMARY
            # ================================================================
            output.append("\n\n" + "=" * 70)
            output.append("DATA QUALITY SUMMARY")
            output.append("=" * 70)
            
            missing_total = df.isnull().sum().sum()
            duplicates = df.duplicated().sum()
            
            output.append(f"\n✓ Total records: {len(df)}")
            output.append(f"✓ Total features: {len(df.columns)}")
            output.append(f"✓ Missing values: {missing_total} ({(missing_total/(len(df)*len(df.columns))*100):.2f}%)")
            output.append(f"✓ Duplicate rows: {duplicates} ({(duplicates/len(df)*100):.2f}%)")
            
            if missing_total == 0 and duplicates == 0:
                output.append("\n🎉 Excellent data quality! No missing values or duplicates.")
            
            output.append("\n" + "=" * 70)
            
            return "\n".join(output)
            
        except FileNotFoundError:
            return f"❌ Error: File not found at path: {csv_path}"
        except Exception as e:
            return f"❌ Error computing statistics: {str(e)}"


# Export tools
__all__ = ['CSVReaderTool', 'DataStatsTool']