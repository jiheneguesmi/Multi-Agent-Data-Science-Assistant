# Multi-Agent Data Science Assistant

An intelligent multi-agent system using **CrewAI** and **Mistral AI** to automate the complete data science workflow, from exploratory data analysis to predictive modeling and professional technical report generation.

## Table of Contents

- Overview
- Architecture
- Features
- Installation
- Usage
- Tested Datasets
- Results
- Critical Analysis
- Project Structure
- Technologies
- Limitations
- Future Improvements
- License

## Overview

This project implements an agentic multi-agent AI system that automates the full data science pipeline:

Business Problem -> EDA -> Modeling -> Technical Report

### Tested Use Cases

**AI Developer Performance Analysis (1,000 records)**  
- Analysis of developer productivity with AI assistance  
- Correlation between AI usage, cognitive factors, and performance  

**Career Outcomes Analysis (5,000 records)**  
- Prediction of career trajectories based on education  
- Impact of skills and certifications on professional success  

## Architecture

### Specialized Agents

The system is composed of four autonomous agents operating sequentially.

<img width="910" height="1924" alt="multi-agent-architecture" src="https://github.com/user-attachments/assets/c52c5eb7-d389-4f24-811d-f113431e186d">

### Project Planner Agent

**Role:** Strategic Project Coordinator  

**Mission:**  
Transform business problems into structured analytical workflows.

**Output:**  
- Project objectives  
- Research questions  
- Methodology and execution plan  

### Data Analyst Agent

**Role:** Exploratory Data Analysis Specialist  

**Mission:**  
Perform comprehensive statistical analysis and pattern discovery.

**Tools:**  
- CSVReaderTool  
- DataStatsTool  

**Output:**  
- EDA report with more than 7 key insights  

### Modeling Agent

**Role:** Machine Learning Engineer  

**Mission:**  
Develop and evaluate predictive models.

**Models:**  
- Linear Regression  
- Random Forest  
- XGBoost  

**Output:**  
- Model comparison using RMSE, MAE, and R²  

### Report Writer Agent

**Role:** Technical Documentation Specialist  

**Mission:**  
Synthesize all findings into a professional technical report.

**Output:**  
- Markdown report with more than 2500 words  

## Features

### Data Analysis

- Descriptive statistics (mean, median, standard deviation, skewness, kurtosis)
- Correlation matrix analysis
- Outlier detection using the IQR method
- Distribution analysis
- Missing value and duplicate detection
- Categorical feature analysis

### Predictive Modeling

- Automatic feature engineering
- Comparison of three baseline models
- Simulated 5-fold cross-validation
- Multiple evaluation metrics
- Feature importance analysis
- Best model recommendation

### Report Generation

- Academic structure with 10 sections
- Executive summary
- Formatted tables and placeholders for figures
- Actionable recommendations
- Limitations discussion
- References and appendices

### Automatic Evaluation

- Overall quality score (/100)
- EDA consistency verification
- Model plausibility checks
- Failure and limitation detection
- JSON export of evaluation results

# AI Developer Performance Analysis - Multi-Agent System

## Quick Start
```bash
python main.py
```

### Expected Output:
```
AI DEVELOPER PERFORMANCE ANALYSIS - MULTI-AGENT SYSTEM
Start Time: 2025-12-26 14:30:00
Dataset: data/career_dataset_large.xlsx
Output: report_career.md

Initializing multi-agent crew...
✓ 4 agents configured
✓ 4 tasks defined

Starting analysis pipeline...
Estimated duration: 15-30 minutes
```

## Report Evaluation
```bash
python evaluation.py
```

### Output:
```
MULTI-AGENT SYSTEM EVALUATION REPORT

Overall System Score: 95.5/100
  - Report Quality: 100.0/100
  - Plan Quality: 100.0/100
  - EDA Coherence: 93.8/100
  - Model Plausibility: 90.0/100
```

## Customization

### Change Dataset
```python
CSV_PATH = "data/your_dataset.csv"
OUTPUT_PATH = "custom_report.md"

BUSINESS_DESCRIPTION = """
Your business description here
"""

llm = LLM(
    model="mistral/mistral-large-latest",
    api_key="your_api_key"
)
```

## Tested Datasets

### Dataset 1: AI Developer Performance

| Characteristic | Value                                |
|----------------|--------------------------------------|
| Records        | 1,000                                |
| Features       | 13 (numeric)                         |
| Target         | Task Success Rate                    |
| Key Insight    | Cognitive Load vs Performance: -0.94 |

### Dataset 2: Career Outcomes

| Characteristic | Value                                      |
|----------------|--------------------------------------------|
| Records        | 5,000                                      |
| Features       | 12+ (mixed)                                |
| Target         | Recommended Career                         |
| Key Insight    | Python skills increase success rate by 12% |

## Results

| Metric             | Dataset 1 | Dataset 2 | Average |
|--------------------|-----------|-----------|---------|
| Report Quality     | 100.0     | 100.0     | 100.0   |
| Plan Quality       | 100.0     | 100.0     | 100.0   |
| EDA Coherence      | 93.8      | 93.8      | 93.8    |
| Model Plausibility | 90.0      | 90.0      | 90.0    |
| Overall Score      | 95.5      | 95.5      | 95.5    |

## Critical Analysis

### Strengths
- Excellent planning and structure
- Coherent and detailed EDA
- Plausible model selection and evaluation
- High-quality technical documentation

### Identified Limitations
- Models are not actually trained
- Metrics are generated by the language model
- No real code execution
- Visualizations are placeholders
- Identical evaluation scores across datasets

## Technologies

### Multi-Agent Framework
- CrewAI 0.28.8
- LangChain

### Language Models
- Mistral AI
- Ollama (local)

### Data Science Stack
- Pandas
- NumPy
- openpyxl

### Development
- Python 3.9+
- python-dotenv

## Limitations

This system acts as an intelligent consultant rather than a fully autonomous data scientist. It excels at planning, documentation, and reasoning but does not execute real machine learning pipelines.

## Future Improvements

- Add feedback and validation loops
- Introduce visualization and feature engineering agents
- Integrate MLflow for experiment tracking
- Enable real model training and evaluation
- Add Docker support and workflow orchestration

