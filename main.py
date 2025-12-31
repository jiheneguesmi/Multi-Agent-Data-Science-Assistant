"""
Multi-Agent Data Science Assistant
Main execution script for AI Developer Performance Analysis
"""

import os
from datetime import datetime
# Set up environment before importing CrewAI
os.environ["MISTRAL_API_KEY"] = ""

from crewai import Agent, Task, Crew, Process, LLM
from tools import CSVReaderTool, DataStatsTool

# ============================================================================
# CONFIGURATION
# ============================================================================

# LLM Configuration - Using Mistral with CrewAI 1.7.2+
llm = LLM(
    model="mistral/mistral-large-latest",
    api_key="96w6TToDa2W3vW5uLQItNwYQsdt7s6AE"
)

# Project inputs
BUSINESS_DESCRIPTION = """
Analyze career paths and educational backgrounds to understand patterns in career recommendations.
The goal is to identify key factors influencing career outcomes, understand the relationships between
education level, specialization, skills, certifications, and academic performance (CGPA) on 
recommended career paths, and provide actionable insights for career guidance and skill development.
"""

CSV_PATH = "data/career_dataset_large.xlsx"
OUTPUT_PATH = "report_career.md"

# ============================================================================
# AGENTS DEFINITION
# ============================================================================

# Agent 1: Project Planner
project_planner = Agent(
    role="Strategic Project Coordinator",
    goal="Transform the business problem into a structured data science workflow with clear milestones and deliverables",
    backstory="""You are an experienced project manager with deep understanding 
    of data science methodologies. You excel at breaking down complex analytical 
    problems into manageable tasks, defining success criteria, and creating 
    actionable roadmaps that guide the entire team toward meaningful insights. 
    You have led numerous data science projects and know how to set teams up 
    for success.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 2: Data Analyst
data_analyst = Agent(
    role="Exploratory Data Analysis Specialist",
    goal="Conduct comprehensive data exploration to uncover patterns, relationships, and anomalies in the developer performance dataset",
    backstory="""You are a meticulous data analyst with expertise in statistical 
    analysis and data visualization. You have a keen eye for detecting data quality 
    issues, identifying meaningful patterns, and translating raw numbers into 
    actionable insights. Your thorough approach ensures no important detail is 
    overlooked, and you're known for your ability to find the story hidden in data.""",
    verbose=True,
    allow_delegation=False,
    tools=[CSVReaderTool(), DataStatsTool()],
    llm=llm
)

# Agent 3: Modeling Agent
modeling_agent = Agent(
    role="Machine Learning Engineer",
    goal="Develop, train, and evaluate predictive models to identify factors influencing developer productivity and performance",
    backstory="""You are a skilled ML practitioner with experience across various 
    modeling techniques. You understand the trade-offs between model complexity 
    and interpretability, always starting with baseline models before exploring 
    advanced techniques. Your focus is on delivering robust, well-validated 
    solutions that provide real business value. You believe in the power of 
    simple, interpretable models when possible.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 4: Report Writer
report_writer = Agent(
    role="Technical Documentation Specialist",
    goal="Synthesize all analytical findings into a comprehensive, well-structured technical report",
    backstory="""You are a technical writer with strong data science background 
    who excels at transforming complex analytical work into clear, actionable 
    documentation. You ensure that insights are accessible to both technical 
    and non-technical stakeholders. Your reports follow academic and industry 
    standards, are well-structured, and tell a compelling story that drives 
    decision-making.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ============================================================================
# TASKS DEFINITION
# ============================================================================

# Task 1: Project Planning
task_planning = Task(
    description=f"""
    Analyze the following business problem and create a comprehensive project plan:
    
    BUSINESS DESCRIPTION:
    {BUSINESS_DESCRIPTION}
    
    DATASET: {CSV_PATH}
    Dataset contains 5,000 records with the following features:
    - Education Level: Matric, Intermediate, Bachelor's, Master's
    - Specialization: Science, Commerce, Business, Finance, Computer Science, etc.
    - Skills: Various technical and professional skills
    - Certifications: Professional certifications
    - CGPA/Percentage: Academic performance metric (0-100)
    - Recommended Career: Target career path based on profile
    
    Your task is to create a detailed project plan that includes:
    1. Clear problem statement and objectives
    2. 5-7 key analytical questions to investigate
    3. Proposed analytical methodology
    4. Success criteria and evaluation metrics
    5. Potential risks and mitigation strategies
    
    Think strategically about what insights would be most valuable for optimizing 
    developer productivity and well-being.
    """,
    expected_output="""A comprehensive project plan document including:
    - Executive summary of the project
    - Clearly defined objectives (3-5 objectives)
    - 5-7 specific research questions
    - Detailed methodology breakdown (EDA → Modeling → Evaluation)
    - Success criteria with measurable metrics
    - Risk assessment and mitigation plan
    Format: Structured text with clear sections and bullet points.""",
    agent=project_planner
)

# Task 2: Exploratory Data Analysis
task_eda = Task(
    description=f"""
    Perform a comprehensive exploratory data analysis on the dataset at: {CSV_PATH}
    
    Use your available tools (CSVReaderTool and DataStatsTool) to:
    
    1. Load and inspect the dataset structure
    2. Generate statistical summaries for all features
    3. Analyze distributions of key variables
    4. Examine correlations between features
    5. Identify missing values, outliers, and data quality issues
    6. Investigate relationships between:
       - AI usage and productivity (LOC, commits)
       - Personal factors (sleep, coffee) and performance
       - Cognitive factors (stress, cognitive load) and errors
       - Task success rate and various input factors
    
    Focus on finding actionable insights that can help optimize developer productivity.
    Based on the project plan from the previous task, ensure your analysis addresses 
    the key research questions.
    """,
    expected_output="""A detailed EDA report containing:
    - Dataset overview (dimensions, data types, basic info)
    - Statistical summary tables for all numeric features
    - Distribution analysis with key statistics (mean, median, std, min, max)
    - Correlation matrix and interpretation
    - Missing value analysis
    - Outlier detection results
    - At least 7 key insights discovered, including:
      * Relationship between AI usage and productivity
      * Impact of personal factors on performance
      * Patterns in high vs low performers
      * Data quality observations
    - Recommendations for feature engineering
    Format: Structured report with sections, tables, and clear insights.""",
    agent=data_analyst,
    context=[task_planning]
)

# Task 3: Model Development
task_modeling = Task(
    description="""
    Based on the EDA findings, develop and evaluate predictive models for developer productivity.
    
    Your tasks:
    1. Define the prediction target (recommend: Task_Success_Rate or Lines_of_Code)
    2. Justify target selection based on business value
    3. Identify and engineer features
    4. Implement 3 baseline models:
       - Linear Regression (or Logistic if classification)
       - Random Forest
       - XGBoost (or another gradient boosting method)
    5. Split data (80/20 train/test)
    6. Train models with cross-validation (5-fold)
    7. Evaluate using appropriate metrics:
       - For regression: RMSE, MAE, R²
       - For classification: Accuracy, Precision, Recall, F1
    8. Perform feature importance analysis
    9. Compare models and recommend the best one
    
    Provide clear reasoning for all modeling decisions.
    """,
    expected_output="""A comprehensive modeling report including:
    - Target variable selection and justification
    - Feature engineering approach
    - Model descriptions and hyperparameters used
    - Training methodology (CV strategy, train/test split)
    - Performance metrics for all 3 models:
      * Cross-validation scores
      * Test set performance
      * Comparison table
    - Feature importance rankings (top 10 features)
    - Model comparison analysis
    - Best model recommendation with reasoning
    - Insights about predictive factors
    - Limitations and potential improvements
    Format: Technical report with tables, metrics, and clear recommendations.""",
    agent=modeling_agent,
    context=[task_planning, task_eda]
)

# Task 4: Report Generation
task_report = Task(
    description=f"""
    Compile all findings from the previous tasks into a comprehensive technical report.
    
    Synthesize outputs from:
    1. Project Planning (objectives and methodology)
    2. EDA (data insights and patterns)
    3. Modeling (predictions and feature importance)
    
    Create a professional technical report following this structure:
    
    # AI Developer Performance Analysis: Technical Report
    
    ## Executive Summary
    - Brief overview of project, key findings, and recommendations (250 words)
    
    ## 1. Introduction
    - Background and context
    - Business problem
    - Objectives
    - Research questions
    
    ## 2. Data and Methodology
    - Dataset description
    - Analytical approach
    - Tools and technologies used
    
    ## 3. Exploratory Data Analysis
    - Data characteristics
    - Statistical findings
    - Key patterns and correlations
    - Data quality assessment
    
    ## 4. Predictive Modeling
    - Model selection rationale
    - Feature engineering
    - Model comparison
    - Performance evaluation
    - Feature importance insights
    
    ## 5. Results and Discussion
    - Key findings
    - Answer to research questions
    - Interpretation of results
    - Business implications
    
    ## 6. Conclusions and Recommendations
    - Summary of insights
    - Actionable recommendations for:
      * Developers (optimizing productivity)
      * Managers (team optimization)
      * Tool developers (AI tool improvements)
    - Future work suggestions
    
    ## 7. Limitations
    - Data limitations
    - Methodological constraints
    - Generalizability concerns
    
    ## References and Appendices
    
    Save the final report to: {OUTPUT_PATH}
    """,
    expected_output="""A complete, professionally formatted technical report (Markdown) with:
    - All required sections properly structured
    - Executive summary (250 words)
    - Comprehensive introduction with clear objectives
    - Detailed methodology section
    - EDA results with key insights
    - Model evaluation with metrics and comparisons
    - Thoughtful discussion of findings
    - Actionable recommendations (minimum 5)
    - Proper conclusions
    - Acknowledgment of limitations
    - Professional formatting with headers, tables, and lists
    - Minimum 2500 words
    - Saved to file: {OUTPUT_PATH}
    
    The report should be publication-ready and suitable for stakeholder presentation.""",
    agent=report_writer,
    context=[task_planning, task_eda, task_modeling],
    output_file=OUTPUT_PATH
)

# ============================================================================
# CREW CREATION AND EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print(" AI DEVELOPER PERFORMANCE ANALYSIS - MULTI-AGENT SYSTEM")
    print("=" * 70)
    print(f"\n Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Dataset: {CSV_PATH}")
    print(f" Output: {OUTPUT_PATH}\n")
    
    # Create the crew
    crew = Crew(
        agents=[project_planner, data_analyst, modeling_agent, report_writer],
        tasks=[task_planning, task_eda, task_modeling, task_report],
        process=Process.sequential,  # Tasks executed in order
        verbose=True
    )
    
    # Prepare inputs
    inputs = {
        "topic": BUSINESS_DESCRIPTION,
        "csv_path": CSV_PATH,
        "OUTPUT_PATH": OUTPUT_PATH
    }
    
    print(" Initializing multi-agent crew...")
    print(f"   - {len(crew.agents)} agents configured")
    print(f"   - {len(crew.tasks)} tasks defined")
    print("\n Starting analysis pipeline...\n")
    
    try:
        # Execute the crew
        result = crew.kickoff(inputs=inputs)
        
        print("\n" + "=" * 70)
        print(" ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f" Report saved to: {OUTPUT_PATH}")
        print("\n Final Output Preview:")
        print("-" * 70)
        print(result)
        print("-" * 70)
        
        # Verify report file was created
        if os.path.exists(OUTPUT_PATH):
            file_size = os.path.getsize(OUTPUT_PATH)
            print(f"\n✓ Report file created successfully ({file_size} bytes)")
        else:
            print("\n  Warning: Report file was not created")
            
    except Exception as e:
        print(f"\n Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()