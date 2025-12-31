"""
Evaluation Script for Multi-Agent System
Analyzes system performance and identifies limitations
"""

import os
import json
from datetime import datetime
from typing import Dict, List


class AgenticSystemEvaluator:
    """Evaluates the multi-agent system performance"""
    
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.evaluation_results = {}
        
    def evaluate_report_quality(self) -> Dict:
        """Evaluate the quality of generated report"""
        
        print("=" * 70)
        print("EVALUATION: Report Quality Assessment")
        print("=" * 70)
        
        if not os.path.exists(self.report_path):
            return {"error": "Report file not found"}
        
        with open(self.report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {}
        
        # 1. Length check
        word_count = len(content.split())
        results['word_count'] = word_count
        results['length_adequate'] = word_count >= 2500
        print(f"\n Word Count: {word_count} {'done' if results['length_adequate'] else 'no (< 2500)'}")
        
        # 2. Structure check
        required_sections = [
            "Executive Summary",
            "Introduction",
            "Methodology",
            "Exploratory Data Analysis",
            "Predictive Modeling",
            "Results",
            "Discussion",
            "Conclusions",
            "Recommendations",
            "Limitations"
        ]
        
        found_sections = []
        missing_sections = []
        
        for section in required_sections:
            if section.lower() in content.lower():
                found_sections.append(section)
            else:
                missing_sections.append(section)
        
        results['sections_found'] = len(found_sections)
        results['sections_total'] = len(required_sections)
        results['structure_complete'] = len(missing_sections) == 0
        
        print(f"\n Section Completeness: {len(found_sections)}/{len(required_sections)}")
        if missing_sections:
            print("   Missing sections:")
            for section in missing_sections:
                print(f"   - {section}")
        
        # 3. Content depth check
        has_statistics = any(term in content.lower() for term in 
                            ['mean', 'median', 'correlation', 'standard deviation'])
        has_model_metrics = any(term in content.lower() for term in 
                               ['rmse', 'mae', 'r²', 'accuracy', 'f1'])
        has_recommendations = content.lower().count('recommend') >= 3
        
        results['has_statistics'] = has_statistics
        results['has_model_metrics'] = has_model_metrics
        results['has_recommendations'] = has_recommendations
        
        print(f"\n Content Depth:")
        print(f"   - Statistical Analysis: {'✓' if has_statistics else '✗'}")
        print(f"   - Model Metrics: {'✓' if has_model_metrics else '✗'}")
        print(f"   - Recommendations: {'✓' if has_recommendations else '✗'}")
        
        # 4. Overall quality score
        quality_score = (
            (1 if results['length_adequate'] else 0) * 25 +
            (len(found_sections) / len(required_sections)) * 35 +
            (1 if has_statistics else 0) * 15 +
            (1 if has_model_metrics else 0) * 15 +
            (1 if has_recommendations else 0) * 10
        )
        
        results['quality_score'] = round(quality_score, 1)
        
        print(f"\n Overall Quality Score: {quality_score:.1f}/100")
        
        if quality_score >= 80:
            print("   Grade: Excellent")
        elif quality_score >= 60:
            print("   Grade: Good")
        elif quality_score >= 40:
            print("   Grade: Acceptable")
        else:
            print("   Grade: Needs Improvement")
        
        return results
    
    def evaluate_plan_reasonableness(self, content: str) -> Dict:
        """Evaluate if the project plan is reasonable"""
        
        print("\n" + "=" * 70)
        print("EVALUATION: Project Plan Reasonableness")
        print("=" * 70)
        
        results = {}
        
        # Check for key planning elements
        has_objectives = 'objective' in content.lower()
        has_questions = 'question' in content.lower()
        has_methodology = 'methodology' in content.lower() or 'approach' in content.lower()
        has_metrics = any(term in content.lower() for term in 
                         ['metric', 'measure', 'kpi', 'success criteria'])
        
        results['has_objectives'] = has_objectives
        results['has_questions'] = has_questions
        results['has_methodology'] = has_methodology
        results['has_metrics'] = has_metrics
        
        print(f"\n Planning Elements:")
        print(f"   - Clear Objectives: {'done' if has_objectives else '✗'}")
        print(f"   - Research Questions: {'done' if has_questions else '✗'}")
        print(f"   - Methodology Defined: {'done' if has_methodology else '✗'}")
        print(f"   - Success Metrics: {'done' if has_metrics else '✗'}")
        
        plan_score = sum([has_objectives, has_questions, has_methodology, has_metrics]) / 4 * 100
        results['plan_score'] = round(plan_score, 1)
        
        print(f"\n Plan Reasonableness Score: {plan_score:.1f}/100")
        
        return results
    
    def evaluate_eda_coherence(self, content: str) -> Dict:
        """Evaluate if EDA is coherent with the dataset"""
        
        print("\n" + "=" * 70)
        print("EVALUATION: EDA Coherence")
        print("=" * 70)
        
        results = {}
        
        # Expected features in the dataset
        expected_features = [
            'hours_coding', 'lines_of_code', 'bugs_found', 'bugs_fixed',
            'ai_usage', 'sleep', 'cognitive', 'stress', 'task_success',
            'coffee', 'commits', 'errors', 'duration'
        ]
        
        features_mentioned = sum(1 for feat in expected_features 
                                if feat.replace('_', ' ') in content.lower())
        
        results['features_mentioned'] = features_mentioned
        results['features_expected'] = len(expected_features)
        
        print(f"\n Dataset Features Mentioned: {features_mentioned}/{len(expected_features)}")
        
        # Check for specific analyses
        has_distributions = 'distribution' in content.lower()
        has_correlations = 'correlation' in content.lower()
        has_outliers = 'outlier' in content.lower()
        has_missing_analysis = 'missing' in content.lower()
        
        results['has_distributions'] = has_distributions
        results['has_correlations'] = has_correlations
        results['has_outliers'] = has_outliers
        results['has_missing_analysis'] = has_missing_analysis
        
        print(f"\n EDA Components:")
        print(f"   - Distribution Analysis: {'✓' if has_distributions else '✗'}")
        print(f"   - Correlation Analysis: {'✓' if has_correlations else '✗'}")
        print(f"   - Outlier Detection: {'✓' if has_outliers else '✗'}")
        print(f"   - Missing Value Check: {'✓' if has_missing_analysis else '✗'}")
        
        # Count insights
        insight_keywords = ['insight', 'finding', 'pattern', 'trend', 'observation']
        insight_count = sum(content.lower().count(word) for word in insight_keywords)
        results['insight_mentions'] = insight_count
        
        print(f"\n Insight Keywords Found: {insight_count}")
        
        coherence_score = (
            (features_mentioned / len(expected_features)) * 40 +
            sum([has_distributions, has_correlations, has_outliers, has_missing_analysis]) / 4 * 40 +
            min(insight_count / 7, 1) * 20
        )
        results['coherence_score'] = round(coherence_score, 1)
        
        print(f"\n📊 EDA Coherence Score: {coherence_score:.1f}/100")
        
        return results
    
    def evaluate_models_plausibility(self, content: str) -> Dict:
        """Evaluate if proposed models are plausible"""
        
        print("\n" + "=" * 70)
        print("EVALUATION: Model Plausibility")
        print("=" * 70)
        
        results = {}
        
        # Check for model types
        model_types = {
            'linear': 'linear regression' in content.lower() or 'logistic' in content.lower(),
            'random_forest': 'random forest' in content.lower(),
            'xgboost': 'xgboost' in content.lower() or 'gradient boosting' in content.lower(),
        }
        
        models_found = sum(model_types.values())
        results['models_found'] = models_found
        results['model_types'] = {k: v for k, v in model_types.items() if v}
        
        print(f"\n🤖 Models Identified: {models_found}/3")
        for model, found in model_types.items():
            print(f"   - {model.replace('_', ' ').title()}: {'✓' if found else '✗'}")
        
        # Check for evaluation metrics
        metrics = {
            'rmse': 'rmse' in content.lower(),
            'mae': 'mae' in content.lower(),
            'r2': 'r²' in content.lower() or 'r-squared' in content.lower() or 'r2' in content.lower(),
            'accuracy': 'accuracy' in content.lower(),
            'f1': 'f1' in content.lower() or 'f-score' in content.lower()
        }
        
        metrics_found = sum(metrics.values())
        results['metrics_found'] = metrics_found
        
        print(f"\n📊 Evaluation Metrics Found: {metrics_found}")
        for metric, found in metrics.items():
            if found:
                print(f"   - {metric.upper()}: ✓")
        
        # Check for model comparison
        has_comparison = 'comparison' in content.lower() or 'compare' in content.lower()
        has_best_model = 'best model' in content.lower() or 'recommended model' in content.lower()
        has_feature_importance = 'feature importance' in content.lower() or 'important features' in content.lower()
        
        results['has_comparison'] = has_comparison
        results['has_best_model'] = has_best_model
        results['has_feature_importance'] = has_feature_importance
        
        print(f"\n✓ Model Analysis Components:")
        print(f"   - Model Comparison: {'✓' if has_comparison else '✗'}")
        print(f"   - Best Model Selected: {'✓' if has_best_model else '✗'}")
        print(f"   - Feature Importance: {'✓' if has_feature_importance else '✗'}")
        
        plausibility_score = (
            (models_found / 3) * 40 +
            min(metrics_found / 3, 1) * 30 +
            sum([has_comparison, has_best_model, has_feature_importance]) / 3 * 30
        )
        results['plausibility_score'] = round(plausibility_score, 1)
        
        print(f"\n📊 Model Plausibility Score: {plausibility_score:.1f}/100")
        
        return results
    
    def identify_failures_and_limitations(self, content: str) -> Dict:
        """Identify where the system failed or had limitations"""
        
        print("\n" + "=" * 70)
        print("EVALUATION: System Limitations & Failures")
        print("=" * 70)
        
        results = {
            'limitations': [],
            'potential_failures': [],
            'improvement_areas': []
        }
        
        # Common failure patterns
        failure_indicators = {
            'Incomplete EDA': not ('correlation' in content.lower() and 'distribution' in content.lower()),
            'Missing model evaluation': not any(m in content.lower() for m in ['rmse', 'mae', 'r²', 'accuracy']),
            'No feature importance': 'feature importance' not in content.lower(),
            'Weak recommendations': content.lower().count('recommend') < 3,
            'No limitations section': 'limitation' not in content.lower()
        }
        
        print("\n🚨 Potential Failures Detected:")
        for failure, is_present in failure_indicators.items():
            if is_present:
                results['potential_failures'].append(failure)
                print(f"   ⚠️  {failure}")
        
        if not results['potential_failures']:
            print("   ✓ No major failures detected")
        
        # Identify limitations
        common_limitations = [
            "Limited dataset size (1,000 records)",
            "Synthetic or sampled data may not represent real-world complexity",
            "LLM-based analysis may lack domain expertise",
            "No causal inference, only correlations",
            "Limited feature engineering by automated agents",
            "Model interpretability challenges",
            "Lack of external validation data"
        ]
        
        print("\n📋 Documented Limitations:")
        limitations_in_report = [lim for lim in common_limitations 
                                 if any(word in content.lower() 
                                       for word in lim.lower().split()[:3])]
        
        if limitations_in_report:
            for lim in limitations_in_report[:5]:
                print(f"   • {lim}")
                results['limitations'].append(lim)
        else:
            print("   ⚠️  No limitations explicitly documented")
            results['potential_failures'].append("Missing limitations discussion")
        
        # Areas for improvement
        print("\n🔧 Recommended Improvements:")
        improvements = [
            "Add more sophisticated feature engineering",
            "Implement ensemble methods",
            "Include cross-validation results",
            "Add visualization outputs",
            "Improve agent collaboration and information passing",
            "Implement error handling and quality checks",
            "Add human-in-the-loop validation points"
        ]
        
        for imp in improvements[:5]:
            print(f"   • {imp}")
            results['improvement_areas'].append(imp)
        
        results['failure_count'] = len(results['potential_failures'])
        results['limitation_count'] = len(results['limitations'])
        
        return results
    
    def run_full_evaluation(self):
        """Run complete evaluation suite"""
        
        print("\n" + "=" * 70)
        print("🔍 MULTI-AGENT SYSTEM EVALUATION REPORT")
        print("=" * 70)
        print(f"Report: {self.report_path}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Load report
        if not os.path.exists(self.report_path):
            print(f"\n❌ Error: Report not found at {self.report_path}")
            return
        
        with open(self.report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Run evaluations
        self.evaluation_results['report_quality'] = self.evaluate_report_quality()
        self.evaluation_results['plan_evaluation'] = self.evaluate_plan_reasonableness(content)
        self.evaluation_results['eda_evaluation'] = self.evaluate_eda_coherence(content)
        self.evaluation_results['model_evaluation'] = self.evaluate_models_plausibility(content)
        self.evaluation_results['limitations'] = self.identify_failures_and_limitations(content)
        
        # Overall summary
        print("\n" + "=" * 70)
        print("📊 OVERALL EVALUATION SUMMARY")
        print("=" * 70)
        
        overall_score = (
            self.evaluation_results['report_quality'].get('quality_score', 0) * 0.25 +
            self.evaluation_results['plan_evaluation'].get('plan_score', 0) * 0.20 +
            self.evaluation_results['eda_evaluation'].get('coherence_score', 0) * 0.25 +
            self.evaluation_results['model_evaluation'].get('plausibility_score', 0) * 0.30
        )
        
        print(f"\n🎯 Overall System Score: {overall_score:.1f}/100")
        print(f"\nBreakdown:")
        print(f"   - Report Quality: {self.evaluation_results['report_quality'].get('quality_score', 0):.1f}/100 (25% weight)")
        print(f"   - Plan Quality: {self.evaluation_results['plan_evaluation'].get('plan_score', 0):.1f}/100 (20% weight)")
        print(f"   - EDA Coherence: {self.evaluation_results['eda_evaluation'].get('coherence_score', 0):.1f}/100 (25% weight)")
        print(f"   - Model Plausibility: {self.evaluation_results['model_evaluation'].get('plausibility_score', 0):.1f}/100 (30% weight)")
        
        print(f"\n⚠️  Failures Detected: {self.evaluation_results['limitations']['failure_count']}")
        print(f"📋 Limitations Identified: {self.evaluation_results['limitations']['limitation_count']}")
        
        # Save evaluation to file
        eval_file = self.report_path.replace('.md', '_evaluation.json')
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"\n💾 Evaluation saved to: {eval_file}")
        
        return self.evaluation_results


def main():
    """Main evaluation function"""
    
    # Evaluate first run
    print("\n" + "=" * 70)
    print("EVALUATING FIRST RUN")
    print("=" * 70)
    
    evaluator1 = AgenticSystemEvaluator("report_final.md")
    results1 = evaluator1.run_full_evaluation()
    
    # If you have a second run, evaluate it
    if os.path.exists("report_final_run2.md"):
        print("\n\n" + "=" * 70)
        print("EVALUATING SECOND RUN")
        print("=" * 70)
        
        evaluator2 = AgenticSystemEvaluator("report_final_run2.md")
        results2 = evaluator2.run_full_evaluation()
        
        # Compare runs
        print("\n\n" + "=" * 70)
        print("📊 COMPARISON BETWEEN RUNS")
        print("=" * 70)
        
        score1 = (results1['report_quality'].get('quality_score', 0) * 0.25 +
                 results1['plan_evaluation'].get('plan_score', 0) * 0.20 +
                 results1['eda_evaluation'].get('coherence_score', 0) * 0.25 +
                 results1['model_evaluation'].get('plausibility_score', 0) * 0.30)
        
        score2 = (results2['report_quality'].get('quality_score', 0) * 0.25 +
                 results2['plan_evaluation'].get('plan_score', 0) * 0.20 +
                 results2['eda_evaluation'].get('coherence_score', 0) * 0.25 +
                 results2['model_evaluation'].get('plausibility_score', 0) * 0.30)
        
        print(f"\nRun 1 Score: {score1:.1f}/100")
        print(f"Run 2 Score: {score2:.1f}/100")
        print(f"Difference: {abs(score1 - score2):.1f} points")
        
        if abs(score1 - score2) < 5:
            print("\n✓ Consistent performance across runs")
        else:
            print("\n⚠️  Significant variance between runs - LLM non-determinism detected")


if __name__ == "__main__":
    main()