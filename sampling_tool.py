#!/usr/bin/env python3
"""
OpenEvidence Evaluation Study - Stratified Sampling Tool

This script creates stratified samples from the MedXperQA dataset for 
evaluating OpenEvidence and Deep Consult performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
import argparse
from pathlib import Path

class MedicalQuestionSampler:
    def __init__(self, dataset_path):
        """Initialize with the medical reasoning dataset."""
        self.df = pd.read_csv(dataset_path)
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare data with additional features for stratification."""
        # Calculate question length
        self.df['question_length'] = self.df['question'].str.len()
        
        # Create complexity categories based on length quartiles
        q25 = self.df['question_length'].quantile(0.25)
        q75 = self.df['question_length'].quantile(0.75)
        
        def categorize_complexity(length):
            if length < q25:
                return 'Short'
            elif length > q75:
                return 'Long'
            else:
                return 'Medium'
        
        self.df['complexity'] = self.df['question_length'].apply(categorize_complexity)
        
        # Create combined stratification key
        self.df['strata'] = (self.df['medical_task'] + '_' + 
                            self.df['body_system'] + '_' + 
                            self.df['complexity'])
        
        print(f"Dataset prepared: {len(self.df)} questions")
        print(f"Unique strata: {self.df['strata'].nunique()}")
    
    def create_stratified_sample(self, sample_size=100, random_state=42):
        """Create stratified sample maintaining proportional representation."""
        
        # Calculate target proportions
        task_props = self.df['medical_task'].value_counts(normalize=True)
        system_props = self.df['body_system'].value_counts(normalize=True)
        complexity_props = self.df['complexity'].value_counts(normalize=True)
        
        print("Target Proportions:")
        print("Medical Tasks:")
        for task, prop in task_props.items():
            print(f"  {task}: {prop:.1%}")
        
        print("\nComplexity Distribution:")
        for comp, prop in complexity_props.items():
            print(f"  {comp}: {prop:.1%}")
        
        # Stratified sampling by medical task (primary stratification)
        sample_dfs = []
        
        for task in self.df['medical_task'].unique():
            task_df = self.df[self.df['medical_task'] == task]
            task_sample_size = max(1, int(sample_size * task_props[task]))
            
            if len(task_df) >= task_sample_size:
                task_sample = task_df.sample(n=task_sample_size, random_state=random_state)
            else:
                task_sample = task_df  # Take all if not enough
            
            sample_dfs.append(task_sample)
        
        # Combine samples
        stratified_sample = pd.concat(sample_dfs, ignore_index=True)
        
        # If we're short, fill from remaining questions
        if len(stratified_sample) < sample_size:
            remaining_ids = set(self.df['id']) - set(stratified_sample['id'])
            remaining_df = self.df[self.df['id'].isin(remaining_ids)]
            additional_needed = sample_size - len(stratified_sample)
            
            if len(remaining_df) >= additional_needed:
                additional_sample = remaining_df.sample(n=additional_needed, random_state=random_state)
                stratified_sample = pd.concat([stratified_sample, additional_sample], ignore_index=True)
        
        return stratified_sample.head(sample_size)
    
    def create_targeted_sample(self, base_sample, additional_size=100, focus_areas=None):
        """Create additional targeted sample for specific analysis."""
        
        if focus_areas is None:
            focus_areas = ['Long', 'Nervous', 'Cardiovascular']
        
        # Exclude questions already in base sample
        remaining_df = self.df[~self.df['id'].isin(base_sample['id'])]
        
        targeted_samples = []
        per_area_size = additional_size // len(focus_areas)
        
        for area in focus_areas:
            # Check if it's complexity, body system, or medical task
            if area in remaining_df['complexity'].values:
                area_df = remaining_df[remaining_df['complexity'] == area]
            elif area in remaining_df['body_system'].values:
                area_df = remaining_df[remaining_df['body_system'] == area]
            elif area in remaining_df['medical_task'].values:
                area_df = remaining_df[remaining_df['medical_task'] == area]
            else:
                continue
            
            if len(area_df) >= per_area_size:
                area_sample = area_df.sample(n=per_area_size, random_state=42)
                targeted_samples.append(area_sample)
        
        if targeted_samples:
            return pd.concat(targeted_samples, ignore_index=True)
        else:
            return remaining_df.sample(n=additional_size, random_state=42)
    
    def analyze_sample_quality(self, sample_df):
        """Analyze how well the sample represents the population."""
        
        print(f"\nSample Quality Analysis (n={len(sample_df)}):")
        print("="*50)
        
        # Compare proportions
        categories = ['medical_task', 'body_system', 'complexity']
        
        for category in categories:
            print(f"\n{category.replace('_', ' ').title()} Distribution:")
            
            pop_props = self.df[category].value_counts(normalize=True).sort_index()
            sample_props = sample_df[category].value_counts(normalize=True).sort_index()
            
            comparison_df = pd.DataFrame({
                'Population': pop_props,
                'Sample': sample_props
            }).fillna(0)
            
            comparison_df['Difference'] = comparison_df['Sample'] - comparison_df['Population']
            
            for idx, row in comparison_df.iterrows():
                print(f"  {idx}: Pop={row['Population']:.1%}, Sample={row['Sample']:.1%}, Diff={row['Difference']:+.1%}")
        
        # Statistical summary
        print(f"\nQuestion Length Statistics:")
        print(f"Population: Mean={self.df['question_length'].mean():.0f}, Std={self.df['question_length'].std():.0f}")
        print(f"Sample: Mean={sample_df['question_length'].mean():.0f}, Std={sample_df['question_length'].std():.0f}")
    
    def export_sample(self, sample_df, filename, include_evaluation_columns=True):
        """Export sample with evaluation tracking columns."""
        
        export_df = sample_df.copy()
        
        if include_evaluation_columns:
            # Add evaluation tracking columns
            export_df['openevidence_answer'] = ''
            export_df['openevidence_reasoning'] = ''
            export_df['openevidence_references'] = ''
            export_df['openevidence_response_time'] = ''
            export_df['deepconsult_answer'] = ''
            export_df['deepconsult_reasoning'] = ''
            export_df['deepconsult_references'] = ''
            export_df['deepconsult_response_time'] = ''
            export_df['evaluator_notes'] = ''
            export_df['evaluation_date'] = ''
            export_df['quality_score'] = ''
        
        export_df.to_csv(filename, index=False)
        print(f"\nSample exported to: {filename}")
        return filename

def main():
    parser = argparse.ArgumentParser(description='Create stratified samples for OpenEvidence evaluation')
    parser.add_argument('--dataset', default='/home/ubuntu/upload/MedXperQA_Reasoning.csv',
                       help='Path to the medical reasoning dataset')
    parser.add_argument('--pilot-size', type=int, default=100,
                       help='Size of pilot sample')
    parser.add_argument('--targeted-size', type=int, default=100,
                       help='Size of additional targeted sample')
    parser.add_argument('--output-dir', default='/home/ubuntu/',
                       help='Output directory for sample files')
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = MedicalQuestionSampler(args.dataset)
    
    # Create pilot sample
    print("Creating Pilot Sample...")
    pilot_sample = sampler.create_stratified_sample(sample_size=args.pilot_size)
    sampler.analyze_sample_quality(pilot_sample)
    
    pilot_file = Path(args.output_dir) / f'pilot_sample_{args.pilot_size}.csv'
    sampler.export_sample(pilot_sample, pilot_file)
    
    # Create targeted sample
    print("\n" + "="*60)
    print("Creating Targeted Sample...")
    targeted_sample = sampler.create_targeted_sample(
        pilot_sample, 
        additional_size=args.targeted_size,
        focus_areas=['Long', 'Nervous', 'Cardiovascular', 'Diagnosis']
    )
    
    targeted_file = Path(args.output_dir) / f'targeted_sample_{args.targeted_size}.csv'
    sampler.export_sample(targeted_sample, targeted_file)
    
    # Create combined sample
    combined_sample = pd.concat([pilot_sample, targeted_sample], ignore_index=True)
    combined_file = Path(args.output_dir) / f'combined_sample_{len(combined_sample)}.csv'
    sampler.export_sample(combined_sample, combined_file)
    
    print(f"\n" + "="*60)
    print("SUMMARY:")
    print(f"Pilot sample: {len(pilot_sample)} questions -> {pilot_file}")
    print(f"Targeted sample: {len(targeted_sample)} questions -> {targeted_file}")
    print(f"Combined sample: {len(combined_sample)} questions -> {combined_file}")
    
    return pilot_file, targeted_file, combined_file

if __name__ == "__main__":
    main()

