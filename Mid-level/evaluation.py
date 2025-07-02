import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import time

class Evaluator:
    # proper experimental evaluation with statistical testing
    def __init__(self, models, attacks, defenses, datasets):
        self.models = models
        self.attacks = attacks
        self.defenses = defenses
        self.datasets = datasets
        self.results = defaultdict(dict)
        
    def evaluate_all(self, num_runs=5, sample_sizes=[100, 500, 1000]):
        # run comprehensive evaluation
        for dataset_name, dataloader in self.datasets.items():
            print(f"\n=== Evaluating on {dataset_name} ===")
            
            for model_name, model in self.models.items():
                print(f"\nModel: {model_name}")
                
                # test different sample sizes
                for sample_size in sample_sizes:
                    print(f"  Sample size: {sample_size}")
                    
                    # multiple runs for statistical significance
                    run_results = []
                    for run in range(num_runs):
                        run_result = self._single_evaluation(
                            model, dataloader, sample_size, run
                        )
                        run_results.append(run_result)
                    
                    # aggregate results
                    aggregated = self._aggregate_runs(run_results)
                    
                    key = f"{dataset_name}_{model_name}_{sample_size}"
                    self.results[key] = aggregated
        
        return self.results
    
    def _single_evaluation(self, model, dataloader, sample_size, run_id):
        # single evaluation run
        device = next(model.parameters()).device
        model.eval()
        
        results = {
            'clean_metrics': {},
            'attack_metrics': {},
            'defense_metrics': {},
            'computational_cost': {}
        }
        
        # get samples
        samples = self._get_samples(dataloader, sample_size)
        images, labels = samples
        images, labels = images.to(device), labels.to(device)
        
        # clean evaluation
        start_time = time.time()
        with torch.no_grad():
            clean_pred = model(images)
        clean_time = time.time() - start_time
        
        results['clean_metrics'] = self._compute_metrics(clean_pred, labels)
        results['computational_cost']['clean_inference'] = clean_time
        
        # attack evaluation
        for attack_name, attack in self.attacks.items():
            start_time = time.time()
            adv_images = attack.pgd_attack(images, labels)  # assuming pgd method
            attack_time = time.time() - start_time
            
            with torch.no_grad():
                adv_pred = model(adv_images)
            
            attack_metrics = self._compute_metrics(adv_pred, labels)
            attack_metrics['attack_success_rate'] = 1.0 - attack_metrics['accuracy']
            
            results['attack_metrics'][attack_name] = attack_metrics
            results['computational_cost'][f'{attack_name}_generation'] = attack_time
        
        # defense evaluation
        for defense_name, defense in self.defenses.items():
            if hasattr(defense, 'detect_and_purify'):
                start_time = time.time()
                defended_pred, detection_pred, uncertainty = defense.detect_and_purify(adv_images)
                defense_time = time.time() - start_time
                
                defense_metrics = self._compute_metrics(defended_pred, labels)
                
                # detection metrics
                adv_labels = torch.ones(adv_images.size(0))  # all adversarial
                detection_metrics = self._compute_detection_metrics(detection_pred, adv_labels)
                defense_metrics.update(detection_metrics)
                
                results['defense_metrics'][defense_name] = defense_metrics
                results['computational_cost'][f'{defense_name}_time'] = defense_time
        
        return results
    
    def _get_samples(self, dataloader, sample_size):
        # sample from dataloader
        all_images, all_labels = [], []
        
        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels)
            
            if len(all_images) * images.size(0) >= sample_size:
                break
        
        all_images = torch.cat(all_images)[:sample_size]
        all_labels = torch.cat(all_labels)[:sample_size]
        
        return all_images, all_labels
    
    def _compute_metrics(self, predictions, labels):
        # comprehensive metric computation
        if predictions.dim() > 1:
            pred_labels = predictions.argmax(dim=1)
            probs = F.softmax(predictions, dim=1)
        else:
            pred_labels = predictions
            probs = None
        
        pred_labels = pred_labels.cpu().numpy()
        labels = labels.cpu().numpy()
        
        # basic metrics
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average='weighted', zero_division = 0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # confidence metrics
        if probs is not None:
            probs = probs.cpu().numpy()
            confidence = np.max(probs, axis=1).mean()
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1).mean()
            
            metrics.update({
                'confidence': confidence,
                'entropy': entropy
            })
        
        return metrics
    
    def _compute_detection_metrics(self, detection_pred, true_labels):
        # detection specific metrics
        if detection_pred.dim() > 1:
            detection_labels = detection_pred.argmax(dim=1).cpu().numpy()
            detection_probs = F.softmax(detection_pred, dim=1)[:, 1].cpu().numpy()
        else:
            detection_labels = detection_pred.cpu().numpy()
            detection_probs = detection_pred.cpu().numpy()
        
        true_labels = true_labels.cpu().numpy()
        
        # detection accuracy
        detection_acc = accuracy_score(true_labels, detection_labels)
        
        # auc if probabilities available
        try:
            auc = roc_auc_score(true_labels, detection_probs)
        except:
            auc = 0.0
        
        return {
            'detection_accuracy': detection_acc,
            'detection_auc': auc
        }
    
    def _aggregate_runs(self, run_results):
        # aggregate multiple runs with confidence intervals
        aggregated = {
            'clean_metrics': {},
            'attack_metrics': {},
            'defense_metrics': {},
            'computational_cost': {}
        }
        
        # aggregate clean metrics
        clean_metrics = [r['clean_metrics'] for r in run_results]
        aggregated['clean_metrics'] = self._compute_stats(clean_metrics)
        
        # aggregate attack metrics
        for attack_name in run_results[0]['attack_metrics'].keys():
            attack_metrics = [r['attack_metrics'][attack_name] for r in run_results]
            aggregated['attack_metrics'][attack_name] = self._compute_stats(attack_metrics)
        
        # aggregate defense metrics
        for defense_name in run_results[0]['defense_metrics'].keys():
            defense_metrics = [r['defense_metrics'][defense_name] for r in run_results]
            aggregated['defense_metrics'][defense_name] = self._compute_stats(defense_metrics)
        
        # aggregate computational costs
        cost_metrics = [r['computational_cost'] for r in run_results]
        aggregated['computational_cost'] = self._compute_stats(cost_metrics)
        
        return aggregated
    
    def _compute_stats(self, metric_list):
        # compute mean, std, confidence intervals
        stats_dict = {}
        
        # get all metric names
        all_metrics = set()
        for metrics in metric_list:
            all_metrics.update(metrics.keys())
        
        for metric_name in all_metrics:
            values = [metrics.get(metric_name, 0.0) for metrics in metric_list]
            values = np.array(values)
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # 95% confidence interval
            ci_low, ci_high = stats.t.interval(
                0.95, len(values)-1, loc=mean_val, scale=stats.sem(values)
            )
            
            stats_dict[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'values': values.tolist()
            }
        
        return stats_dict
    
    def SST(self, baseline_key, comparison_key, metric='accuracy'):
        # test if improvement is statistically significant
        baseline_results = self.results[baseline_key]['clean_metrics'][metric]['values']
        comparison_results = self.results[comparison_key]['clean_metrics'][metric]['values']
        
        # welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(comparison_results, baseline_results, equal_var=False)
        
        # effect size (cohen's d)
        pooled_std = np.sqrt(((len(baseline_results) - 1) * np.var(baseline_results, ddof=1) +
                             (len(comparison_results) - 1) * np.var(comparison_results, ddof=1)) /
                            (len(baseline_results) + len(comparison_results) - 2))
        
        cohens_d = (np.mean(comparison_results) - np.mean(baseline_results)) / pooled_std
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        }
    
    def generate_report(self, save_path='evaluation_report.json'):
        # generate comprehensive report
        report = {
            'summary': self._generate_summary(),
            'detailed_results': dict(self.results),
            'statistical_tests': self._run_statistical_tests(),
            'recommendations': self._generate_recommendations()
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_summary(self):
        # create summary statistics
        summary = {}
        
        for key, results in self.results.items():
            clean_acc = results['clean_metrics']['accuracy']['mean']
            
            # best attack (lowest accuracy)
            best_attack_acc = float('inf')
            worst_attack = None
            for attack_name, attack_results in results['attack_metrics'].items():
                acc = attack_results['accuracy']['mean']
                if acc < best_attack_acc:
                    best_attack_acc = acc
                    worst_attack = attack_name
            
            # best defense (highest accuracy against attacks)
            best_defense_acc = 0.0
            best_defense = None
            for defense_name, defense_results in results['defense_metrics'].items():
                acc = defense_results['accuracy']['mean']
                if acc > best_defense_acc:
                    best_defense_acc = acc
                    best_defense = defense_name
            
            summary[key] = {
                'clean_accuracy': clean_acc,
                'best_attack': worst_attack,
                'best_attack_accuracy': best_attack_acc,
                'best_defense': best_defense,
                'best_defense_accuracy': best_defense_acc,
                'robustness_gap': clean_acc - best_attack_acc
            }
        
        return summary
    
    def _run_statistical_tests(self):
        # run pairwise statistical tests
        tests = {}
        
        # group results by dataset and sample size
        grouped = defaultdict(list)
        for key in self.results.keys():
            dataset, model, sample_size = key.split('_')
            group_key = f"{dataset}_{sample_size}"
            grouped[group_key].append((key, model))
        
        # compare models within each group
        for group_key, model_keys in grouped.items():
            if len(model_keys) < 2:
                continue
            
            tests[group_key] = {}
            for i, (key1, model1) in enumerate(model_keys):
                for j, (key2, model2) in enumerate(model_keys[i+1:], i+1):
                    comparison_key = f"{model1}_vs_{model2}"
                    tests[group_key][comparison_key] = self.SST(key1, key2)
        
        return tests
    
    def _generate_recommendations(self):
        # ai-generated recommendations based on results
        recommendations = []
        
        # analyze robustness gaps
        high_gap_models = []
        for key, summary in self._generate_summary().items():
            if summary['robustness_gap'] > 0.3:  # 30% drop
                high_gap_models.append(key)
        
        if high_gap_models:
            recommendations.append({
                'type': 'robustness',
                'message': f"Models {high_gap_models} show large robustness gaps. Consider adversarial training.",
                'priority': 'high'
            })
        
        # analyze defense effectiveness
        effective_defenses = []
        for key, results in self.results.items():
            for defense_name, defense_results in results['defense_metrics'].items():
                if defense_results['accuracy']['mean'] > 0.8:
                    effective_defenses.append(defense_name)
        
        if effective_defenses:
            recommendations.append({
                'type': 'defense',
                'message': f"Defenses {set(effective_defenses)} show good performance. Consider deployment.",
                'priority': 'medium'
            })
        
        return recommendations
    
    def plot_results(self, save_dir='plots/'):
        # create visualization plots
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # robustness comparison plot
        self._plot_robustness_comparison(save_dir)
        
        # defense effectiveness plot
        self._plot_defense_effectiveness(save_dir)
        
        # computational cost analysis
        self._plot_computational_costs(save_dir)
    
    def _plot_robustness_comparison(self, save_dir):
        # compare model robustness
        fig, ax = plt.subplots(figsize=(12, 8))
        
        models = []
        clean_accs = []
        attack_accs = []
        
        for key, results in self.results.items():
            if '1000' in key:  # use largest sample size
                models.append(key.replace('_1000', ''))
                clean_accs.append(results['clean_metrics']['accuracy']['mean'])
                
                # get worst attack accuracy
                worst_acc = float('inf')
                for attack_results in results['attack_metrics'].values():
                    acc = attack_results['accuracy']['mean']
                    if acc < worst_acc:
                        worst_acc = acc
                attack_accs.append(worst_acc if worst_acc != float('inf') else 0)
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, clean_accs, width, label='Clean Accuracy', alpha=0.8)
        ax.bar(x + width/2, attack_accs, width, label='Adversarial Accuracy', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Robustness Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/robustness_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_defense_effectiveness(self, save_dir):
        # defense effectiveness heatmap
        defense_data = []
        defense_names = []
        model_names = []
        
        for key, results in self.results.items():
            if '1000' in key and results['defense_metrics']:
                model_name = key.replace('_1000', '')
                model_names.append(model_name)
                
                row = []
                for defense_name, defense_results in results['defense_metrics'].items():
                    if defense_name not in defense_names:
                        defense_names.append(defense_name)
                    row.append(defense_results['accuracy']['mean'])
                defense_data.append(row)
        
        if defense_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            defense_matrix = np.array(defense_data)
            sns.heatmap(defense_matrix, annot=True, fmt='.3f',
                       xticklabels=defense_names, yticklabels=model_names,
                       cmap='RdYlBu_r', ax=ax)
            
            ax.set_title('Defense Effectiveness (Accuracy)')
            plt.tight_layout()
            plt.savefig(f'{save_dir}/defense_effectiveness.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_computational_costs(self, save_dir):
        # computational cost comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = []
        clean_times = []
        defense_times = []
        
        for key, results in self.results.items():
            if '1000' in key:
                models.append(key.replace('_1000', ''))
                clean_times.append(results['computational_cost']['clean_inference']['mean'])
                
                # average defense time
                defense_time = 0
                count = 0
                for cost_name, cost_data in results['computational_cost'].items():
                    if 'defense' in cost_name or 'time' in cost_name:
                        defense_time += cost_data['mean']
                        count += 1
                defense_times.append(defense_time / max(count, 1))
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, clean_times, width, label='Clean Inference', alpha=0.8)
        ax.bar(x + width/2, defense_times, width, label='Defense Time', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computational Cost Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/computational_costs.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # demo usage
    print("Comprehensive evaluation framework ready!")
    print("Usage:")
    print("1. Initialize with models, attacks, defenses, datasets")
    print("2. Run evaluate_all() for complete evaluation")
    print("3. Generate report with statistical analysis")
    print("4. Create visualization plots")
    
    # example initialization
    evaluator = Evaluator(
        models={'model1': None},  # replace with actual models
        attacks={'pgd': None},    # replace with actual attacks
        defenses={'vae': None},   # replace with actual defenses
        datasets={'cifar10': None}  # replace with actual datasets
    ) 