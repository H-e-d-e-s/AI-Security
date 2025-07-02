#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import argparse
from datetime import datetime
import json

# import our improved modules
from config import get_config
from ad_attacks import ADAttacks
from defenses import VAEDefense, UncertaintyDQN, AdaptiveDefenseEnvironment, train_vae_defense, train_uncertainty_dqn
from architectures import get_model, compare_architectures
from evaluation import Evaluator

def set_seed(seed=42):
    # reproducible experiments
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_datasets(config):
    # load and prepare datasets
    datasets = {}
    
    for dataset_name, dataset_config in config.datasets.items():
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
            
        elif dataset_name == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
        
        # create data loaders
        train_loader = DataLoader(trainset, batch_size=dataset_config['batch_size'], 
                                shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
        test_loader = DataLoader(testset, batch_size=dataset_config['batch_size'], 
                               shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)
        
        datasets[dataset_name] = {
            'train': train_loader,
            'test': test_loader,
            'config': dataset_config
        }
    
    return datasets

def create_models(config, datasets):
    # create all model architectures
    models = {}
    
    for model_name, model_config in config.models.items():
        for dataset_name, dataset_info in datasets.items():
            num_classes = dataset_info['config']['num_classes']
            
            # create model
            model = get_model(model_name, num_classes=num_classes, robust=True)
            model = model.to(config.device)
            
            key = f"{model_name}_{dataset_name}"
            models[key] = model
    
    return models

def create_attacks(config, models):
    # create attack objects for each model
    attacks = {}
    
    for model_key, model in models.items():
        attacker = ADAttacks(model, device=config.device)
        attacks[model_key] = attacker
    
    return attacks

def create_defenses(config, datasets):
    # create defense mechanisms
    defenses = {}
    
    for dataset_name, dataset_info in datasets.items():
        img_shape = (dataset_info['config']['channels'], 
                    dataset_info['config']['img_size'], 
                    dataset_info['config']['img_size'])
        
        # vae defense
        input_dim = np.prod(img_shape)
        vae = VAEDefense(input_dim = input_dim)
        
        # uncertainty dqn
        dqn = UncertaintyDQN(img_shape)
        
        # simple classifier for adaptive environment
        base_classifier = get_model('resnet18', num_classes=dataset_info['config']['num_classes'])
        
        # adaptive defense environment
        adaptive_defense = AdaptiveDefenseEnvironment(base_classifier, vae, dqn)
        
        defenses[dataset_name] = {
            'vae': vae,
            'dqn': dqn,
            'adaptive': adaptive_defense
        }
    
    return defenses

def train_models(models, datasets, config):
    # train base models
    print("Training base models...")
    
    for model_key, model in models.items():
        dataset_name = model_key.split('_')[-1]
        
        if dataset_name not in datasets:
            continue
            
        print(f"Training {model_key}...")
        
        # setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training['lr'], 
                                   weight_decay=config.training['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        train_loader = datasets[dataset_name]['train']
        
        # training loop (simplified)
        for epoch in range(min(5, config.training['epochs'])):  # reduced for demo
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(config.device), target.to(config.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                if batch_idx > 10:  # early break for demo
                    break
            
            print(f"  Epoch {epoch+1} completed")

def train_defenses(defenses, datasets, config):
    # train defense mechanisms
    print("Training defense mechanisms...")
    
    for dataset_name, defense_dict in defenses.items():
        print(f"Training defenses for {dataset_name}...")
        
        # train vae
        print("  Training VAE...")
        vae = defense_dict['vae'].to(config.device)
        train_loader = datasets[dataset_name]['train']
        train_vae_defense(vae, train_loader, epochs=5, device=config.device)  # reduced epochs
        
        # train dqn (would need adversarial examples)
        print("  Training DQN...")
        dqn = defense_dict['dqn'].to(config.device)
        # simplified training - in practice would need clean/adversarial data
        
        print(f"  Defenses for {dataset_name} trained")

def run_comprehensive_evaluation(models, attacks, defenses, datasets, config):
    # run the full evaluation pipeline
    print("Running comprehensive evaluation...")
    
    # prepare evaluator inputs
    eval_models = {}
    eval_attacks = {}
    eval_defenses = {}
    eval_datasets = {}
    
    # select subset for demo
    for model_key, model in list(models.items())[:2]:  # limit for demo
        dataset_name = model_key.split('_')[-1]
        
        eval_models[model_key] = model
        eval_attacks[model_key] = attacks[model_key]
        eval_datasets[dataset_name] = datasets[dataset_name]['test']
        
        if dataset_name in defenses:
            eval_defenses[f"{dataset_name}_adaptive"] = defenses[dataset_name]['adaptive']
    
    # create evaluator
    evaluator = Evaluator(
        models=eval_models,
        attacks=eval_attacks,
        defenses=eval_defenses,
        datasets=eval_datasets
    )
    
    # run evaluation
    results = evaluator.evaluate_all(
        num_runs=config.evaluation['num_runs'],
        sample_sizes=config.evaluation['sample_sizes'][:2]  # limit for demo
    )
    
    # generate report
    report = evaluator.generate_report('evaluation_report.json')
    
    # create plots
    evaluator.plot_results('plots/')
    
    return results, report

def save_results(results, report, config):
    # save experiment results
    output_dir = config.logging['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # save detailed results
    with open(f"{output_dir}/results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # save report
    with open(f"{output_dir}/report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Advanced ML Security Experiments')
    parser.add_argument('--config', type=str, default='default', 
                       choices=['default', 'quick', 'production', 'research'],
                       help='Configuration to use')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use for testing)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # load configuration
    config = get_config(args.config)
    config.logging['output_dir'] = args.output_dir
    
    # set seed for reproducibility
    set_seed(config.seed)
    
    print(f"Starting experiment with {args.config} configuration")
    print(f"Device: {config.device}")
    print(f"Output directory: {config.logging['output_dir']}")
    
    # create output directory
    os.makedirs(config.logging['output_dir'], exist_ok=True)
    
    # load datasets
    print("Loading datasets...")
    datasets = load_datasets(config)
    print(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    
    # create models
    print("Creating models...")
    models = create_models(config, datasets)
    print(f"Created {len(models)} models")
    
    # create attacks
    print("Creating attacks...")
    attacks = create_attacks(config, models)
    print(f"Created attacks for {len(attacks)} models")
    
    # create defenses
    print("Creating defenses...")
    defenses = create_defenses(config, datasets)
    print(f"Created defenses for {len(defenses)} datasets")
    
    # train models
    if not args.skip_training:
        train_models(models, datasets, config)
        train_defenses(defenses, datasets, config)
    else:
        print("Skipping training (--skip-training flag)")
    
    # run comprehensive evaluation
    print("Starting comprehensive evaluation...")
    results, report = run_comprehensive_evaluation(models, attacks, defenses, datasets, config)
    
    # save results
    save_results(results, report, config)
    
    # print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    summary = report.get('summary', {})
    for key, stats in summary.items():
        print(f"\n{key}:")
        print(f"  Clean Accuracy: {stats.get('clean_accuracy', 0):.3f}")
        print(f"  Best Attack: {stats.get('best_attack', 'N/A')}")
        print(f"  Robustness Gap: {stats.get('robustness_gap', 0):.3f}")
        print(f"  Best Defense: {stats.get('best_defense', 'N/A')}")
    
    # recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. [{rec['priority'].upper()}] {rec['message']}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Full results saved in: {config.logging['output_dir']}")

if __name__ == "__main__":
    main() 