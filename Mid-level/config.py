import torch

# experimental configuration
class Config:
    
    # hardware setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4
    pin_memory = True
    
    # datasets config
    datasets = {
        'cifar10': {
            'num_classes': 10,
            'img_size': 32,
            'channels': 3,
            'batch_size': 128
        },
        'cifar100': {
            'num_classes': 100,
            'img_size': 32,
            'channels': 3,
            'batch_size': 128
        },
        'imagenet': {
            'num_classes': 1000,
            'img_size': 224,
            'channels': 3,
            'batch_size': 64
        }
    }
    
    # model architectures to test
    models = {
        'resnet18': {'type': 'resnet', 'depth': 18},
        'resnet50': {'type': 'resnet', 'depth': 50},
        'vit_small': {'type': 'vit', 'embed_dim': 384, 'depth': 12},
        'efficientnet_b0': {'type': 'efficientnet', 'variant': 'b0'},
        'defense_aware': {'type': 'custom', 'pathways': 3}
    }
    
    # attack configurations
    attacks = {
        'fgsm': {
            'eps': [0.01, 0.03, 0.1, 0.3],
            'targeted': False
        },
        'pgd': {
            'eps': [0.01, 0.03, 0.1],
            'alpha': 0.01,
            'steps': [10, 20, 40],
            'random_start': True
        },
        'cw': {
            'c': [1e-4, 1e-3, 1e-2],
            'kappa': 0,
            'max_iter': 1000,
            'lr': 0.01
        },
        'autoattack': {
            'eps': [0.01, 0.03, 0.1],
            'version': 'standard',
            'individual': True
        }
    }
    
    # defense configurations
    defenses = {
        'vae_defense': {
            'latent_dim': [32, 64, 128],
            'beta': [0.1, 1.0, 10.0],  # kl weight
            'training_epochs': 100
        },
        'uncertainty_dqn': {
            'dropout_rate': [0.3, 0.5],
            'mc_samples': [50, 100],
            'confidence_threshold': [0.8, 0.9]
        },
        'ensemble_defense': {
            'num_models': [3, 5, 7],
            'diversity_weight': 0.1
        },
        'certified_defense': {
            'noise_std': [0.12, 0.25, 0.5],
            'num_samples': [1000, 5000],
            'alpha': 0.001
        }
    }
    
    # training hyperparameters
    training = {
        'epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'momentum': 0.9,
        'scheduler': 'cosine',
        'warmup_epochs': 5
    }
    
    # adversarial training
    adversarial_training = {
        'enabled': True,
        'attack_type': 'pgd',
        'eps': 0.03,
        'alpha': 0.01,
        'steps': 10,
        'mix_ratio': 0.5  # ratio of adversarial examples
    }
    
    # evaluation settings
    evaluation = {
        'sample_sizes': [100, 500, 1000, 5000],
        'num_runs': 5,
        'confidence_level': 0.95,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'save_predictions': True
    }
    
    # certification settings
    certification = {
        'enabled': True,
        'methods': ['randomized_smoothing', 'interval_bound_propagation'],
        'radii': [0.0, 0.25, 0.5, 1.0, 2.0]
    }
    
    # computational budget
    computational = {
        'max_attack_queries': 10000,
        'max_defense_time': 60,  # seconds per sample
        'parallel_attacks': True,
        'memory_limit': '16GB'
    }
    
    # logging and output
    logging = {
        'log_level': 'INFO',
        'save_models': True,
        'save_attacks': True,
        'output_dir': './results',
        'experiment_name': 'advanced_ml_security',
        'wandb_project': 'adversarial_ml'
    }
    
    # reproducibility
    seed = 42
    deterministic = True


# specific configurations for different experiment types
class QuickTestConfig(Config):
    # fast testing config
    evaluation = {
        'sample_sizes': [100],
        'num_runs': 1,
        'confidence_level': 0.95,
        'metrics': ['accuracy'],
        'save_predictions': False
    }
    
    attacks = {
        'fgsm': {'eps': [0.1], 'targeted': False},
        'pgd': {'eps': [0.1], 'alpha': 0.01, 'steps': [10], 'random_start': True}
    }


class ProductionConfig(Config):
    # production ready config
    evaluation = {
        'sample_sizes': [1000, 5000, 10000],
        'num_runs': 10,
        'confidence_level': 0.99,
        'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'save_predictions': True
    }
    
    computational = {
        'max_attack_queries': 50000,
        'max_defense_time': 300,
        'parallel_attacks': True,
        'memory_limit': '64GB'
    }


class ResearchConfig(Config):
    # comprehensive research config
    datasets = {
        'cifar10': {'num_classes': 10, 'img_size': 32, 'channels': 3, 'batch_size': 128},
        'cifar100': {'num_classes': 100, 'img_size': 32, 'channels': 3, 'batch_size': 128},
        'imagenet': {'num_classes': 1000, 'img_size': 224, 'channels': 3, 'batch_size': 32},
        'mnist': {'num_classes': 10, 'img_size': 28, 'channels': 1, 'batch_size': 256},
        'svhn': {'num_classes': 10, 'img_size': 32, 'channels': 3, 'batch_size': 128}
    }
    
    attacks = {
        'fgsm': {'eps': [0.005, 0.01, 0.03, 0.1, 0.3], 'targeted': False},
        'pgd': {'eps': [0.005, 0.01, 0.03, 0.1], 'alpha': 0.01, 'steps': [10, 20, 40, 100], 'random_start': True},
        'cw': {'c': [1e-5, 1e-4, 1e-3, 1e-2], 'kappa': 0, 'max_iter': 2000, 'lr': 0.01},
        'autoattack': {'eps': [0.005, 0.01, 0.03, 0.1], 'version': 'standard', 'individual': True},
        'square': {'eps': [0.01, 0.03, 0.1], 'max_queries': 10000},
        'boundary': {'max_queries': 25000, 'spherical_step': 0.01}
    }


def get_config(config_type='default'):
    # factory function for configs
    if config_type == 'quick':
        return QuickTestConfig()
    elif config_type == 'production':
        return ProductionConfig()
    elif config_type == 'research':
        return ResearchConfig()
    else:
        return Config()


# hyperparameter search spaces
search_spaces = {
    'vae_defense': {
        'latent_dim': [16, 32, 64, 128, 256],
        'beta': [0.01, 0.1, 1.0, 10.0, 100.0],
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'batch_size': [64, 128, 256]
    },
    
    'dqn_defense': {
        'learning_rate': [1e-4, 1e-3, 1e-2],
        'replay_buffer_size': [1000, 5000, 10000],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'target_update_freq': [10, 50, 100]
    },
    
    'adversarial_training': {
        'eps': [0.01, 0.03, 0.1],
        'alpha': [0.001, 0.01, 0.1],
        'steps': [5, 10, 20],
        'mix_ratio': [0.3, 0.5, 0.7]
    }
}


# architecture specific configs
architecture_configs = {
    'vit': {
        'patch_size': [4, 8, 16],
        'embed_dim': [256, 384, 512],
        'depth': [6, 12, 24],
        'num_heads': [6, 8, 12],
        'mlp_ratio': [2.0, 4.0, 8.0]
    },
    
    'resnet': {
        'depth': [18, 34, 50, 101],
        'width_multiplier': [1, 2, 4],
        'dropout': [0.0, 0.1, 0.3],
        'activation': ['relu', 'swish', 'gelu']
    },
    
    'efficientnet': {
        'variant': ['b0', 'b1', 'b2', 'b3'],
        'dropout': [0.2, 0.3, 0.4],
        'stochastic_depth': [0.0, 0.1, 0.2]
    }
}


if __name__ == "__main__":
    # test config loading
    configs = ['default', 'quick', 'production', 'research']
    
    for config_type in configs:
        cfg = get_config(config_type)
        print(f"\n{config_type.upper()} CONFIG:")
        print(f"  Device: {cfg.device}")
        print(f"  Datasets: {list(cfg.datasets.keys())}")
        print(f"  Models: {list(cfg.models.keys())}")
        print(f"  Attacks: {list(cfg.attacks.keys())}")
        print(f"  Sample sizes: {cfg.evaluation['sample_sizes']}")
    
    print("\nConfigurations loaded successfully!") 