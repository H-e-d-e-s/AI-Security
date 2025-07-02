import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import deque
import random

class VAEDefense(nn.Module):
    # variational autoencoder for input purification
    def __init__(self, input_dim=3*32*32, hidden_dim=256, latent_dim=64):
        super().__init__()
        
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # latent space
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x.view(x.size(0), -1))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        original_shape = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon.view(original_shape), mu, logvar
    
    def vae_loss(self, recon_x, x, mu, logvar):
        # reconstruction + kl divergence
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss


class UncertaintyDQN(nn.Module):
    # dqn with dropout for uncertainty estimation
    def __init__(self, input_shape, num_actions=2):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # spatial dropout
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        
        # calculate flattened size
        dummy_input = torch.zeros(1, *input_shape)
        conv_output = self.conv_layers(dummy_input)
        flattened_size = conv_output.view(1, -1).size(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)
    
    def predict_with_uncertainty(self, x, num_samples=50):
        # monte carlo dropout for uncertainty
        self.train()  # enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        self.eval()  # back to eval mode
        return mean_pred, uncertainty


class DefenseEnsemble:
    # combine multiple detection methods
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = torch.tensor(self.weights) / sum(self.weights)
    
    def predict(self, x):
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(F.softmax(pred, dim=1))
        
        # weighted average
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def predict_with_disagreement(self, x):
        # measure disagreement between models
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        
        # disagreement as uncertainty measure
        disagreement = predictions.std(dim=0).mean(dim=1)
        
        return mean_pred, disagreement


class AdaptiveDefenseEnvironment:
    # rl environment that adapts to new attacks
    def __init__(self, classifier, vae_defense, uncertainty_dqn):
        self.classifier = classifier
        self.vae_defense = vae_defense
        self.uncertainty_dqn = uncertainty_dqn
        self.adaptation_buffer = deque(maxlen=1000)
        
    def detect_and_purify(self, x):
        # multi-stage defense pipeline
        
        # stage 1: uncertainty-based detection
        detection_pred, uncertainty = self.uncertainty_dqn.predict_with_uncertainty(x)
        
        # stage 2: vae purification if high uncertainty
        high_uncertainty_mask = uncertainty.max(dim=1)[0] > 0.1
        
        purified_x = x.clone()
        if high_uncertainty_mask.any():
            suspicious_samples = x[high_uncertainty_mask]
            purified_samples, _, _ = self.vae_defense(suspicious_samples)
            purified_x[high_uncertainty_mask] = purified_samples
        
        # stage 3: final classification
        final_pred = self.classifier(purified_x)
        
        return final_pred, detection_pred, uncertainty
    
    def adapt_to_new_attack(self, attack_samples, clean_samples):
        # online learning to adapt defenses
        self.adaptation_buffer.extend([(x, 1) for x in attack_samples])  # adversarial=1
        self.adaptation_buffer.extend([(x, 0) for x in clean_samples])   # clean=0
        
        if len(self.adaptation_buffer) > 100:
            self._retrain_detector()
    
    def _retrain_detector(self):
        # quick adaptation of detection model
        optimizer = optim.Adam(self.uncertainty_dqn.parameters(), lr=0.001)
        
        # sample from adaptation buffer
        batch = random.sample(self.adaptation_buffer, min(64, len(self.adaptation_buffer)))
        
        for x, label in batch:
            x = x.unsqueeze(0) if x.dim() == 3 else x
            pred = self.uncertainty_dqn(x)
            loss = F.cross_entropy(pred, torch.tensor([label]))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class CertifiedDefense:
    # randomized smoothing for certified robustness
    def __init__(self, model, noise_std=0.25):
        self.model = model
        self.noise_std = noise_std
    
    def certify_prediction(self, x, num_samples=1000, alpha=0.001):
        # cohen et al. certified defense
        batch_size = x.shape[0]
        num_classes = 10  # assuming cifar10
        
        # add gaussian noise and collect votes
        votes = torch.zeros(batch_size, num_classes)
        
        for _ in range(num_samples):
            noise = torch.randn_like(x) * self.noise_std
            noisy_x = x + noise
            
            with torch.no_grad():
                pred = self.model(noisy_x)
                votes += F.one_hot(pred.argmax(dim=1), num_classes).float()
        
        # find most voted class
        top_class = votes.argmax(dim=1)
        
        # calculate certification radius
        p_top = votes.max(dim=1)[0] / num_samples
        
        # simplified certification (proper version needs more math)
        certified_radius = self.noise_std * torch.sqrt(2 * torch.log(torch.tensor(1/alpha)))
        
        # only certify if confidence is high enough
        is_certified = p_top > 0.5 + torch.sqrt(torch.log(torch.tensor(1/alpha)) / (2 * num_samples))
        
        return top_class, certified_radius, is_certified


def train_vae_defense(vae, dataloader, epochs=50, device='cuda'):
    # train vae on clean data
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            optimizer.zero_grad()
            recon_data, mu, logvar = vae(data)
            loss = vae.vae_loss(recon_data, data, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}')


def train_uncertainty_dqn(dqn, clean_loader, adv_loader, epochs=100, device='cuda'):
    # train dqn to distinguish clean vs adversarial
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    dqn.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # alternate between clean and adversarial batches
        for (clean_data, _), (adv_data, _) in zip(clean_loader, adv_loader):
            # clean samples (label=0)
            clean_data = clean_data.to(device)
            clean_pred = dqn(clean_data)
            clean_labels = torch.zeros(clean_data.size(0), dtype=torch.long).to(device)
            clean_loss = F.cross_entropy(clean_pred, clean_labels)
            
            # adversarial samples (label=1)
            adv_data = adv_data.to(device)
            adv_pred = dqn(adv_data)
            adv_labels = torch.ones(adv_data.size(0), dtype=torch.long).to(device)
            adv_loss = F.cross_entropy(adv_pred, adv_labels)
            
            total_loss = clean_loss + adv_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0:
            print(f'DQN Epoch {epoch}, Loss: {total_loss:.4f}')


def evaluate_defense_system(defense_env, test_loader, attacker, device='cuda'):
    # comprehensive evaluation
    total_samples = 0
    correct_clean = 0
    correct_defended = 0
    detected_correctly = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # generate adversarial examples
        adv_images = attacker.pgd_attack(images, labels)
        
        # test defense system
        defended_pred, detection_pred, uncertainty = defense_env.detect_and_purify(adv_images)
        
        # check accuracy
        correct_defended += (defended_pred.argmax(dim=1) == labels).sum().item()
        
        # check detection (assuming we know these are adversarial)
        detected_correctly += (detection_pred.argmax(dim=1) == 1).sum().item()
        
        total_samples += images.size(0)
        
        if total_samples > 1000:  # limit for demo
            break
    
    print(f"Defense accuracy: {correct_defended/total_samples:.2%}")
    print(f"Detection accuracy: {detected_correctly/total_samples:.2%}")


if __name__ == "__main__":
    # demo the enhanced defense system
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create defense components
    vae_defense = VAEDefense()
    uncertainty_dqn = UncertaintyDQN((3, 32, 32))
    
    # create adaptive defense environment
    base_classifier = nn.Sequential(
        nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64*6*6, 10)
    )
    
    defense_env = AdaptiveDefenseEnvironment(base_classifier, vae_defense, uncertainty_dqn)
    
    print("Enhanced defense system ready!") 