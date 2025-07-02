import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

class ADAttacks:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def pgd_attack(self, images, labels, eps=0.3, alpha=2/255, iters=40):
        # pgd is like fgsm but iterative - much stronger
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        
        # start with random noise
        adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        for i in range(iters):
            adv_images.requires_grad_()
            outputs = self.model(adv_images)
            
            # maximize loss = minimize negative loss
            cost = F.cross_entropy(outputs, labels)
            grad = torch.autograd.grad(cost, adv_images, 
                                     retain_graph=False, create_graph=False)[0]
            
            # take step in gradient direction
            adv_images = adv_images.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
            
        return adv_images
    
    def cw_attack(self, images, labels, c=1e-4, kappa=0, max_iter=1000, lr=0.01):
        # carlini wagner - optimization based, very strong
        batch_size = images.shape[0]
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # use tanh space for box constraints
        w = torch.zeros_like(images, requires_grad=True)
        optimizer = optim.Adam([w], lr=lr)
        
        best_adv = images.clone()
        best_l2 = float('inf') * torch.ones(batch_size).to(self.device)
        
        for step in range(max_iter):
            # convert from tanh space
            adv = (torch.tanh(w) + 1) / 2
            
            # l2 distance
            l2_dist = torch.sum((adv - images) ** 2, dim=(1, 2, 3))
            
            outputs = self.model(adv)
            
            # cw loss function
            real = torch.sum(outputs * F.one_hot(labels, outputs.shape[1]), dim=1)
            other = torch.max((1 - F.one_hot(labels, outputs.shape[1])) * outputs - 
                            F.one_hot(labels, outputs.shape[1]) * 10000, dim=1)[0]
            
            # want real < other (misclassification)
            f_loss = torch.clamp(real - other + kappa, min=0)
            
            # total loss
            cost = l2_dist + c * f_loss
            
            optimizer.zero_grad()
            cost.sum().backward()
            optimizer.step()
            
            # update best adversarial examples
            mask = (f_loss == 0) & (l2_dist < best_l2)
            best_l2[mask] = l2_dist[mask]
            best_adv[mask] = adv[mask].detach()
            
        return best_adv
    
    def auto_attack(self, images, labels, eps=8/255, version='standard'):
        # ensemble of strongest attacks - current sota
        batch_size = images.shape[0]
        
        # run multiple attacks and take worst case
        attacks = []
        
        # apgd-ce (adaptive pgd with cross entropy)
        adv1 = self.apgd_ce(images, labels, eps)
        attacks.append(adv1)
        
        # apgd-dlr (adaptive pgd with dlr loss)
        adv2 = self.apgd_dlr(images, labels, eps)
        attacks.append(adv2)
        
        # fab attack
        adv3 = self.fab_attack(images, labels, eps)
        attacks.append(adv3)
        
        # square attack
        adv4 = self.square_attack(images, labels, eps)
        attacks.append(adv4)
        
        # find worst case for each sample
        worst_adv = images.clone()
        for i in range(batch_size):
            best_attack = None
            min_confidence = float('inf')
            
            for attack in attacks:
                with torch.no_grad():
                    pred = self.model(attack[i:i+1])
                    confidence = torch.max(F.softmax(pred, dim=1))
                    
                    if confidence < min_confidence:
                        min_confidence = confidence
                        best_attack = attack[i]
            
            if best_attack is not None:
                worst_adv[i] = best_attack
        
        return worst_adv
    
    def apgd_ce(self, images, labels, eps, steps=100):
        # adaptive pgd with momentum and step size adaptation
        adv = images.clone()
        momentum = torch.zeros_like(images)
        alpha = eps / 4
        
        for i in range(steps):
            adv.requires_grad_()
            logits = self.model(adv)
            cost = F.cross_entropy(logits, labels)
            
            grad = torch.autograd.grad(cost, adv)[0]
            
            # momentum update
            momentum = 0.9 * momentum + grad / torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            
            # adaptive step size
            if i % 20 == 0 and i > 0:
                alpha *= 0.8
            
            adv = adv.detach() + alpha * momentum.sign()
            adv = torch.clamp(adv, images - eps, images + eps)
            adv = torch.clamp(adv, 0, 1)
            
        return adv
    
    def apgd_dlr(self, images, labels, eps, steps=100):
        # dlr loss version - difference of logit ratio
        adv = images.clone()
        alpha = eps / 4
        
        for i in range(steps):
            adv.requires_grad_()
            logits = self.model(adv)
            
            # dlr loss
            y_true = logits[range(labels.shape[0]), labels]
            y_others = logits.clone()
            y_others[range(labels.shape[0]), labels] = -float('inf')
            y_max = torch.max(y_others, dim=1)[0]
            
            dlr_loss = -(y_true - y_max) / (logits.max(dim=1)[0] - logits.kthvalue(2, dim=1)[0] + 1e-12)
            
            grad = torch.autograd.grad(dlr_loss.sum(), adv)[0]
            adv = adv.detach() + alpha * grad.sign()
            adv = torch.clamp(adv, images - eps, images + eps)
            adv = torch.clamp(adv, 0, 1)
            
        return adv
    
    def fab_attack(self, images, labels, eps):
        # fast adaptive boundary attack - decision boundary
        # simplified version
        return self.pgd_attack(images, labels, eps)  # placeholder for now
    
    def square_attack(self, images, labels, eps, max_queries=5000):
        # query efficient black box attack
        # simplified random search version
        adv = images.clone()
        best_loss = float('inf')
        
        for _ in range(max_queries // 10):  # reduced for demo
            # random square perturbation
            h, w = images.shape[-2:]
            s = np.random.randint(h//10, h//3)
            
            mask = torch.zeros_like(images)
            x = np.random.randint(0, h - s)
            y = np.random.randint(0, w - s)
            
            mask[:, :, x:x+s, y:y+s] = 1
            delta = torch.empty_like(images).uniform_(-eps, eps) * mask
            
            candidate = torch.clamp(images + delta, 0, 1)
            
            with torch.no_grad():
                loss = F.cross_entropy(self.model(candidate), labels)
                if loss > best_loss:
                    best_loss = loss
                    adv = candidate
        
        return adv


def load_data(dataset='cifar10', batch_size=128):
    # quick data loading
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def evaluate_attacks(model, test_loader, device='cuda'):
    # test all attacks and compare
    attacker = ADAttacks(model, device)
    
    total_clean = 0
    total_pgd = 0
    total_cw = 0
    total_auto = 0
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # clean accuracy
        with torch.no_grad():
            clean_pred = model(images).argmax(dim=1)
            total_clean += (clean_pred == labels).sum().item()
        
        # pgd attack
        pgd_adv = attacker.pgd_attack(images, labels)
        with torch.no_grad():
            pgd_pred = model(pgd_adv).argmax(dim=1)
            total_pgd += (pgd_pred == labels).sum().item()
        
        # cw attack  
        cw_adv = attacker.cw_attack(images, labels)
        with torch.no_grad():
            cw_pred = model(cw_adv).argmax(dim=1)
            total_cw += (cw_pred == labels).sum().item()
        
        break  # just test one batch for demo
    
    print(f"Clean accuracy: {total_clean/len(labels):.2%}")
    print(f"PGD robustness: {total_pgd/len(labels):.2%}")
    print(f"C&W robustness: {total_cw/len(labels):.2%}")


if __name__ == "__main__":
    # demo usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load some model and data
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # cifar10 classes
    model = model.to(device)
    
    train_loader, test_loader = load_data()
    evaluate_attacks(model, test_loader, device) 