import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class VisionTransformer(nn.Module):
    # vit for adversarial robustness comparison
    def __init__(self, img_size=32, patch_size=4, num_classes=10, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # final norm and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # init weights
        self._init_weights()
    
    def _init_weights(self):
        # proper transformer initialization
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # add positional embedding
        x = x + self.pos_embed
        
        # transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # final norm and classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        
        return self.head(cls_token_final)


class TransformerBlock(nn.Module):
    # standard transformer block
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout)
    
    def forward(self, x):
        # residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttention(nn.Module):
    # multi head self attention
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        B, N, C = x.shape
        
        # generate q, k, v
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    # mlp block for transformer
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class RobustResNet(nn.Module):
    # resnet with robustness modifications
    def __init__(self, num_classes=10, arch='resnet18'):
        super().__init__()
        
        # load pretrained resnet
        if arch == 'resnet18':
            self.backbone = models.resnet18(pretrained=False)
        elif arch == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
        
        # modify first conv for cifar (smaller images)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # remove maxpool for cifar
        
        # modify final layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
        # add spectral normalization for robustness
        self._apply_spectral_norm()
        
        # gradient penalty weight
        self.gp_weight = 0.1
    
    def _apply_spectral_norm(self):
        # spectral normalization on conv layers
        for module in self.backbone.modules():
            if isinstance(module, nn.Conv2d):
                nn.utils.spectral_norm(module)
            elif isinstance(module, nn.Linear):
                nn.utils.spectral_norm(module)
    
    def forward(self, x):
        return self.backbone(x)
    
    def compute_gradient_penalty(self, real_data, fake_data):
        # gradient penalty for stability
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        outputs = self.forward(interpolated)
        gradients = torch.autograd.grad(
            outputs=outputs.sum(),
            inputs=interpolated,
            create_graph=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class EfficientNetRobust(nn.Module):
    # efficientnet with defensive modifications
    def __init__(self, num_classes=10, model_name='efficientnet_b0'):
        super().__init__()
        
        # load efficientnet backbone
        self.backbone = models.efficientnet_b0(pretrained=False)
        
        # modify classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # add batch normalization for stability
        self.input_bn = nn.BatchNorm2d(3)
        
    def forward(self, x):
        # input normalization
        x = self.input_bn(x)
        return self.backbone(x)


class DefenseAwareArchitecture(nn.Module):
    # architecture designed with adversarial robustness in mind
    def __init__(self, num_classes=10):
        super().__init__()
        
        # multiple pathways for robustness
        self.pathway1 = self._create_pathway(3, 64)
        self.pathway2 = self._create_pathway(3, 64)
        self.pathway3 = self._create_pathway(3, 64)
        
        # attention mechanism to combine pathways
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64*3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        # final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # noise layers for implicit regularization
        self.noise_std = 0.1
    
    def _create_pathway(self, in_channels, out_channels):
        # each pathway uses different kernel sizes
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # add training noise for robustness
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # parallel pathways
        feat1 = self.pathway1(x)
        feat2 = self.pathway2(x)
        feat3 = self.pathway3(x)
        
        # combine features
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        
        # attention weights
        attn_weights = self.attention(combined).unsqueeze(-1).unsqueeze(-1)
        attn_weights = attn_weights.repeat(1, 1, feat1.size(2), feat1.size(3))
        
        # weighted combination
        final_feat = (attn_weights[:, 0:1] * feat1 + 
                     attn_weights[:, 1:2] * feat2 + 
                     attn_weights[:, 2:3] * feat3)
        
        return self.classifier(final_feat)


def get_model(arch='resnet18', num_classes=10, robust=True):
    # factory function for different architectures
    if arch == 'vit':
        return VisionTransformer(num_classes=num_classes)
    elif arch == 'resnet18':
        return RobustResNet(num_classes=num_classes, arch='resnet18') if robust else models.resnet18(num_classes=num_classes)
    elif arch == 'resnet50':
        return RobustResNet(num_classes=num_classes, arch='resnet50') if robust else models.resnet50(num_classes=num_classes)
    elif arch == 'efficientnet':
        return EfficientNetRobust(num_classes=num_classes)
    elif arch == 'defense_aware':
        return DefenseAwareArchitecture(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def compare_architectures(architectures, test_loader, attacker, device='cuda'):
    # compare robustness across different architectures
    results = {}
    
    for arch_name in architectures:
        print(f"\nTesting {arch_name}...")
        model = get_model(arch_name).to(device)
        model.eval()
        
        clean_acc = 0
        adv_acc = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # clean accuracy
            with torch.no_grad():
                clean_pred = model(images)
                clean_acc += (clean_pred.argmax(1) == labels).sum().item()
            
            # adversarial accuracy
            if attacker:
                adv_images = attacker.pgd_attack(images, labels)
                with torch.no_grad():
                    adv_pred = model(adv_images)
                    adv_acc += (adv_pred.argmax(1) == labels).sum().item()
            
            total += labels.size(0)
            
            if total > 1000:  # limit for demo
                break
        
        results[arch_name] = {
            'clean_accuracy': clean_acc / total,
            'adversarial_accuracy': adv_acc / total if attacker else 0.0
        }
        
        print(f"Clean: {results[arch_name]['clean_accuracy']:.2%}")
        print(f"Robust: {results[arch_name]['adversarial_accuracy']:.2%}")
    
    return results


if __name__ == "__main__":
    # demo different architectures
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # test architectures
    architectures = ['resnet18', 'vit', 'efficientnet', 'defense_aware']
    
    for arch in architectures:
        model = get_model(arch, num_classes=10)
        print(f"\n{arch} parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # test forward pass
        dummy_input = torch.randn(1, 3, 32, 32)
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
    
    print("\nAll architectures loaded successfully!") 