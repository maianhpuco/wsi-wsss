import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VQGANViTClassifier(nn.Module):
    def __init__(self, vqgan_model, codebook_weights, num_classes=4, patch_size=14, vit_model_name="vit_base_patch16_224"):
        super(VQGANViTClassifier, self).__init__()
        self.vqgan = vqgan_model.eval()  # Pre-trained VQ-GAN model
        self.codebook_weights = codebook_weights  # [n_embed, embed_dim]
        self.embed_dim = codebook_weights.shape[1]  # e.g., 256
        self.patch_size = patch_size  # Expected number of patches per dimension (14x14 for ViT)

        # Load ViT model from timm
        self.vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0)  # num_classes=0 to get features
        vit_dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            vit_output = self.vit(vit_dummy_input)
        self.vit_hidden_dim = vit_output.shape[-1]  # e.g., 768 for vit_base_patch16_224

        # Project VQ-GAN embeddings to match ViT's expected patch embedding dimension
        self.vit_patch_dim = self.vit_hidden_dim  # ViT's patch embedding dimension
        self.projection = nn.Linear(self.embed_dim, self.vit_patch_dim)

        # Classification head
        self.classifier = nn.Linear(self.vit_hidden_dim, num_classes)

        # Verify the expected number of patches for ViT
        self.expected_num_patches = (224 // 16) * (224 // 16)  # 14 * 14 = 196 for 224x224 images with patch size 16
        print(f"Expected number of patches for ViT: {self.expected_num_patches}")

    def preprocess_vqgan(self, x):
        return 2. * x - 1.  # Normalize to [-1, 1] as required by VQ-GAN

    def forward(self, images):
        # images: [batch_size, 3, 224, 224]
        batch_size = images.shape[0]
        device = self.codebook_weights.device
        images = images.to(device)

        # Step 1: Encode images with VQ-GAN to get indices
        images = self.preprocess_vqgan(images)  # [batch_size, 3, 224, 224]
        with torch.no_grad():
            _, _, [_, _, indices] = self.vqgan.encode(images)  # [batch_size, H', W'], H'=W'=28
        H, W = indices.shape[-2:]  # e.g., 28x28
        print(f"VQ-GAN indices grid: {H}x{W}")

        # Step 2: Convert indices to embeddings
        embeddings = self.codebook_weights[indices]  # [batch_size, H', W', embed_dim]
        embeddings = embeddings.permute(0, 3, 1, 2)  # [batch_size, embed_dim, H', W']

        # Step 3: Downsample the embeddings to match ViT's expected grid (14x14)
        target_size = self.patch_size  # 14
        embeddings = F.adaptive_avg_pool2d(embeddings, (target_size, target_size))  # [batch_size, embed_dim, 14, 14]
        embeddings = embeddings.permute(0, 2, 3, 1)  # [batch_size, 14, 14, embed_dim]
        embeddings = embeddings.view(batch_size, target_size * target_size, self.embed_dim)  # [batch_size, 196, embed_dim]

        # Step 4: Project embeddings to match ViT's expected patch embedding dimension
        embeddings = self.projection(embeddings)  # [batch_size, num_patches, vit_patch_dim]

        # Step 5: Prepare embeddings for ViT
        # ViT expects [batch_size, num_patches, hidden_dim] + a [CLS] token
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # [batch_size, num_patches + 1, hidden_dim]
        print(f"Embeddings shape after CLS token: {embeddings.shape}")
        print(f"Positional embeddings shape: {self.vit.pos_embed.shape}")
        embeddings += self.vit.pos_embed[:, :embeddings.size(1), :]  # Add positional embeddings

        # Step 6: Pass through ViT
        vit_output = self.vit.blocks(embeddings)  # [batch_size, num_patches + 1, hidden_dim]
        vit_output = self.vit.norm(vit_output)  # [batch_size, num_patches + 1, hidden_dim]
        cls_output = vit_output[:, 0, :]  # Take the [CLS] token: [batch_size, hidden_dim]

        # Step 7: Classification
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        return logits