import torch
import torch.nn as nn
import timm

class VQGANProcessor(nn.Module):
    def __init__(self, vqgan_model):
        super(VQGANProcessor, self).__init__()
        self.vqgan = vqgan_model.eval()  # Pre-trained VQ-GAN model

    def preprocess_vqgan(self, x):
        return 2. * x - 1.  # Normalize to [-1, 1] as required by VQ-GAN

    def forward(self, images):
        # images: [batch_size, 3, 224, 224]
        device = next(self.vqgan.parameters()).device
        images = images.to(device)
        images = self.preprocess_vqgan(images)  # [batch_size, 3, 224, 224]

        with torch.no_grad():
            _, _, [_, _, indices] = self.vqgan.encode(images)  # [batch_size, H', W'], H'=W'=28
        return indices  # [batch_size, 28, 28]

class ViTClassifier(nn.Module):
    def __init__(self, codebook_weights, num_classes=4, vit_model_name="vit_base_patch16_224"):
        super(ViTClassifier, self).__init__()
        self.codebook_weights = codebook_weights  # [n_embed, embed_dim]
        self.embed_dim = codebook_weights.shape[1]  # e.g., 256

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

        # Adjust positional embeddings to match VQ-GAN output (28x28 = 784 patches)
        self.num_patches = 28 * 28  # VQ-GAN output grid is 28x28
        expected_positions = self.num_patches + 1  # +1 for [CLS] token
        print(f"Expected number of positions (including CLS): {expected_positions}")

        # Resize ViT's positional embeddings
        original_pos_embed = self.vit.pos_embed  # [1, 197, 768]
        print(f"Original positional embeddings shape: {original_pos_embed.shape}")
        new_pos_embed = nn.Parameter(
            torch.nn.functional.interpolate(
                original_pos_embed.permute(0, 2, 1),  # [1, 768, 197]
                size=expected_positions,  # Resize to 785 positions
                mode='linear'
            ).permute(0, 2, 1)  # [1, 785, 768]
        )
        self.vit.pos_embed = new_pos_embed
        print(f"New positional embeddings shape: {self.vit.pos_embed.shape}")

    def forward(self, indices):
        # indices: [batch_size, H', W'], H'=W'=28
        batch_size = indices.shape[0]
        device = self.codebook_weights.device
        indices = indices.to(device)

        H, W = indices.shape[-2:]  # e.g., 28x28
        print(f"Input indices grid: {H}x{W}")
        assert H * W == self.num_patches, f"Expected {self.num_patches} patches, got {H * W}"

        # Step 1: Convert indices to embeddings
        embeddings = self.codebook_weights[indices]  # [batch_size, H', W', embed_dim]
        embeddings = embeddings.view(batch_size, H * W, self.embed_dim)  # [batch_size, num_patches, embed_dim]

        # Step 2: Project embeddings to match ViT's expected patch embedding dimension
        embeddings = self.projection(embeddings)  # [batch_size, num_patches, vit_patch_dim]

        # Step 3: Prepare embeddings for ViT
        # ViT expects [batch_size, num_patches, hidden_dim] + a [CLS] token
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # [batch_size, num_patches + 1, hidden_dim]
        print(f"Embeddings shape after CLS token: {embeddings.shape}")
        embeddings += self.vit.pos_embed[:, :embeddings.size(1), :]  # Add positional embeddings

        # Step 4: Pass through ViT
        vit_output = self.vit.blocks(embeddings)  # [batch_size, num_patches + 1, hidden_dim]
        vit_output = self.vit.norm(vit_output)  # [batch_size, num_patches + 1, hidden_dim]
        cls_output = vit_output[:, 0, :]  # Take the [CLS] token: [batch_size, hidden_dim]

        # Step 5: Classification
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        return logits