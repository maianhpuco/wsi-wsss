import torch
import torch.nn as nn
import timm

class VQGANViTClassifier(nn.Module):
    def __init__(self, vqgan_model, codebook_weights, num_classes=4, patch_size=16, vit_model_name="vit_base_patch16_224"):
        super(VQGANViTClassifier, self).__init__()
        self.vqgan = vqgan_model.eval()  # Pre-trained VQ-GAN model
        self.codebook_weights = codebook_weights  # [n_embed, embed_dim]
        self.embed_dim = codebook_weights.shape[1]  # e.g., 512
        self.patch_size = patch_size  # Number of patches per dimension (e.g., 16x16 patches for a 256x256 image)

        # Load ViT model from timm
        self.vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0)  # num_classes=0 to get features
        self.vit.eval()  # We'll fine-tune it during training
        vit_dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            vit_output = self.vit(vit_dummy_input)
        self.vit_hidden_dim = vit_output.shape[-1]  # e.g., 768 for vit_base_patch16_224

        # Project VQ-GAN embeddings to match ViT's expected patch embedding dimension
        self.vit_patch_dim = self.vit_hidden_dim  # ViT's patch embedding dimension
        self.projection = nn.Linear(self.embed_dim, self.vit_patch_dim)

        # Classification head
        self.classifier = nn.Linear(self.vit_hidden_dim, num_classes)

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
            _, _, [_, _, indices] = self.vqgan.encode(images)  # [batch_size, H', W'], H'=W'=14 for 224x224 images
        H, W = indices.shape[-2:]  # Should be 14x14 for 224x224 images with patch size 16

        # Step 2: Convert indices to embeddings
        embeddings = self.codebook_weights[indices]  # [batch_size, H', W', embed_dim]
        embeddings = embeddings.view(batch_size, H * W, self.embed_dim)  # [batch_size, num_patches, embed_dim]

        # Step 3: Project embeddings to match ViT's expected patch embedding dimension
        embeddings = self.projection(embeddings)  # [batch_size, num_patches, vit_patch_dim]

        # Step 4: Prepare embeddings for ViT
        # ViT expects [batch_size, num_patches, hidden_dim] + a [CLS] token
        cls_token = self.vit.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        embeddings = torch.cat([cls_token, embeddings], dim=1)  # [batch_size, num_patches + 1, hidden_dim]
        embeddings += self.vit.pos_embed[:, :embeddings.size(1), :]  # Add positional embeddings

        # Step 5: Pass through ViT
        vit_output = self.vit.blocks(embeddings)  # [batch_size, num_patches + 1, hidden_dim]
        vit_output = self.vit.norm(vit_output)  # [batch_size, num_patches + 1, hidden_dim]
        cls_output = vit_output[:, 0, :]  # Take the [CLS] token: [batch_size, hidden_dim]

        # Step 6: Classification
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        return logits 