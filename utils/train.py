import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, train_loader, val_loader, optimizer, num_epochs, device):
    # Multi-label classification loss (binary cross-entropy with logits)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc="Training"):
            images = batch["image"].to(device)  # [batch_size, 3, 224, 224]
            labels = batch["class_label"].to(device).float()  # [batch_size, 4]

            optimizer.zero_grad()

            # Forward pass
            logits = model(images)  # [batch_size, 4]

            # Compute loss
            loss = criterion(logits, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute accuracy (threshold at 0.5 for multi-label)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += ((preds == labels).float().sum(dim=1) == 4).sum().item()
            train_total += images.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch["image"].to(device)
                # Derive class_label from label if not provided
                if "class_label" in batch:
                    labels = batch["class_label"].to(device).float()
                else:
                    labels_mask = batch["label"].to(device)  # [batch_size, 224, 224]
                    # Create multi-label classification labels
                    labels = torch.zeros(labels_mask.size(0), 4, device=device)
                    for cls in range(1, 5):  # Classes 1-4 (TUM, STR, LYM, NEC)
                        labels[:, cls-1] = (labels_mask == cls).any(dim=(1, 2)).float()

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += ((preds == labels).float().sum(dim=1) == 4).sum().item()
                val_total += images.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

def train_indice(model, train_loader, val_loader, optimizer, num_epochs, device):
    # Multi-label classification loss (binary cross-entropy with logits)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc="Training"):
            indices = batch["indices"].to(device)  # [batch_size, 28, 28]
            labels = batch["class_label"].to(device).float()  # [batch_size, 4]

            optimizer.zero_grad()

            # Forward pass
            logits = model(indices)  # [batch_size, 4]

            # Compute loss
            loss = criterion(logits, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute accuracy (threshold at 0.5 for multi-label)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += ((preds == labels).float().sum(dim=1) == 4).sum().item()
            train_total += indices.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                indices = batch["indices"].to(device)  # [batch_size, 28, 28]
                # Derive class_label from label if not provided
                if "class_label" in batch:
                    labels = batch["class_label"].to(device).float()
                else:
                    labels_mask = batch["label"].to(device)  # [batch_size, 224, 224]
                    # Create multi-label classification labels
                    labels = torch.zeros(labels_mask.size(0), 4, device=device)
                    for cls in range(1, 5):  # Classes 1-4 (TUM, STR, LYM, NEC)
                        labels[:, cls-1] = (labels_mask == cls).any(dim=(1, 2)).float()

                logits = model(indices)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += ((preds == labels).float().sum(dim=1) == 4).sum().item()
                val_total += indices.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")