
import os
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, class_weights, device,
                epochs=30, lr=1e-4, model_name="model", patience=5):

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_val_acc = 0.0
    save_path = f"/content/best_{model_name}.pth"

    wait = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        #Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Early Stopping 조건 확인
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            wait = 0
            torch.save(model.state_dict(), save_path)
            print(f"모델 저장됨 → {save_path}")
        else:
            wait += 1
            print(f"개선 없음 ({wait}/{patience})")

            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    return model, best_val_acc
