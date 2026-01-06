import torch
from torch.utils.data import DataLoader, TensorDataset
from models.UNet import get_unet_acdc, get_unet_brats
from train_utils import get_loss_function, load_config
from data_loader import ACDCDataset, BraTSDataset
import torch.optim as optim
import os

def train_unet(dataset_name="ACDC", data_path=None, batch_size=8, epochs=50, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training U-Net on {dataset_name} using {device}")

    # 1. 加载预划分的数据 (train/val 子目录)
    if data_path:
        data_path=os.path.join(data_path, dataset_name+"_Processed/")
        print(data_path)

        train_dir = os.path.join(data_path, "train")
        val_dir = os.path.join(data_path, "val")
        
        if dataset_name == "ACDC":
            train_ds = ACDCDataset(train_dir)
            val_ds = ACDCDataset(val_dir)
            model = get_unet_acdc().to(device)
        else:
            train_ds = BraTSDataset(train_dir)
            val_ds = BraTSDataset(val_dir)
            model = get_unet_brats().to(device)
            
        print(f"Loaded pre-split data: Train={len(train_ds)}, Val={len(val_ds)}")
    else:
        print("Warning: data_path/train not found, using dummy data.")
        model = get_unet_acdc().to(device) if dataset_name == "ACDC" else get_unet_brats().to(device)
        dummy_x = torch.randn(batch_size * 2, 1 if dataset_name == "ACDC" else 4, 224, 224)
        dummy_y = torch.randint(0, 4, (batch_size * 2, 224, 224))
        train_ds = TensorDataset(dummy_x, dummy_y)
        val_ds = TensorDataset(dummy_x, dummy_y)

    # 2. 数据加载器
    num_workers = 4 if dataset_name == "BraTS" else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 3. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = get_loss_function(deep_supervision=False)

    # 4. 训练循环
    scaler = torch.amp.GradScaler('cuda') if dataset_name == "BraTS" else None
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(batch_x)
                    loss = loss_fn(pred, batch_y)
                
                # 检查 loss 是否为 NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at epoch {epoch+1}")
                    continue

                scaler.scale(loss).backward()
                # 梯度裁剪防止梯度爆炸导致 NaN
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"./checkpoints/unet/{dataset_name.lower()}/unet_best.pth")

    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    # 示例: 训练 ACDC
    config = load_config("config.yaml")

    dataset_name = config['train']['dataset_name']
    batch_size = config['train']['batch_size']
    epochs = config['train']['epochs']
    lr = config['train'].get('learning_rate', 1e-3)
    data_path = config['path']['data_path']

    train_unet(dataset_name=dataset_name, data_path=data_path, batch_size=batch_size, epochs=epochs, lr=lr)
    # 示例: 训练 BraTS
    # train_unet(dataset_name="BraTS", batch_size=2, epochs=5)
