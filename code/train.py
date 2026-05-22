import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing import QuantileTransformer
import joblib
from data import DataReader, NumpyDataset, HybridNormalizer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from FNO import FNO2d
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

time_pairs = [
    ('2024030100', '2024030101'), ('2024030101', '2024030102'),
    ('2024030102', '2024030103'), ('2024030103', '2024030104'),
    ('2024030104', '2024030105'),
    ('2024030106', '2024030107'), ('2024030107', '2024030108'),
    ('2024030108', '2024030109'), ('2024030109', '2024030110'),
    ('2024030110', '2024030111'), ('2024030111', '2024030112'),
    ('2024030112', '2024030113'), ('2024030113', '2024030114'),
    ('2024030114', '2024030115'), ('2024030115', '2024030116'),
    ('2024030116', '2024030117'),
    ('2024030118', '2024030119'), ('2024030119', '2024030120'),
    ('2024030120', '2024030121'), ('2024030121', '2024030122'),
    ('2024030122', '2024030123'), ('2024030123', '2024030200'),
]

directories = {
    'pm25': 'your_path/pm25',
    'output': 'your_path/output',
    'T2': 'your_path/T2',
    'E_pm25': 'your_path/E_pm25',
    'HGT': 'your_path/HGT',
    'U10': 'your_path/U10',
    'V10': 'your_path/V10',
    'LAI': 'your_path/LAI',
    'QVAPOR': 'your_path/QVAPOR',
    'PBLH': 'your_path/PBLH',
    'UST': 'your_path/UST',
    'HFX': 'your_path/HFX',
}

def create_hybrid_normalizer(normalization_data):
    normalizer = HybridNormalizer()
    normalizer.fit(normalization_data)
    return normalizer

data_reader = DataReader(
    i_range=range(1, 8),
    j_range=range(1, 12),
    directories=directories,
    time_pairs=time_pairs,
)

print("Loading data for normalization statistics...")
normalization_data = data_reader.load_all_data_for_normalization()

normalizer = create_hybrid_normalizer(normalization_data)

save_path = "./hybrid_normalizer.pkl"
normalizer.save(save_path)

norm_info = normalizer.get_normalization_info()
print("\n=== Normalization Methods ===")
for field, info in norm_info.items():
    print(f"{field:<10}: {info['method']}")

dataset = NumpyDataset(data_reader, normalizer=normalizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

val_time_pairs = [('2024030105', '2024030106'), ('2024030117', '2024030118')]
val_data_reader = DataReader(
    i_range=range(1, 8),
    j_range=range(1, 12),
    directories=directories,
    time_pairs=val_time_pairs,
)

val_normalizer = HybridNormalizer()
val_normalizer.load(save_path)

val_dataset = NumpyDataset(
    data_reader=val_data_reader,
    normalizer=val_normalizer,
    transform=None,
)

val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

H_START, H_END = 10, 90
W_START, W_END = 10, 130

criterion = nn.MSELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FNO2d(in_channels=len(directories)-1, out_channels=1,
              modes1=24, modes2=24, width=32).to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3,
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=6,
    min_lr=1e-6,
    verbose=True
)

num_epochs = 120
early_stop_patience = 20

train_losses = []
val_mae = []
val_rmse = []
learning_rates = []

best_val_mae = float('inf')
best_epoch = 0
early_stop_counter = 0

log_file = './training_log.txt'
with open(log_file, 'a') as f:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels, _, _ in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            outputs_crop = outputs[:, :, H_START:H_END, W_START:W_END]
            labels_crop = labels[:, :, H_START:H_END, W_START:W_END]

            loss = criterion(outputs_crop, labels_crop)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(dataloader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_mae_value = 0.0
        val_rmse_value = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for val_images, val_labels, _, _ in val_dataloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_outputs = model(val_images)

                val_outputs_crop = val_outputs[:, :, H_START:H_END, W_START:W_END]
                val_labels_crop = val_labels[:, :, H_START:H_END, W_START:W_END]

                val_outputs_denorm = val_normalizer.inverse_transform(
                    val_outputs_crop.cpu().numpy(), data_type='output'
                )
                val_labels_denorm = val_normalizer.inverse_transform(
                    val_labels_crop.cpu().numpy(), data_type='output'
                )

                batch_mae = mean_absolute_error(val_labels_denorm.flatten(), val_outputs_denorm.flatten())
                batch_rmse = np.sqrt(np.mean((val_outputs_denorm.flatten() - val_labels_denorm.flatten()) ** 2))

                val_mae_value += batch_mae
                val_rmse_value += batch_rmse
                val_batch_count += 1

        avg_val_mae = val_mae_value / val_batch_count
        avg_val_rmse = val_rmse_value / val_batch_count
        val_mae.append(avg_val_mae)
        val_rmse.append(avg_val_rmse)

        scheduler.step(avg_val_mae)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        f.write(f"Epoch {epoch + 1}/{num_epochs} Train Loss: {avg_train_loss:.4f} Val MAE: {avg_val_mae:.4f} Val RMSE: {avg_val_rmse:.4f}\n")
        f.flush()

        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_epoch = epoch
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mae': best_val_mae,
                'val_rmse': avg_val_rmse,
                'crop_params': {'h_start': H_START, 'h_end': H_END, 'w_start': W_START, 'w_end': W_END}
            }, './best_fno_model.pth')
            f.write(f">>> Best model saved! Val MAE: {best_val_mae:.4f}\n")
        else:
            early_stop_counter += 1
            f.write(f"  Early stop count: {early_stop_counter}/{early_stop_patience}\n")

        if early_stop_counter >= early_stop_patience:
            f.write(f'Early stopping triggered at epoch {epoch+1}\n')
            break
