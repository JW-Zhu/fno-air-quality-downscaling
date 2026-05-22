import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing import QuantileTransformer
import joblib


class DataReader:
    """
    Data reader module with outlier blacklist filtering.
    """
    def __init__(self, i_range, j_range, directories, time_pairs=None):
        self.i_range = i_range
        self.j_range = j_range
        self.directories = directories
        
        self.time_pairs = time_pairs if time_pairs else [
            ('2024030100', '2024030101'),
            ('2024030101', '2024030102'),
            ('2024030102', '2024030103'),
            ('2024030103', '2024030104')
        ]
        
        # =========================================================
        # 剔除异常文件 (数值 > 500 的坏数据)
        # 格式：i{i}_j{j}_{start_time}
        # 这些数据会导致归一化失效或验证集 MAE 爆炸
        # =========================================================
        self.blacklist = {
            "i1_j6_2024030100",
            "i3_j1_2024030100",
            "i3_j1_2024030101",
            "i5_j2_2024030100",
            "i5_j2_2024030101",
            "i6_j2_2024030100",
            "i6_j2_2024030101",
            "i7_j1_2024030100",
            "i7_j1_2024030101",
            "i7_j10_2024030122",
        }
        
        self.valid_samples = self._check_files()
        
    def _check_files(self):
        """
        Pre-check all files, filter out blacklisted samples, return valid sample list.
        """
        valid_samples = []
        skipped_count = 0
        
        for i in self.i_range:
            for j in self.j_range:
                for start_time, end_time in self.time_pairs:
                    sample_id = f"i{i}_j{j}_{start_time}"
                    if sample_id in self.blacklist:
                        skipped_count += 1
                        continue
                    
                    paths = {key: os.path.join(dir_path, f"i{i}_j{j}", f"{start_time}.npy") 
                            for key, dir_path in self.directories.items()}
                    
                    if all(os.path.exists(path) for path in paths.values()):
                        valid_samples.append((i, j, start_time, end_time))

        print(f"Data check complete: {len(valid_samples)} valid samples ({skipped_count} blacklisted files excluded)")
        return valid_samples

    def load_sample_data(self, idx):
        """
        Load all data for a single sample.
        """
        i, j, start_time, end_time = self.valid_samples[idx]
        paths = {key: os.path.join(dir_path, f"i{i}_j{j}", f"{start_time}.npy") 
                 for key, dir_path in self.directories.items()}
        data = {key: np.load(path) for key, path in paths.items()}
        data['metadata'] = (i, j, start_time, end_time)
        return data

    def load_all_data_for_normalization(self, max_samples=10000):
        """
        Load data for computing normalization statistics.
        """
        print("Loading data for normalization...")
        sample_count = min(max_samples, len(self.valid_samples))
        indices = np.random.choice(len(self.valid_samples), sample_count, replace=False)
        data_dict = {key: [] for key in self.directories.keys()}
        for idx in indices:
            sample_data = self.load_sample_data(idx)
            for key in data_dict.keys():
                data_dict[key].append(sample_data[key])
        return data_dict


class NumpyDataset(Dataset):
    def __init__(self, data_reader, normalizer=None, transform=None):
        """
        PyTorch Dataset for loading and transforming data.
        """
        self.data_reader = data_reader
        self.normalizer = normalizer
        self.transform = transform
        self.valid_samples = data_reader.valid_samples
        self.fields = ['pm25', 'LAI', 'QVAPOR', 'E_pm25', 'HGT', 'U10',
                        'V10', 'T2', 'PBLH', 'UST', 'HFX', 'output']

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_data = self.data_reader.load_sample_data(idx)
        i, j, start_time, end_time = sample_data['metadata']
        
        if self.normalizer:
            data = {field: self.normalizer.transform(sample_data[field], field) for field in self.fields}
        else:
            data = {field: sample_data[field] for field in self.fields}

        tensors = {field: torch.tensor(data[field], dtype=torch.float32) for field in self.fields}

        if self.transform:
            for field in self.fields:
                tensors[field] = self.transform(tensors[field])

        for field in self.fields:
            tensors[field] = tensors[field].unsqueeze(0)
        
        input_tensor = torch.cat([tensors[field] for field in self.fields[:-1]], dim=0)

        return input_tensor, tensors['output'], i, j
   
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
import joblib


class CustomRobustScaler2:
    """
    Robust scaler using IQR (25-75) scaling to resist outlier influence.
    """
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.scaler = RobustScaler(quantile_range=self.quantile_range)
        
    def fit(self, data):
        data_flat = data.flatten()
        data_valid = data_flat[~np.isnan(data_flat)]
        if len(data_valid) > 0:
            self.scaler.fit(data_valid.reshape(-1, 1))
        return self
        
    def transform(self, data):
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        return self.scaler.transform(data_flat).reshape(original_shape)
        
    def inverse_transform(self, data):
        original_shape = data.shape
        data_flat = data.reshape(-1, 1)
        return self.scaler.inverse_transform(data_flat).reshape(original_shape)

class HybridNormalizer:
    """
    Hybrid normalizer: specific variables use CustomRobustScaler, others use appropriate normalization methods.
    """
    def __init__(self, quantile_range=(1, 99)):
        self.quantile_range = quantile_range
        self.custom_robust_fields = ['E_pm25', 'pm25', 'output']
        self.scalers = {}
        self.normalization_params = {}
        self.fitted = False
        self.custom_robust_scaler = None
        
    def fit(self, data_dict):
        """
        Fit the normalizer on a dictionary of data arrays.
        """
        print("Fitting hybrid normalizer (Global Normalization)...")
        
        custom_robust_data = []
        for field in self.custom_robust_fields:
            if field in data_dict and len(data_dict[field]) > 0:
                field_data = np.concatenate([np.array(data).flatten() for data in data_dict[field]])
                field_data = field_data[~np.isnan(field_data)]
                field_data = field_data[np.isfinite(field_data)]
                custom_robust_data.extend(field_data)
        
        if custom_robust_data:
            custom_robust_data = np.array(custom_robust_data)
            self.custom_robust_scaler = CustomRobustScaler2(self.quantile_range)
            self.custom_robust_scaler.fit(custom_robust_data)
            print(f"CustomRobustScaler fitted ({len(custom_robust_data)} data points)")
        else:
            self.custom_robust_scaler = None
        
        for field, data_list in data_dict.items():
            if len(data_list) == 0:
                continue
            
            if field in self.custom_robust_fields:
                self.scalers[field] = 'custom_robust'
                continue

            all_data = np.concatenate([np.array(data).flatten() for data in data_list])
            all_data = all_data[~np.isnan(all_data)]
            all_data = all_data[np.isfinite(all_data)]
            
            if len(all_data) == 0:
                continue
            
            if field in ['T2', 'U10', 'V10', 'LAI', 'QVAPOR', 'UST', 'HFX']:
                scaler = StandardScaler()
                method = 'standard'
            elif field == 'PBLH':
                scaler = PowerTransformer(method='yeo-johnson')
                method = 'power_yeo_johnson'
            else:
                scaler = RobustScaler()
                method = 'robust'
            
            scaler.fit(all_data.reshape(-1, 1))
            self.scalers[field] = scaler
            print(f"Field {field}: using {method} normalization")
            
            self.normalization_params[field] = {
                'min': float(all_data.min()),
                'max': float(all_data.max()),
                'mean': float(all_data.mean()),
                'std': float(all_data.std()),
            }
        
        self.fitted = True
        return self
    
    def transform(self, data, field=None, data_type=None):
        """
        Apply normalization to data.
        """
        if data_type is not None: field = data_type
        if field is None: raise ValueError("field parameter is required")
        if not self.fitted: raise ValueError("Call fit() first")
        if field not in self.scalers: return data
            
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
        
        if self.scalers[field] == 'custom_robust':
            if self.custom_robust_scaler is None: return data
            transformed = self.custom_robust_scaler.transform(data_reshaped)
        else:
            transformed = self.scalers[field].transform(data_reshaped)
        
        return transformed.reshape(original_shape)
    
    def inverse_transform(self, data, field=None, data_type=None):
        """
        Apply inverse normalization to data.
        """
        if data_type is not None: field = data_type
        if field is None: raise ValueError("field parameter is required")
        if not self.fitted: raise ValueError("Call fit() first")
        if field not in self.scalers: return data
            
        original_shape = data.shape
        data_reshaped = data.reshape(-1, 1)
        
        if self.scalers[field] == 'custom_robust':
            if self.custom_robust_scaler is None: return data
            inverted = self.custom_robust_scaler.inverse_transform(data_reshaped)
        else:
            inverted = self.scalers[field].inverse_transform(data_reshaped)
        
        return inverted.reshape(original_shape)

    def save(self, filepath):
        save_data = {
            'quantile_range': self.quantile_range,
            'custom_robust_fields': self.custom_robust_fields,
            # 保存 scaler 对象
            'scalers': self.scalers, 
            'custom_robust_scaler': self.custom_robust_scaler,
            'normalization_params': self.normalization_params,
            'fitted': self.fitted
        }
        joblib.dump(save_data, filepath)
        print(f"归一化器已保存: {filepath}")
    
    def load(self, filepath):
        saved_data = joblib.load(filepath)
        self.quantile_range = saved_data['quantile_range']
        self.custom_robust_fields = saved_data['custom_robust_fields']
        self.scalers = saved_data['scalers']
        self.custom_robust_scaler = saved_data['custom_robust_scaler']
        self.normalization_params = saved_data['normalization_params']
        self.fitted = saved_data['fitted']
        print(f"归一化器已加载")
        return self
    
    def get_normalization_info(self):
        info = {}
        for field, params in self.normalization_params.items():
            method = 'custom_robust' if field in self.custom_robust_fields else type(self.scalers[field]).__name__
            info[field] = {'method': method, **params}
        return info