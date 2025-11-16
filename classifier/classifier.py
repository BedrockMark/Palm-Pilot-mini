import os
import json
import logging
from typing import Optional, List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("main")
logger = logging.getLogger("main")

class GestureMLP(nn.Module):
    """
    多层感知机模型，用于手势分类
    
    结构:
    - Input: 63维 (21个关键点 × 3坐标)
    - Hidden Layer 1: 128 neurons, ReLU, Dropout(0.3)
    - Hidden Layer 2: 64 neurons, ReLU, Dropout(0.3)
    - Output: num_classes (动态调整)
    
    支持动态扩展输出层以适应新增手势类别
    """
    
    def __init__(self, input_dim: int = 63, hidden_dims: List[int] = [128, 64], num_classes: int = 2, dropout_rate: float = 0.3):
        """
        初始化MLP模型
        
        参数:
        - input_dim: 输入维度，默认63 (21个关键点 × 3坐标)
        - hidden_dims: 隐藏层维度列表，默认[128, 64]
        - num_classes: 输出类别数，默认2
        - dropout_rate: Dropout比率，默认0.3
        """
        super(GestureMLP, self).__init__()
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
        - x: 输入张量，shape: [batch_size, input_dim]
        
        返回:
        - 输出 logits，shape: [batch_size, num_classes]
        """
        return self.network(x)
    
    def extend_output_layer(self, new_num_classes: int) -> None:
        """
        扩展输出层以适应新的类别数，保留旧权重以防止灾难性遗忘
        
        参数:
        - new_num_classes: 新的类别数
        """
        if new_num_classes <= self.num_classes:
            logger.warning(f"New number of classes ({new_num_classes}) is not greater than current ({self.num_classes})")
            return
        
        # 获取当前网络结构
        layers = list(self.network.children())
        
        # 保存旧输出层的权重和偏置
        old_output_layer = layers[-1]
        if isinstance(old_output_layer, nn.Linear):
            old_weight = old_output_layer.weight.data.clone()
            old_bias = old_output_layer.bias.data.clone() if old_output_layer.bias is not None else None
        
        # 移除旧的输出层
        layers = layers[:-1]
        
        # 从后往前查找最后一个Linear层，获取其输出维度
        last_hidden_dim = 64  # 默认值
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                last_hidden_dim = layer.out_features
                break
        
        # 获取设备信息（从旧层获取）
        device = next(self.network.parameters()).device if len(list(self.network.parameters())) > 0 else torch.device('cpu')
        
        # 创建新的输出层并移动到正确的设备
        new_output_layer = nn.Linear(last_hidden_dim, new_num_classes).to(device)
        
        # 如果旧输出层存在，将旧权重复制到新输出层的前面部分
        if isinstance(old_output_layer, nn.Linear) and old_weight is not None:
            with torch.no_grad():
                new_output_layer.weight.data[:self.num_classes] = old_weight
                if old_bias is not None:
                    new_output_layer.bias.data[:self.num_classes] = old_bias
                # 新类别使用Xavier初始化
                nn.init.xavier_uniform_(new_output_layer.weight.data[self.num_classes:])
                if new_output_layer.bias is not None:
                    nn.init.zeros_(new_output_layer.bias.data[self.num_classes:])
        
        # 添加新的输出层
        layers.append(new_output_layer)
        
        # 重新构建网络
        self.network = nn.Sequential(*layers)
        self.num_classes = new_num_classes
        
        logger.info(f"Extended output layer to {new_num_classes} classes")

class GestureDataset(Dataset):
    """
    手势数据集类
    """
    
    def __init__(self, data: List[np.ndarray], labels: List[int]):
        """
        Initiate a dataset
        
        args:
        - data: List of 21 hand critical points in shape(21,3)
        - labels: List of corresponding gesture index
        """
        if len(data) != len(labels):
            raise ValueError(f"Data length ({len(data)}) does not match labels length ({len(labels)})")
        
        self.data = [torch.FloatTensor(sample.flatten()) for sample in data]
        self.labels = torch.LongTensor(labels)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


class GestureClassifier:
    """
    手势分类器类
    
    功能:
    1. **模型保存/加载**: 实现save_model()和load_model()方法
    2. **数据增强**(可选): 在训练时添加噪声、微小旋转等增强鲁棒性
    3. **可视化**(可选): 绘制训练曲线、混淆矩阵
    4. **日志记录**: 使用logging记录关键操作
    """
    
    def __init__(self, model_path: Optional[str] = None, weight_data: Optional[str] = None):
        """
        初始化手势分类器
        
        参数:
        - model_path: 预训练的PyTorch模型权重文件路径 (可选)
        - weight_data: 对应的权重数据/元数据文件路径 (可选,如类别标签、归一化参数等)
        
        功能:
        - 如果提供model_path,加载预训练模型
        - 如果未提供,初始化一个新的MLP神经网络
        - 初始化类别映射字典 (gesture_id -> gesture_name)
        - 设置设备 (CPU/CUDA)
        """
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # 初始化类别映射
        self.gesture_map: Dict[int, str] = {}
        self.next_gesture_id = 0
        
        # 初始化回放缓冲区：存储每个类别的代表性样本，防止灾难性遗忘
        # 格式: {gesture_id: List[np.ndarray]}，每个类别最多保存 max_replay_samples 个样本
        self.replay_buffer: Dict[int, List[np.ndarray]] = {}
        self.max_replay_samples = 50  # 每个类别最多保存的样本数
        
        # 初始化模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, weight_data)
        else:
            # 创建新模型
            self.model = GestureMLP(input_dim=63, num_classes=2).to(self.device)
            logger.info("Initialized new model")
    
    def normalize(self, landmarks: np.ndarray, rotation_normalize: bool = True) -> np.ndarray:
        """
        归一化MediaPipe手部关键点数据
        
        参数:
        - landmarks: MediaPipe检测到的手部21个关键点坐标 (x, y, z)
                    shape: [21, 3] 或 list of landmark objects
        - rotation_normalize: bool,是否进行旋转归一化 (默认True)
        
        处理步骤:
        1. 平移归一化:以手腕关键点(landmark[0])为原点,所有点减去手腕坐标
        2. 尺度归一化:计算手部范围(max-min),将坐标缩放到[-1,1]
        3. 旋转归一化(可选):
        - 基于手掌方向(如手腕到中指根部向量)计算旋转角度
        - 将手势旋转到标准方向,消除手部旋转影响
        
        返回:
        - normalized_landmarks: 归一化后的关键点数组 (shape: [21, 3])
        """
        # 参数验证
        if landmarks is None:
            raise ValueError("Landmarks cannot be None")
        
        # 转换为numpy数组
        if isinstance(landmarks, list):
            # 如果是MediaPipe的landmark对象列表
            if hasattr(landmarks[0], 'x'):
                landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            else:
                landmarks_array = np.array(landmarks)
        else:
            landmarks_array = np.array(landmarks)
        
        # 验证形状
        if landmarks_array.shape != (21, 3):
            raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks_array.shape}")
        
        # 1. 平移归一化：以手腕(索引0)为原点
        wrist = landmarks_array[0].copy()
        normalized = landmarks_array - wrist
        
        # 2. 尺度归一化：缩放到[-1, 1]
        max_val = np.abs(normalized).max()
        if max_val > 0:
            normalized = normalized / max_val
        
        # 3. 旋转归一化（可选）
        if rotation_normalize:
            # 使用手腕到中指根部(索引9)的向量作为参考方向
            if np.linalg.norm(normalized[9]) > 1e-6:
                # 计算旋转角度，使中指根部指向正y方向
                reference_vector = normalized[9]
                angle = np.arctan2(reference_vector[0], reference_vector[1])
                
                # 绕z轴旋转
                cos_angle = np.cos(-angle)
                sin_angle = np.sin(-angle)
                rotation_matrix = np.array([
                    [cos_angle, -sin_angle, 0],
                    [sin_angle, cos_angle, 0],
                    [0, 0, 1]
                ])
                
                normalized = normalized @ rotation_matrix.T
        
        return normalized
    
    def _add_noise(self, data: np.ndarray, noise_scale: float = 0.01) -> np.ndarray:
        """
        添加噪声进行数据增强
        
        参数:
        - data: 输入数据
        - noise_scale: 噪声缩放因子
        
        返回:
        - 添加噪声后的数据
        """
        noise = np.random.normal(0, noise_scale, data.shape)
        return data + noise
    
    def train(self, training_data: List[np.ndarray], labels: List[int], gesture_name: str, 
              epochs: int = 100, lr: float = 0.001, batch_size: int = 32, 
              use_augmentation: bool = True) -> Dict[str, List[float]]:
        """
        训练MLP模型识别新手势类别
        
        参数:
        - training_data: 训练数据列表,每个元素是归一化后的手势关键点 (shape: [21, 3])
        - labels: 对应的标签列表
        - gesture_name: 新增的手势类别名称
        - epochs: 训练轮数，默认100
        - lr: 学习率，默认0.001
        - batch_size: 批次大小，默认32
        - use_augmentation: 是否使用数据增强，默认True
        
        功能:
        1. 数据预处理:
        - 将training_data转换为PyTorch张量
        - 创建DataLoader进行批处理
        
        2. 模型动态扩展:
        - 如果gesture_name是新类别,扩展输出层维度
        - 更新类别映射字典
        
        3. 训练过程:
        - 使用交叉熵损失函数
        - Adam优化器
        - 显示训练进度和损失
        
        4. 保存模型:
        - 训练完成后自动保存模型权重
        - 保存类别映射等元数据
        
        返回:
        - training_history: 训练历史(loss, accuracy等)
        """
        # 参数验证
        if not training_data or not labels:
            raise ValueError("Training data and labels cannot be empty")
        
        if len(training_data) != len(labels):
            raise ValueError(f"Training data length ({len(training_data)}) does not match labels length ({len(labels)})")
        
        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0")
        
        if lr <= 0:
            raise ValueError("Learning rate must be greater than 0")
        
        logger.info(f"Starting training for gesture: {gesture_name}, {len(training_data)} samples")
        
        # 检查是否需要添加新类别
        if gesture_name not in self.gesture_map.values():
            # 添加新类别
            gesture_id = self.next_gesture_id
            self.gesture_map[gesture_id] = gesture_name
            self.next_gesture_id += 1
            logger.info(f"Added new gesture class: {gesture_name} (ID: {gesture_id})")
            
            # 扩展模型输出层
            new_num_classes = len(self.gesture_map)
            self.model.extend_output_layer(new_num_classes)
        else:
            # 获取已存在类别的ID
            gesture_id = next(k for k, v in self.gesture_map.items() if v == gesture_name)
            logger.info(f"Using existing gesture class: {gesture_name} (ID: {gesture_id})")
        
        # 验证标签范围
        max_label = max(labels)
        min_label = min(labels)
        num_classes = len(self.gesture_map)
        
        if max_label >= num_classes:
            raise ValueError(f"Label {max_label} exceeds number of classes ({num_classes})")
        
        if min_label < 0:
            raise ValueError(f"Label {min_label} is negative")
        
        # 验证标签是否与gesture_name匹配（如果所有标签相同）
        unique_labels = set(labels)
        if len(unique_labels) == 1:
            label_value = list(unique_labels)[0]
            if label_value != gesture_id:
                logger.warning(f"Label {label_value} does not match gesture_id {gesture_id} for '{gesture_name}'. Using provided labels.")
        
        # 更新回放缓冲区：将新训练样本添加到缓冲区
        if gesture_id not in self.replay_buffer:
            self.replay_buffer[gesture_id] = []
        
        # 添加新样本到回放缓冲区（随机采样以保持多样性）
        for sample in training_data:
            if len(self.replay_buffer[gesture_id]) < self.max_replay_samples:
                self.replay_buffer[gesture_id].append(sample.copy())
            else:
                # 缓冲区已满，随机替换
                if np.random.random() < 0.5:  # 50%概率替换
                    idx = np.random.randint(0, self.max_replay_samples)
                    self.replay_buffer[gesture_id][idx] = sample.copy()
        
        # 收集回放数据：从其他类别的缓冲区中获取样本
        replay_data = []
        replay_labels = []
        for other_gesture_id, samples in self.replay_buffer.items():
            if other_gesture_id != gesture_id and len(samples) > 0:
                # 从每个旧类别中采样，数量与新类别样本数成比例
                num_replay = min(len(samples), max(1, len(training_data) // 2))
                if num_replay > 0:
                    if len(samples) == 1:
                        selected_indices = [0]
                    else:
                        selected_indices = np.random.choice(len(samples), min(num_replay, len(samples)), replace=False)
                    for idx in selected_indices:
                        replay_data.append(samples[idx].copy())
                        replay_labels.append(other_gesture_id)
        
        # 合并新数据和回放数据
        all_training_data = training_data + replay_data
        all_labels = labels + replay_labels
        
        if len(replay_data) > 0:
            unique_old_classes = len(set(replay_labels))
            logger.info(f"Training with {len(training_data)} new samples and {len(replay_data)} replay samples from {unique_old_classes} old classes")
        else:
            logger.info(f"Training with {len(training_data)} new samples (no previous classes to replay)")
        
        # 数据增强
        if use_augmentation:
            # 只对新数据进行增强，回放数据不增强
            augmented_data = all_training_data.copy()
            for _ in range(len(training_data)):  # 新数据量翻倍
                sample = training_data[np.random.randint(0, len(training_data))]
                augmented_sample = self._add_noise(sample)
                augmented_data.append(augmented_sample)
            
            # 扩展标签（只扩展新数据的标签）
            all_labels_extended = all_labels + labels
        else:
            augmented_data = all_training_data
            all_labels_extended = all_labels
        
        # 创建数据集和数据加载器
        dataset = GestureDataset(augmented_data, all_labels_extended)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 设置损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 训练历史
        training_history = {
            'loss': [],
            'accuracy': []
        }
        
        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            avg_loss = epoch_loss / len(dataloader)
            accuracy = 100.0 * correct / total
            
            training_history['loss'].append(avg_loss)
            training_history['accuracy'].append(accuracy)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        logger.info(f"Training completed for gesture: {gesture_name}")
        return training_history
    
    def inference(self, landmarks: np.ndarray, confidence_threshold: float = 0.7) -> Tuple[int, str, float]:
        """
        推理识别手势类别
        
        参数:
        - landmarks: MediaPipe检测到的手部关键点
        - confidence_threshold: 置信度阈值,低于此值认为无法识别，默认0.7
        
        处理流程:
        1. 调用normalize方法归一化输入数据
        2. 将数据转换为PyTorch张量
        3. 通过模型前向传播获取预测
        4. 应用Softmax获取概率分布
        5. 判断最高置信度是否超过阈值
        
        返回:
        - gesture_id: 识别到的手势序号 (int),如果无法识别返回-1
        - gesture_name: 手势名称 (str)
        - confidence: 置信度分数 (float)
        """
        # 参数验证
        if landmarks is None:
            raise ValueError("Landmarks cannot be None")
        
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        # 归一化
        try:
            normalized = self.normalize(landmarks, rotation_normalize=True)
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return -1, "unknown", 0.0
        
        # 转换为张量
        input_tensor = torch.FloatTensor(normalized.flatten()).unsqueeze(0).to(self.device)
        
        # 推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence_value = confidence.item()
            gesture_id = predicted.item()
        
        # 检查置信度阈值
        if confidence_value < confidence_threshold:
            logger.debug(f"Confidence {confidence_value:.4f} below threshold {confidence_threshold}")
            return -1, "unknown", confidence_value
        
        # 获取手势名称
        gesture_name = self.gesture_map.get(gesture_id, "unknown")
        
        return gesture_id, gesture_name, confidence_value
    
    def save_model(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        保存模型权重和元数据
        
        参数:
        - model_path: 模型权重保存路径
        - metadata_path: 元数据保存路径（可选，默认与model_path同目录）
        """
        if metadata_path is None:
            metadata_path = model_path.replace('.pth', '_metadata.json')
        
        # 保存模型权重
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
        
        # 保存元数据（包括回放缓冲区）
        # 将回放缓冲区转换为可序列化格式（numpy数组转列表）
        replay_buffer_serializable = {}
        for gesture_id, samples in self.replay_buffer.items():
            replay_buffer_serializable[str(gesture_id)] = [sample.tolist() for sample in samples]
        
        metadata = {
            'gesture_map': self.gesture_map,
            'next_gesture_id': self.next_gesture_id,
            'replay_buffer': replay_buffer_serializable,
            'max_replay_samples': self.max_replay_samples,
            'model_config': {
                'input_dim': 63,
                'hidden_dims': [128, 64],
                'num_classes': len(self.gesture_map),
                'dropout_rate': 0.3
            }
        }
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def load_model(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        加载模型权重和元数据
        
        参数:
        - model_path: 模型权重文件路径
        - metadata_path: 元数据文件路径（可选）
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if metadata_path is None:
            metadata_path = model_path.replace('.pth', '_metadata.json')
        
        # 加载元数据
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.gesture_map = {int(k): v for k, v in metadata['gesture_map'].items()}
                self.next_gesture_id = metadata.get('next_gesture_id', len(self.gesture_map))
                model_config = metadata.get('model_config', {})
                
                # 加载回放缓冲区
                if 'replay_buffer' in metadata:
                    replay_buffer_data = metadata['replay_buffer']
                    self.replay_buffer = {}
                    for gesture_id_str, samples_list in replay_buffer_data.items():
                        gesture_id = int(gesture_id_str)
                        self.replay_buffer[gesture_id] = [np.array(sample) for sample in samples_list]
                    logger.info(f"Loaded replay buffer with {len(self.replay_buffer)} gesture classes")
                else:
                    self.replay_buffer = {}
                    logger.info("No replay buffer found in metadata, initializing empty buffer")
                
                # 加载最大回放样本数
                self.max_replay_samples = metadata.get('max_replay_samples', 50)
                
                logger.info(f"Loaded metadata: {len(self.gesture_map)} gesture classes")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}, using default values")
                model_config = {'input_dim': 63, 'hidden_dims': [128, 64], 'num_classes': 2, 'dropout_rate': 0.3}
        else:
            logger.warning(f"Metadata file not found: {metadata_path}, using default values")
            model_config = {'input_dim': 63, 'hidden_dims': [128, 64], 'num_classes': 2, 'dropout_rate': 0.3}
        
        # 创建模型
        num_classes = len(self.gesture_map) if self.gesture_map else model_config.get('num_classes', 2)
        self.model = GestureMLP(
            input_dim=model_config.get('input_dim', 63),
            hidden_dims=model_config.get('hidden_dims', [128, 64]),
            num_classes=num_classes,
            dropout_rate=model_config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        # 加载权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise