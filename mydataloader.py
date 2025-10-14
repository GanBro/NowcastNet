import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
from pathlib import Path
import glob


class Sevir(data.Dataset):
    """
    SEVIR天气数据集类 - 直接从目录读取npz文件

    输出格式:
        - 输入: 4帧 × 256×256
        - 输出: 15帧 × 256×256
        - 数值范围: [-9, 60]
        - 格式: (B, T, H, W)
    """

    def __init__(self, train=True, data_path=None, data_dir=None):
        """
        初始化SEVIR数据集

        参数:
            train (bool): True=训练集, False=测试集
            data_path (str): 单个npz文件路径（用于测试单个文件）
            data_dir (str): npz文件所在目录（默认会搜索所有npz文件）
        """
        super(Sevir, self).__init__()
        self.train = train

        if data_path is not None:
            # 单文件模式
            data_path = self._resolve_path(data_path)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件不存在: {data_path}")

            self.single_file_mode = True
            self.data_path = data_path
            self.data = [data_path]
            print(f"✓ 加载单个文件: {data_path}")
        else:
            # 目录模式：从指定目录读取所有npz文件
            self.single_file_mode = False

            # 尝试多个可能的数据目录
            possible_dirs = [
                data_dir,  # 用户指定的目录
                "./data",  # 项目根目录下的data文件夹
                "../data",  # 上级目录的data文件夹
            ]

            # 找到第一个存在的目录
            self.data_dir = None
            for dir_path in possible_dirs:
                if dir_path and os.path.exists(dir_path):
                    self.data_dir = dir_path
                    break

            if self.data_dir is None:
                raise FileNotFoundError(
                    f"找不到SEVIR数据目录。请指定data_dir参数，或将.npz文件放在以下任一目录:\n" +
                    "\n".join(f"  - {d}" for d in possible_dirs if d)
                )

            # 搜索所有npz文件
            npz_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))

            if len(npz_files) == 0:
                raise FileNotFoundError(
                    f"在目录 {self.data_dir} 中没有找到任何.npz文件"
                )

            print(f"✓ 在 {self.data_dir} 中找到 {len(npz_files)} 个.npz文件")

            # 划分训练集和测试集 (80/20分割)
            split_idx = int(len(npz_files) * 0.8)
            if train:
                self.data = npz_files[:split_idx]
                print(f"✓ 训练集: {len(self.data)} 个样本")
            else:
                self.data = npz_files[split_idx:]
                print(f"✓ 测试集: {len(self.data)} 个样本")

    @staticmethod
    def _resolve_path(path_str):
        """将相对路径转换为绝对路径"""
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path.resolve())

    def __getitem__(self, idx):
        """
        获取单个样本

        返回:
            tuple: (input, target)
                - input: (4, 256, 256) - 4帧输入
                - target: (15, 256, 256) - 15帧输出
                - 数值范围: [-9, 60]
        """
        # 加载数据
        if self.single_file_mode:
            input_target = np.load(self.data_path)
        else:
            npz_file = self.data[idx]
            input_target = np.load(npz_file)

        # 提取VIL数据: (384, 384, 49)
        vil = input_target["vil"][:, :, 0:38:2]  # 提取19帧

        # 转换为张量: (H,W,T) -> (T,H,W)
        vil = torch.from_numpy(vil).contiguous().float().permute(2, 0, 1)

        # 缩放到256x256
        vil = F.interpolate(
            vil.unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 数值范围映射: [0, 254] -> [-9, 60]
        vil = vil / 254.0
        vil = vil * (60 - (-9)) + (-9)

        # 拆分为输入和目标
        input_data = vil[0:4, :, :]  # 前4帧
        target = vil[4:19, :, :]  # 后15帧

        return input_data, target

    def __len__(self):
        """返回数据集样本总数"""
        return len(self.data)


def load_data(batch_size, val_batch_size, data_root, num_workers):
    """
    创建训练、验证和测试数据加载器

    参数:
        batch_size (int): 训练集批次大小
        val_batch_size (int): 验证/测试集批次大小
        data_root (str): 数据根目录路径
        num_workers (int): 数据加载的并行工作进程数

    返回:
        tuple: (dataloader_train, dataloader_validation, dataloader_test)
    """
    train_set = Sevir(train=True, data_dir=data_root if data_root else None)
    test_set = Sevir(train=False, data_dir=data_root if data_root else None)

    dataloader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

    dataloader_validation = torch.utils.data.DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    return dataloader_train, dataloader_validation, dataloader_test


def load_single_file(file_path, batch_size=1, num_workers=0):
    """
    加载单个.npz数据文件用于测试

    参数:
        file_path (str): .npz文件路径
        batch_size (int): 批次大小
        num_workers (int): 并行工作进程数

    返回:
        DataLoader: 单文件数据加载器
    """
    dataset = Sevir(train=True, data_path=file_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    return dataloader
