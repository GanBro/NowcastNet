# from collections import namedtuple
# import os
# import numpy as np
# import pandas as pd
# import dask.array as da
# from tqdm import tqdm
# from collections import namedtuple
# from torchvision import transforms
# import numba
#
# from DGMR.utils.dataset import RainDataset
# from DGMR.utils.data_parser import *
#
# import warnings
# warnings.filterwarnings("ignore")
# import time
#
# @numba.jit()
# def count_nonzeros(memmap):
#     rain_record = []
#     for map in memmap:
#         rain_record.append((map>0).sum())
#     return np.array(rain_record)
#
# def get_memmap(npy_path, dtype=None, shape=None, get_nonzeros=True):
#     if isinstance(npy_path, list):
#         memmap_list = []
#         rain_record = []
#         os.system("echo")
#
#         for npy, sp in zip(npy_path, shape):
#             memmap = np.memmap(npy, dtype=dtype, shape=tuple(sp), order='C')
#             memmap_list.append(memmap)
#             if get_nonzeros:
#                 os.system("echo Counting nonzeros may take a while...")
#                 start_time = time.time()
#                 rain_record.append(count_nonzeros(memmap))
#                 total_time = int(time.time() - start_time)
#                 os.system(f"echo Finish counting! {total_time} seconds are taken.")
#
#         rain_record = np.concatenate(rain_record, axis=0) if get_nonzeros else None
#         final_memmap = da.concatenate(memmap_list, axis=0)
#         os.system(f"echo Total length of memory-map : {final_memmap.shape[0]}")
#
#         return final_memmap, rain_record
#
#     elif isinstance(npy_path, str):
#         memmap = np.memmap(npy_path, dtype=dtype, shape=tuple(shape))
#
#         if get_nonzeros:
#             os.system("echo Counting nonzeros may take a while...")
#             start_time = time.time()
#             rain_record = count_nonzeros(memmap)
#             total_time = int(time.time() - start_time)
#             os.system(f"echo Finish counting! {total_time} seconds are taken.")
#         else:
#             rain_record = None
#         os.system(f"echo Total length of memory-map : {memmap.shape[0]}")
#
#         return memmap, rain_record
#
# def save_rain_record(rain_record, npy_path):
#     rain_record_df = pd.DataFrame(rain_record, columns=["nonzeros"])
#     if isinstance(npy_path, str):
#         label = npy_path.split("/")[-1].split(".")[0]
#     elif isinstance(npy_path, list):
#         label = ""
#         for npy in npy_path:
#             label += npy.split("/")[-1].split(".")[0]
#             label += "_"
#         label = label[:-1]
#     rain_record_df.to_csv(f"rain_record_{label}.csv", index=False)
#     return
#
# def get_arguments(
#     params: namedtuple=None
#     ):
#     args = {}
#     for k in params._asdict().keys():
#         args[k.lower()] = params._asdict()[k]
#     return args
#
# def make_dataset(
#     cfg,
#     mode: str=None
#     ):
#     npy_path = cfg.SETTINGS.DATA_PATH
#     memory_map, rain_record = get_memmap(
#         npy_path,
#         dtype=getattr(np, cfg.SETTINGS.DATA_TYPE),
#         shape=cfg.SETTINGS.DATA_SHAPE,
#         get_nonzeros=True if cfg.SETTINGS.RAIN_RECORD_PATH is None else False
#         )
#
#     if cfg.SETTINGS.RAIN_RECORD_PATH is not None:
#         rain_record = pd.read_csv(cfg.SETTINGS.RAIN_RECORD_PATH)
#     else:
#         save_rain_record(rain_record=rain_record, npy_path=npy_path)
#
#     parser = None
#     if cfg.PARAMS.PARSER.FUNCTION is not None:
#         parser_name = cfg.PARAMS.PARSER.FUNCTION
#         parser_args = get_arguments(cfg.PARAMS.PARSER.PARAMS)
#         parser = lambda x: eval(parser_name)(x, **parser_args)
#         os.system(f"echo Parser: {parser_name}")
#
#     normalizer = None
#     if cfg.PARAMS.NORMALIZER.FUNCTION is not None:
#         normalizer_name = cfg.PARAMS.NORMALIZER.FUNCTION
#         normalizer_args = get_arguments(cfg.PARAMS.NORMALIZER.PARAMS)
#         normalizer = lambda x: eval(normalizer_name)(x, **normalizer_args)
#         os.system(f"echo Normalizer: {normalizer_name}")
#
#     if mode == "train":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.RandomCrop(cfg.PARAMS.INPUT_SIZE)
#         ])
#     elif mode == "test" or mode == "val":
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.CenterCrop(cfg.PARAMS.INPUT_SIZE)
#         ])
#
#     dataset = RainDataset(
#         memory_map=memory_map,
#         rain_record=rain_record,
#         rain_coverage=cfg.PARAMS.COVERAGE,
#         in_step=cfg.PARAMS.INPUT_FRAME,
#         out_step=cfg.PARAMS.OUTPUT_FRAME,
#         parser=parser,
#         normalizer=normalizer,
#         transform=transform
#     )
#     return dataset
from collections import namedtuple
import os
import numpy as np
import pandas as pd
import dask.array as da
from tqdm import tqdm
from collections import namedtuple
from torchvision import transforms
import numba

from DGMR.utils.dataset import RainDataset
from DGMR.utils.data_parser import *

import warnings

warnings.filterwarnings("ignore")
import time

# ========== 添加SEVIR数据集导入 ==========
try:
    from mydataloader import Sevir

    SEVIR_AVAILABLE = True
except ImportError:
    SEVIR_AVAILABLE = False
    print("Warning: mydataloader not found, SEVIR dataset unavailable")


@numba.jit()
def count_nonzeros(memmap):
    rain_record = []
    for map in memmap:
        rain_record.append((map > 0).sum())
    return np.array(rain_record)


def get_memmap(npy_path, dtype=None, shape=None, get_nonzeros=True):
    if isinstance(npy_path, list):
        memmap_list = []
        rain_record = []
        os.system("echo")

        for npy, sp in zip(npy_path, shape):
            memmap = np.memmap(npy, dtype=dtype, shape=tuple(sp), order='C')
            memmap_list.append(memmap)
            if get_nonzeros:
                os.system("echo Counting nonzeros may take a while...")
                start_time = time.time()
                rain_record.append(count_nonzeros(memmap))
                total_time = int(time.time() - start_time)
                os.system(f"echo Finish counting! {total_time} seconds are taken.")

        rain_record = np.concatenate(rain_record, axis=0) if get_nonzeros else None
        final_memmap = da.concatenate(memmap_list, axis=0)
        os.system(f"echo Total length of memory-map : {final_memmap.shape[0]}")

        return final_memmap, rain_record

    elif isinstance(npy_path, str):
        memmap = np.memmap(npy_path, dtype=dtype, shape=tuple(shape))

        if get_nonzeros:
            os.system("echo Counting nonzeros may take a while...")
            start_time = time.time()
            rain_record = count_nonzeros(memmap)
            total_time = int(time.time() - start_time)
            os.system(f"echo Finish counting! {total_time} seconds are taken.")
        else:
            rain_record = None
        os.system(f"echo Total length of memory-map : {memmap.shape[0]}")

        return memmap, rain_record


def save_rain_record(rain_record, npy_path):
    rain_record_df = pd.DataFrame(rain_record, columns=["nonzeros"])
    if isinstance(npy_path, str):
        label = npy_path.split("/")[-1].split(".")[0]
    elif isinstance(npy_path, list):
        label = ""
        for npy in npy_path:
            label += npy.split("/")[-1].split(".")[0]
            label += "_"
        label = label[:-1]
    rain_record_df.to_csv(f"rain_record_{label}.csv", index=False)
    return


def get_arguments(
        params: namedtuple = None
):
    args = {}
    for k in params._asdict().keys():
        args[k.lower()] = params._asdict()[k]
    return args


def _check_data_files_exist(data_path):
    """检查数据文件是否存在"""
    if isinstance(data_path, list):
        for path in data_path:
            if not os.path.exists(path):
                return False
        return True
    elif isinstance(data_path, str):
        return os.path.exists(data_path)
    else:
        return False


def make_dataset(
        cfg,
        mode: str = None
):
    """
    根据配置创建数据集

    参数:
        cfg: 配置对象
        mode: "train", "val", 或 "test"

    返回:
        Dataset对象
    """
    # ========== 智能判断使用哪个数据集 ==========
    use_sevir = False

    # 判断条件1: 检查DATA_PATH中的文件是否存在
    if hasattr(cfg.SETTINGS, 'DATA_PATH'):
        if isinstance(cfg.SETTINGS.DATA_PATH, list) and len(cfg.SETTINGS.DATA_PATH) > 0:
            # 如果配置的Nimrod文件不存在，则使用SEVIR
            if not _check_data_files_exist(cfg.SETTINGS.DATA_PATH):
                print("警告: 配置的Nimrod数据文件不存在")
                use_sevir = True
        else:
            # DATA_PATH为空列表或None，使用SEVIR
            use_sevir = True
    else:
        use_sevir = True

    # 判断条件2: 显式指定使用SEVIR
    if hasattr(cfg.SETTINGS, 'DATA_TYPE'):
        if cfg.SETTINGS.DATA_TYPE in ['sevir', 'SEVIR', 'float32']:
            use_sevir = True

    # ========== 使用SEVIR数据集 ==========
    if use_sevir:
        if not SEVIR_AVAILABLE:
            raise ImportError(
                "SEVIR数据集不可用。请确保:\n"
                "1. mydataloader.py 文件存在于项目根目录\n"
                "2. SEVIR数据文件路径正确配置"
            )

        print("=" * 60)
        print("使用 SEVIR 数据集")
        print(f"模式: {mode}")
        print(f"输入帧数: 4")
        print(f"输出帧数: 15")
        print(f"图像尺寸: 256x256")
        print("=" * 60)

        if mode == "train":
            return Sevir(train=True, data_dir="/mnt/hdd1/liudufu/project/datasets/event_sevir")
        elif mode in ["val", "test"]:
            return Sevir(train=False, data_dir="/mnt/hdd1/liudufu/project/datasets/event_sevir")
        else:
            raise ValueError(f"未知的模式: {mode}")

        # if mode == "train":
        #     return Sevir(train=True)
        # elif mode in ["val", "test"]:
        #     return Sevir(train=False)
        # else:
        #     raise ValueError(f"未知的模式: {mode}")

    # ========== 使用原始Nimrod数据集 ==========
    print("=" * 60)
    print("使用 Nimrod 数据集")
    print(f"模式: {mode}")
    print("=" * 60)

    npy_path = cfg.SETTINGS.DATA_PATH
    memory_map, rain_record = get_memmap(
        npy_path,
        dtype=getattr(np, cfg.SETTINGS.DATA_TYPE),
        shape=cfg.SETTINGS.DATA_SHAPE,
        get_nonzeros=True if cfg.SETTINGS.RAIN_RECORD_PATH is None else False
    )

    if cfg.SETTINGS.RAIN_RECORD_PATH is not None:
        rain_record = pd.read_csv(cfg.SETTINGS.RAIN_RECORD_PATH)
    else:
        save_rain_record(rain_record=rain_record, npy_path=npy_path)

    parser = None
    if cfg.PARAMS.PARSER.FUNCTION is not None:
        parser_name = cfg.PARAMS.PARSER.FUNCTION
        parser_args = get_arguments(cfg.PARAMS.PARSER.PARAMS)
        parser = lambda x: eval(parser_name)(x, **parser_args)
        os.system(f"echo Parser: {parser_name}")

    normalizer = None
    if cfg.PARAMS.NORMALIZER.FUNCTION is not None:
        normalizer_name = cfg.PARAMS.NORMALIZER.FUNCTION
        normalizer_args = get_arguments(cfg.PARAMS.NORMALIZER.PARAMS)
        normalizer = lambda x: eval(normalizer_name)(x, **normalizer_args)
        os.system(f"echo Normalizer: {normalizer_name}")

    if mode == "train":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(cfg.PARAMS.INPUT_SIZE)
        ])
    elif mode == "test" or mode == "val":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(cfg.PARAMS.INPUT_SIZE)
        ])

    dataset = RainDataset(
        memory_map=memory_map,
        rain_record=rain_record,
        rain_coverage=cfg.PARAMS.COVERAGE,
        in_step=cfg.PARAMS.INPUT_FRAME,
        out_step=cfg.PARAMS.OUTPUT_FRAME,
        parser=parser,
        normalizer=normalizer,
        transform=transform
    )
    return dataset
