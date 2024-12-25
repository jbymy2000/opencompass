import csv
import json
import os.path as osp
from os import environ

from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class RPBenchCharactorDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)
        dataset = {}
        filename = osp.join(path, f'{name}.jsonl')
        data = []
        # 打开 JSONL 文件并逐行读取
        with open(filename, encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())  # 加载每行 JSON 数据
                    data.append({
                        "id": item.get("id", ""),  # 获取 ID，默认为空
                        "background": item.get("background", ""),
                        "npc_profile": item.get("npc_profile", {}),
                        "conversation": item.get("conversation", [])
                    })
                except json.JSONDecodeError as e:
                    print(f"跳过无法解析的行: {line}. 错误: {e}")
        
        # 使用 Dataset 和 DatasetDict 构建数据集
        dataset = Dataset.from_list(data)
        
        return dataset