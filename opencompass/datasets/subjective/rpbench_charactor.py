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


class CEvalDatasetClean(BaseDataset):

    # load the contamination annotations of CEval from
    # https://github.com/liyucheng09/Contamination_Detector
    @staticmethod
    def load_contamination_annotations(path, split='val'):
        import requests

        assert split == 'val', 'Now we only have annotations for val set'
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope.utils.config_ds import MS_DATASETS_CACHE
            annotation_cache_path = osp.join(
                MS_DATASETS_CACHE, 'ceval_contamination_annotations.json')
            link_of_annotations = 'https://modelscope.cn/datasets/opencompass/Contamination_Detector/resolve/master/ceval_annotations.json'  # noqa
        else:
            annotation_cache_path = osp.join(
                path, split, 'ceval_contamination_annotations.json')
            link_of_annotations = 'https://github.com/liyucheng09/Contamination_Detector/releases/download/v0.1.1rc/ceval_annotations.json'  # noqa

        if osp.exists(annotation_cache_path):
            with open(annotation_cache_path, 'r') as f:
                annotations = json.load(f)
            return annotations
        annotations = json.loads(requests.get(link_of_annotations).text)
        with open(annotation_cache_path, 'w') as f:
            json.dump(annotations, f)
        return annotations

    @staticmethod
    def load(path: str, name: str):
        path = get_data_path(path)
        dataset = {}
        if environ.get('DATASET_SOURCE') == 'ModelScope':
            from modelscope import MsDataset
            dataset = MsDataset.load(dataset_name=path, subset_name=name)
            # 向该数据添加 'is_clean' 字段
            annotations = CEvalDatasetClean.load_contamination_annotations(
                path, 'val')
            val = dataset['val']
            val_data = []
            for index in range(val.num_rows):
                row = val[index]
                row_id = f'{name}-{index}'
                row.update({
                    'is_clean':
                    annotations[row_id][0]
                    if row_id in annotations else 'not labeled'
                })
                val_data.append(row)
            dataset['val'] = Dataset.from_list(val_data)
        else:
            for split in ['dev', 'val', 'test']:
                if split == 'val':
                    annotations = \
                        CEvalDatasetClean.load_contamination_annotations(
                            path, split)
                filename = osp.join(path, split, f'{name}_{split}.csv')
                with open(filename, encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    for row_index, row in enumerate(reader):
                        item = dict(zip(header, row))
                        item.setdefault('explanation', '')
                        item.setdefault('answer', '')
                        if split == 'val':
                            row_id = f'{name}-{row_index}'
                            if row_id in annotations:
                                item['is_clean'] = annotations[row_id][0]
                            else:
                                item['is_clean'] = 'not labeled'
                        dataset.setdefault(split, []).append(item)
            dataset = DatasetDict(
                {i: Dataset.from_list(dataset[i])
                 for i in dataset})
        return dataset
