import lightning as L
import os
import torch
import numpy as np

from pathlib import Path
from loguru import logger
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from .bucket_sampler import DistributedBucketSampler
from utils import load_semantic, load_phones, batch_sequences
from text import cleaned_text_to_sequence

class Text2SemanticDataModule(L.LightningDataModule):
    def __init__(
        self,
        config,
        train_semantic_path,
        train_phoneme_path,
        dev_semantic_path=None,
        dev_phoneme_path=None,
    ):
        super().__init__()
        self.config = config
        self.train_semantic_path = train_semantic_path
        self.train_phoneme_path = train_phoneme_path
        self.dev_semantic_path = dev_semantic_path
        self.dev_phoneme_path = dev_phoneme_path
        self.num_workers = self.config["data"]["num_workers"]

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        self._train_dataset = Text2SemanticDataset(
            phoneme_path=self.train_phoneme_path,
            semantic_path=self.train_semantic_path,
            max_sec=self.config["data"]["max_sec"],
            pad_val=self.config["data"]["pad_val"],
        )
        self._dev_dataset = self._train_dataset
        # self._dev_dataset = Text2SemanticDataset(
        #     phoneme_path=self.dev_phoneme_path,
        #     semantic_path=self.dev_semantic_path,
        #     max_sample=self.config['data']['max_eval_sample'],
        #     max_sec=self.config['data']['max_sec'],
        #     pad_val=self.config['data']['pad_val'])

    def train_dataloader(self):
        batch_size=self.config["train"]["batch_size"]//2 if self.config["train"].get("if_dpo",False) else self.config["train"]["batch_size"]
        sampler = DistributedBucketSampler(self._train_dataset, batch_size=batch_size)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=self.num_workers,
            persistent_workers=True,
            prefetch_factor=16,
        )

    def val_dataloader(self):
        return DataLoader(
            self._dev_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=self._train_dataset.collate,
            num_workers=max(self.num_workers, 12),
            persistent_workers=True,
            prefetch_factor=16,
        )

class Text2SemanticDataset(Dataset):
    """dataset class for text tokens to semantic model training."""

    def __init__(
        self,
        phoneme_path: str,
        semantic_path: str,
        max_sample: int = None,
        max_sec: int = 100,
        pad_val: int = 1024,
        # min value of phoneme/sec
        min_ps_ratio: int = 3,
        # max value of phoneme/sec
        max_ps_ratio: int = 25,
    ) -> None:
        super().__init__()

        # get dict
        self.phoneme_path = phoneme_path
        self.semantic_path = semantic_path
        self.bert_dir = os.path.join(os.path.dirname(self.semantic_path), "1-bert")
        assert os.path.exists(self.phoneme_path)
        assert os.path.exists(self.semantic_path)

        # load semantic feature (ssl feature)
        self.semantic_data = load_semantic(self.semantic_path)
        self.phoneme_data = load_phones(self.phoneme_path)

        # pad for semantic tokens
        self.PAD = pad_val
        self.hz = int(os.environ.get("hz", "25hz")[:-2])

        # max seconds of semantic token
        self.max_sec = max_sec
        self.min_ps_ratio = min_ps_ratio
        self.max_ps_ratio = max_ps_ratio

        if max_sample is not None:
            self.semantic_data = self.semantic_data[:max_sample]

        self.item_names, self.semantic_phoneme = self.init_batch()
        del self.semantic_data
        del self.phoneme_data

    def init_batch(self):
        items_list = self.phoneme_data.keys()
        num_deleted_bigger = 0
        num_deleted_ps = 0

        semantic_phoneme = []
        item_names = []
        for idx, item_name in enumerate(items_list):
            phoneme_str, text = self.phoneme_data[item_name]
            semantic_str = self.semantic_data[item_name]

            # get token list
            semantic_ids = [int(i) for i in semantic_str.split(" ")]
           
            if (len(semantic_ids) > self.max_sec * self.hz):  
                num_deleted_bigger += 1
                continue
            
            phonemes = phoneme_str.split(" ")
            phoneme_ids = cleaned_text_to_sequence(phonemes)
    
            if (len(phoneme_ids) > self.max_sec * self.hz / 2.5):  
                num_deleted_ps += 1
                continue

            ps_ratio = len(phoneme_ids) / (len(semantic_ids) / self.hz)
            if (ps_ratio > self.max_ps_ratio or ps_ratio < self.min_ps_ratio):  
                num_deleted_ps += 1
                continue

            semantic_phoneme.append((semantic_ids, phoneme_ids))
            item_names.append(item_name)
       
        if num_deleted_bigger > 0:
            logger.info(
                f"Deleted {num_deleted_bigger} audios who's duration are bigger than {self.max_sec} seconds"
            )
        if num_deleted_ps > 0:
            logger.info(
                f"deleted {num_deleted_ps} audios who's phoneme/sec are bigger than {self.max_ps_ratio} or smaller than {self.min_ps_ratio}"
            )

        assert len(item_names) == len(semantic_phoneme)
        logger.info(f"Total sample number {len(item_names)}")
        return item_names, semantic_phoneme

    def __get_item_names__(self) -> List[str]:
        return self.item_names

    def __len__(self) -> int:
        return len(self.semantic_phoneme)

    def __getitem__(self, idx: int) -> Dict:
        semantic_ids, phoneme_ids = self.semantic_phoneme[idx]
        item_name = self.item_names[idx]
        phoneme_ids_len = len(phoneme_ids)
        # semantic tokens target
        semantic_ids_len = len(semantic_ids)

        bert_path = f"{self.bert_dir}/{item_name}.pt"
        if os.path.exists(bert_path):
            bert_feature = torch.load(bert_path, map_location="cpu")
            assert bert_feature.shape[-1] == len(phoneme_ids)
        else:
            # bert_feature=torch.zeros_like(phoneme_ids,dtype=torch.float32)
            bert_feature = None

        return {
            "idx": idx,
            "phoneme_ids": phoneme_ids,
            "phoneme_ids_len": phoneme_ids_len,
            "semantic_ids": semantic_ids,
            "semantic_ids_len": semantic_ids_len,
            "bert_feature": bert_feature,
        }

    def get_sample_length(self, idx: int):
        semantic_ids = self.semantic_phoneme[idx][0]
        sec = 1.0 * len(semantic_ids) / self.hz
        return sec

    def collate(self, examples: List[Dict]) -> Dict:
        sample_index: List[int] = []
        phoneme_ids: List[torch.Tensor] = []
        phoneme_ids_lens: List[int] = []
        semantic_ids: List[torch.Tensor] = []
        semantic_ids_lens: List[int] = []
        # return

        for item in examples:
            sample_index.append(item["idx"])
            phoneme_ids.append(np.array(item["phoneme_ids"], dtype=np.int64))
            semantic_ids.append(np.array(item["semantic_ids"], dtype=np.int64))
            phoneme_ids_lens.append(item["phoneme_ids_len"])
            semantic_ids_lens.append(item["semantic_ids_len"])

        # pad 0
        phoneme_ids = batch_sequences(phoneme_ids)
        semantic_ids = batch_sequences(semantic_ids, pad_val=self.PAD)

        # # convert each batch to torch.tensor
        phoneme_ids = torch.tensor(phoneme_ids)
        semantic_ids = torch.tensor(semantic_ids)
        phoneme_ids_lens = torch.tensor(phoneme_ids_lens)
        semantic_ids_lens = torch.tensor(semantic_ids_lens)
        bert_padded = torch.FloatTensor(len(examples), 1024, max(phoneme_ids_lens))
        bert_padded.zero_()

        for idx, item in enumerate(examples):
            bert = item["bert_feature"]
            if bert != None:
                bert_padded[idx, :, : bert.shape[-1]] = bert

        return {
            # List[int]
            "ids": sample_index,
            # torch.Tensor (B, max_phoneme_length)
            "phoneme_ids": phoneme_ids,
            # torch.Tensor (B)
            "phoneme_ids_len": phoneme_ids_lens,
            # torch.Tensor (B, max_semantic_ids_length)
            "semantic_ids": semantic_ids,
            # torch.Tensor (B)
            "semantic_ids_len": semantic_ids_lens,
            # torch.Tensor (B, 1024, max_phoneme_length)
            "bert_feature": bert_padded,
        }