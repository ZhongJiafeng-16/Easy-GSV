import os
import torch
import torch.utils.data
import torch.nn.functional as F
import lightning as L
import librosa

from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from modules.sovits.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from .bucket_sampler import DistributedBucketSampler
from loguru import logger

class TextAudioSpeakerDataModule(L.LightningDataModule):
    def __init__(
        self,
        config,
        train_wav2phone_file,
        train_hubert_dir,
    ):
        super().__init__()
        self.config = config
        self.train_wav2phone_file = train_wav2phone_file
        self.train_hubert_dir = train_hubert_dir

    def prepare_data(self):
        pass

    def setup(self, stage=None, output_logs=False):
        self._train_dataset = TextAudioSpeakerDataset(
            wav2phone_file=self.train_wav2phone_file,
            hubert_dir=self.train_hubert_dir,
            max_wav_value=self.config.data.max_wav_value,
            sampling_rate=self.config.data.sampling_rate,
            filter_length=self.config.data.filter_length,
            hop_length=self.config.data.hop_length,
            win_length=self.config.data.win_length,
        )

    def train_dataloader(self):
        batch_size=self.config["train"]["batch_size"]
        sampler = DistributedBucketSampler(self._train_dataset, batch_size=batch_size, 
                bucket_width=100)
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._train_dataset.collate,
            num_workers=self.config.data.num_workers,
            persistent_workers=True,
            prefetch_factor=16,
        )

class TextAudioSpeakerDataset(Dataset):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(self, 
        wav2phone_file,
        hubert_dir,
        max_wav_value,
        sampling_rate,
        filter_length,
        hop_length,
        win_length,
    ):
        self.max_wav_value = max_wav_value
        self.sampling_rate = sampling_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        self.wav2phone = wav2phone_file
        self.hubert_dir = hubert_dir
        assert os.path.exists(self.hubert_dir)
        assert os.path.exists(self.wav2phone)

        self.datas = []
        self.lengths = []
        with open(self.wav2phone, "r", encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            wav_path, language, phoneme, text = line.split('|')
            wav_id = Path(wav_path).stem
            hubert_path = f"{self.hubert_dir}/{wav_id}.pt"

            if not os.path.exists(hubert_path) or not os.path.exists(wav_path):
                logger.warning(f"Audio file or Hubert feature not exists, skip {wav_id}.")
                continue

            phoneme = phoneme.split(' ')
            phoneme_ids = cleaned_text_to_sequence(phoneme)
            duration = os.path.getsize(wav_path) / self.sampling_rate / 2

            if duration == 0 or duration > 54:
                logger.warning(f"Zero duration or too long duration, skip {wav_id}")
                continue
            
            self.lengths.append(duration * self.sampling_rate // self.hop_length)
            self.datas.append([wav_path, hubert_path, phoneme_ids])
        
        logger.info(f"Total {len(self.datas)} audio sample.")

    def get_audio_text_speaker_pair(self, data):
        wav_path, hubert_path,phoneme_ids = data
        phoneme_ids = torch.FloatTensor(phoneme_ids)
        spec, wav = self.get_audio(wav_path)
        ssl = torch.load(hubert_path, map_location="cpu")
        if (ssl.shape[-1] != spec.shape[-1]):
            ssl = F.pad(ssl.float(), (0, 1), mode="replicate").to(ssl.dtype)
            logger.warning(f"SSL feature length <{ssl.shape}> not match spec length <{spec.shape}>, pad it.")
        return {
            "ssl": ssl,
            "spec": spec,
            "wav": wav,
            "phoneme_ids": phoneme_ids,
        }

    def get_audio(self, filename):
        audio_array, sr = librosa.load(filename, sr=self.sampling_rate)
        audio = torch.FloatTensor(audio_array).unsqueeze(0)
        spec = spectrogram_torch(audio, self.filter_length, self.sampling_rate, self.hop_length, 
                                 self.win_length, center=False)
        spec = spec.squeeze(0)
        return spec, audio

    def __getitem__(self, index):
        # with torch.no_grad():
        return self.get_audio_text_speaker_pair(self.datas[index])

    def __len__(self):
        return len(self.datas)

    def get_sample_length(self, index):
        return self.lengths[index]

    def collate(self, examples: List[Dict]) -> Dict:
        # TODO: 需要测试
        ssl_list: List[torch.Tensor] = []
        spec_list: List[torch.Tensor] = []
        wav_list: List[torch.Tensor] = []
        phoneme_ids_list: List[torch.Tensor] = []
        batch_size = len(examples)

        for item in examples:
            ssl_list.append(item["ssl"])
            spec_list.append(item["spec"])
            wav_list.append(item["wav"])
            phoneme_ids_list.append(item["phoneme_ids"])

        max_ssl_len = max([x.size(-1) for x in ssl_list])
        max_spec_len = max([x.size(-1) for x in spec_list])
        max_wav_len = max([x.size(-1) for x in wav_list])
        max_phoneme_len = max([x.size(0) for x in phoneme_ids_list])

        ssl_lengths = torch.LongTensor(batch_size)
        spec_lengths = torch.LongTensor(batch_size)
        wav_lengths = torch.LongTensor(batch_size)
        phoneme_lengths = torch.LongTensor(batch_size)

        spec_padded = torch.FloatTensor(batch_size, ssl_list[0].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(batch_size, 1, max_wav_len)
        ssl_padded = torch.FloatTensor(batch_size, spec_list[0].size(0), max_ssl_len)
        phoneme_padded = torch.LongTensor(batch_size, max_phoneme_len)

        spec_padded.zero_()
        wav_padded.zero_()
        ssl_padded.zero_()
        phoneme_padded.zero_()

        for i in range(batch_size):
            ssl = ssl_list[i]
            spec = spec_list[i]
            wav = wav_list[i]
            phoneme_ids = phoneme_ids_list[i]

            ssl_padded[i, :, :ssl.size(2)] = ssl
            spec_padded[i, :, :spec.size(1)] = spec
            wav_padded[i, :, :wav.size(1)] = wav
            phoneme_padded[i, :phoneme_ids.size(0)] = phoneme_ids
            
        return {
            # List[int]
            "ssl": ssl_padded,
            "ssl_len": ssl_lengths,
            # torch.Tensor (B, max_phoneme_length)
            "spec": spec_padded,
            "spec_len": spec_lengths,
            # torch.Tensor (B)
            "wav": wav_padded,
            "wav": wav_lengths,
            # torch.Tensor (B, max_semantic_ids_length)
            "phoneme": phoneme_padded,
            "phoneme_len": phoneme_lengths,
        }
