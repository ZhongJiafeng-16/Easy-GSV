# -*- coding: utf-8 -*-
import os
import torch
import click
import librosa
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from tqdm import tqdm
from loguru import logger
from pathlib import Path
from modules.cnhubert import CNHubert
from modules.sovits import SynthesizerTrn
from utils import load_hparams, RuntimeTracker

CUHUBERT_CKPT_PATH = "/home/zhongjiafeng/repo/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"
CONFIG_PATH = "config/s2.json"
QUANTIZER_CKPT_PATH = "/home/zhongjiafeng/repo/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"


@click.command()
@click.option("--inp_text", type=str, default="/home/zhongjiafeng/repo/Easy-GSV/data/EN_B0000567-wav2phone.txt")
@click.option("--output_dir", type=str, default="/data/zhongjiafeng/gsv")
@click.option("--exp_name", type=str, default="EN_B0000567")
@click.option("--save_hubert", type=bool, default=False)
def main(
    inp_text,
    output_dir,
    exp_name,
    save_hubert,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hubert_dir = f"{output_dir}/2-hubert"
    semantic_dir = f"{output_dir}/3-semantic"
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)

    tracker = RuntimeTracker()
    tracker.start("Load Hubert Model")
    # load model
    hubert_model = CNHubert(CUHUBERT_CKPT_PATH)
    hubert_model = hubert_model.to(device)
    tracker.end()

    tracker.start("Load Quantizer Model")
    # load quantizer
    hps = load_hparams(CONFIG_PATH)
    quantizer = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    quantizer.load_state_dict(torch.load(QUANTIZER_CKPT_PATH, map_location=device)["weight"], strict=False)
    quantizer.to(device)
    quantizer.eval()
    tracker.end()

    with open(inp_text, "r", encoding='utf-8') as f:
        lines = f.readlines()

    tracker.start("Extract Codec")
    datas = [l.strip().split('|')[0] for l in lines]

    for wav_path in tqdm(datas, desc="Extract Semantic", mininterval=500):
        wav_id = Path(wav_path).stem
        hubert_path = f"{hubert_dir}/{wav_id}.pt"
        semantic_path = f"{semantic_dir}/{wav_id}_semantic.txt"
        if not os.path.exists(semantic_path):
            try:
                temp_audio_16k, _ = librosa.load(wav_path, sr=16000)
                temp_audio_16k = torch.from_numpy(temp_audio_16k)
                temp_audio_16k = temp_audio_16k.to(device)
                hubert_feature = hubert_model.model(temp_audio_16k.unsqueeze(0))["last_hidden_state"].transpose(1,2)
                codes = quantizer.extract_latent(hubert_feature)
                semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
                with open(semantic_path, 'w', encoding='utf-8') as f:
                    f.write(semantic)
                if save_hubert:
                    torch.save(hubert_feature.cpu(), hubert_path)
            except Exception as e:
                logger.warning(f"Error in extract hubert {wav_path}, skip. {e}")
                continue
            
    tracker.end()
    tracker.print_all_records()    
        
if __name__ == '__main__':
    main()
