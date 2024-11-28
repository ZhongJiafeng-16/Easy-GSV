import click
import os 
import torch
import json
import multiprocessing
from loguru import logger
from pathlib import Path
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from text.cleaner import english_clean_text
from utils.tracker import RuntimeTracker

def load_audio_from_json(json_files):
    if isinstance(json_files, str):
        json_files = [json_files]

    datas = []
    for json_file in json_files:
        json_file = Path(json_file)
        audio_root = json_file.parent / json_file.stem
        json_data = []
        data = []
        with open(json_file, "r", encoding='utf-8') as f:
            for i in f.readlines():
                line = json.loads(i)
                json_data.append(line)

        for item in json_data:
            wav_path = f"{audio_root}/{item['id']}.mp3"

            if not os.path.exists(wav_path):
                logger.warning(f"{item['id']} not in {audio_root}, skip.")
                continue
                
            language = item["language"]
            text = item["text"]
            data.append([wav_path, language, text])

        datas.extend(data)
        logger.info(f"Load {json_file} with {len(data)} audio file.")

    logger.info(f"Load {len(datas)} audio files.")
    return datas


def process_text(data):
    wav_path, language, text = data
    wav_id = Path(wav_path).stem
    text = text.strip()
    # get phones, word2ph, norm_text
    try:
        phones, word2ph, norm_text = english_clean_text(text, language)
    except Exception as e:
        logger.warning(f"Error in process {wav_id} text, skip. {e}")
        return None

    phones = " ".join(phones)
    res = [wav_path, language, phones, word2ph, norm_text]
    return res

@click.command()
@click.option("--inp_text", type=str, multiple=True, default=[
    "/data/zhongjiafeng/Emilia/EN_B00005.jsonl",
    "/data/zhongjiafeng/Emilia/EN_B00006.jsonl",
    "/data/zhongjiafeng/Emilia/EN_B00007.jsonl",
    ])
@click.option("--output_dir", type=str, default="/data/zhongjiafeng/gsv")
@click.option("--num_worker", type=int, default=1)
@click.option("--exp_name", type=str, default="EN_B0000567")
def main(
    inp_text,
    output_dir,
    num_worker,
    exp_name,
):
    tracker = RuntimeTracker()
    tracker.start("Text Process")

    # load wav files
    datas = load_audio_from_json(inp_text)

    total_res = []
    if num_worker > 1:     # more slow than single process, dn not know why
        with multiprocessing.Pool(processes=num_worker) as executor:
            for res in tqdm(
                executor.imap_unordered(process_text, datas, chunksize=5000),
                total=len(datas),
                desc="Text Process",
            ):
                if res:
                    total_res.append(res)
    else:
        for data in tqdm(datas, desc="Text Process"):
            res = process_text(data)
            if res is not None:
                total_res.append(res)
    
    tracker.end()
    tracker.print_all_records()

    # save
    save_path = f"{output_dir}/{exp_name}-wav2phone.txt"
    with open(save_path, "w", encoding='utf-8') as f:
        for wav_path, language, phones, _, norm_text in total_res:
            f.write(f"{wav_path}|{language}|{phones}|{norm_text}\n")
    logger.info(f"Save to {save_path}")
    
if __name__ == "__main__":
    main()