import os
import click
from pathlib import Path
from tqdm import tqdm
from loguru import logger

@click.command()
@click.option("--root", type=str, default="data/3-semantic")
@click.option("--target", type=str, default="data/EN_B0000567-semantic.txt")
def merge_txt_file(root, target):
    semantic_file_list = Path(root).glob("**/*_semantic.txt")
    datas = []
    for semantic_file in tqdm(list(semantic_file_list)):
        wav_id = semantic_file.stem.replace("_semantic","")
        semantic = semantic_file.read_text()
        datas.append([wav_id, semantic])

    with open(target, "w", encoding='utf-8') as f:
        for wav_id, semantic in datas:
            f.write(f"{wav_id}|{semantic}\n")
    
    logger.info(f"Merge all semantic file in to single one {target}")

if __name__ == '__main__':
    merge_txt_file()