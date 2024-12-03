import os
os.environ["GRADIO_TEMP_DIR"] = "./TEMP"
import torch
import gradio as gr
import numpy as np
import re
import librosa
import LangSegment

import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from utils import HParams, RuntimeTracker, remove_parameter_prefix
from loguru import logger
from glob import glob
from modules.sovits import SynthesizerTrn, spectrogram_torch
from modules.gpt import Text2SemanticDecoder
from modules.cnhubert import CNHubert
from text.cleaner import clean_text, cleaned_text_to_sequence
from faster_whisper import WhisperModel

g_gpt_weight_root = ["/home/zhongjiafeng/repo/GPT-SoVITS/GPT_SoVITS/pretrained_models","/home/zhongjiafeng/repo/Easy-GSV/pretrain"]
g_sovits_wigth_root = ["/home/zhongjiafeng/repo/GPT-SoVITS/GPT_SoVITS/pretrained_models"]

g_ssl_model = None
g_sovits_model = None
g_gpt_model = None
g_asr_model = None

g_gpt_config = None
g_sovits_config = None

SUPPORT_LANGUAGE = ["en"]
CUHUBERT_CKPT_PATH = "/home/zhongjiafeng/repo/GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base"
ASR_CKPT_PATH = "/home/zhongjiafeng/repo/Lychee/weigth/whsiper"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_weights_names():
    SoVITS_names = []
    GPT_names = []
    for path in g_sovits_wigth_root:
        SoVITS_names.extend(glob(f"{path}/**/*.pth", recursive=True))
    for path in g_gpt_weight_root:
        GPT_names.extend(glob(f"{path}/**/*.ckpt", recursive=True))
    return SoVITS_names, GPT_names

def get_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=g_sovits_config.data.sampling_rate)
    audio = torch.FloatTensor(audio).unsqueeze(0)
    spec = spectrogram_torch(
        audio,
        g_sovits_config.data.filter_length,
        g_sovits_config.data.sampling_rate,
        g_sovits_config.data.hop_length,
        g_sovits_config.data.win_length,
        center=False,
    )
    return spec

def custom_sort_key(s):
    parts = re.split('(\d+)', s)
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def punctuation_cut(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)

def check_text(texts):
    _text=[]
    if all(text in [None, " ", "\n",""] for text in texts):
        raise ValueError("Found unvalid text!")
    for text in texts:
        if text in  [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text

def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

def get_phones_and_bert(text, language, final=False):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_","")
        LangSegment.setfilters(["en"])
        formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text(formattext, language)
        phones = cleaned_text_to_sequence(phones)
        bert = torch.zeros((1024, len(phones)), dtype=torch.float32).to(device)
    
    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, final=True)

    return phones, bert, norm_text

def get_tts_wav(
        ref_wav_path, 
        prompt_text, 
        prompt_language, 
        intput_text, 
        text_language, 
        top_k=20, 
        top_p=0.6, 
        temperature=0.6, 
        ref_free=False,
        speed=1,
    ):
    if not ref_wav_path:
        gr.Warning("Please upload a reference audio!")
    if not intput_text:
        gr.Warning("Please input a text!")

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        # if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        logger.info(f"Processed Reference Text: <{prompt_text}>")

    sample_rate = g_sovits_config.data.sampling_rate
    intput_text = intput_text.strip("\n")
    logger.info(f"Processed Input Text: <{intput_text}>")

    zero_wav = np.zeros(int(sample_rate * 0.3), dtype=np.float32)

    if not ref_free: 
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])  # 在后面添加静音片段
            ssl_content = g_ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
            codes = g_sovits_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)


    intput_text = punctuation_cut(intput_text)
    logger.info(f"Input Text after cut: {intput_text}")
    intput_texts = intput_text.split("\n")
    intput_texts = check_text(intput_texts)
    intput_texts = merge_short_text_in_array(intput_texts, 5)
    audio_opt = []
    if not ref_free:
        prompt_text_phones, prompt_text_bert, norm_prompt_text = get_phones_and_bert(prompt_text, prompt_language)

    for idx, text in enumerate(intput_texts):
        if (len(text.strip()) == 0):
            continue
        
        logger.info(f"The {idx} sentence before process: <{text}>")
        input_text_phones, input_text_bert, norm_input_text = get_phones_and_bert(text, text_language)
        logger.info(f"The {idx} sentence after process: <{norm_input_text}>")

        if not ref_free:
            bert = torch.cat([prompt_text_bert, input_text_bert], 1)
            all_phoneme_ids = torch.LongTensor(prompt_text_phones + input_text_phones).to(device).unsqueeze(0)
        else:
            bert = input_text_bert
            all_phoneme_ids = torch.LongTensor(input_text_phones).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        
        with torch.no_grad():
            pred_semantic, idx = g_gpt_model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=int(g_sovits_config.model.semantic_frame_rate[:-2]) * g_gpt_config.data.max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
       
        refers=[]
        if(len(refers)==0):
            refers = [get_spectrogram(ref_wav_path).to(torch.float32).to(device)]

        audio = g_sovits_model.decode(pred_semantic, torch.LongTensor(input_text_phones).to(device).unsqueeze(0), \
                                      refers, speed=speed).detach().cpu().numpy()[0, 0]
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
    yield sample_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)

def b_refresh_weight():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}

def change_sovits_weights(sovits_path):
    global g_sovits_model, g_sovits_config
    sovits_ckpt = torch.load(sovits_path, map_location="cpu")
    config = sovits_ckpt["config"]
    g_sovits_config = HParams(**config)
    
    config = g_sovits_config
    g_sovits_model = SynthesizerTrn(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        n_speakers=config.data.n_speakers,
        **config.model
    )
    
    g_sovits_model.load_state_dict(sovits_ckpt["weight"], strict=False)
    g_sovits_model = g_sovits_model.to(device)
    g_sovits_model.eval()

    total = sum([param.nelement() for param in g_sovits_model.parameters()])
    current_sovits_name = os.path.basename(sovits_path)
    logger.info(f"Load SoVITS {current_sovits_name}.")
    logger.info(f"Number of parameter: {total / 1e6:.2f}M")

def change_gpt_weights(gpt_path):
    global g_gpt_model, g_gpt_config
    gpt_ckpt = torch.load(gpt_path, map_location="cpu")
    config = gpt_ckpt["config"]
    g_gpt_config = HParams(**config)  
    g_gpt_model = Text2SemanticDecoder(config)
    state_dict = remove_parameter_prefix(gpt_ckpt["weight"],'model.')
    g_gpt_model.load_state_dict(state_dict)

    g_gpt_model = g_gpt_model.to(device)
    g_gpt_model.eval()
    total = sum([param.nelement() for param in g_gpt_model.parameters()])
    current_gpt_name = os.path.basename(gpt_path)
    logger.info(f"Load GPT {current_gpt_name}.")
    logger.info(f"Number of parameter: {total / 1e6:.2f}M")

def change_auto_asr(auto_asr):
    global g_asr_model
    if auto_asr and g_asr_model is None:
        logger.info(f"Loading Whisper model.")
        g_asr_model = WhisperModel("medium", download_root=ASR_CKPT_PATH)
        logger.info(f"Load Whisper model finish.")
    else:
        logger.info(f"Unload Whisper model.")
        del g_asr_model
        g_asr_model = None
    return gr.update(interactive=False if g_asr_model is None else True)

def b_asr(audio_file, language=None):
    segments, info = g_asr_model.transcribe(audio_file, language=language, beam_size=5, \
        initial_prompt="Punctuation is needed in any language.")
    text = ''.join([s.text for s in segments])
    text = text.strip()
    return gr.update(value=text)

def init():
    global g_ssl_model
    time_tracker = RuntimeTracker()
    time_tracker.start("Load CNHubert MODEL")
    g_ssl_model = CNHubert(CUHUBERT_CKPT_PATH)
    g_ssl_model = g_ssl_model.to(device)
    g_ssl_model.eval()
    time_tracker.end()

    sovits_names, gpt_names = get_weights_names()
    return sovits_names, gpt_names
    
def main():
    sovits_names, gpt_names = init()
    with gr.Blocks(title="GPT-SoVITS WebUI") as app:
        with gr.Row():
            GPT_dropdown = gr.Dropdown(
                label="GPT weight list", 
                choices=sorted(gpt_names, key=custom_sort_key), 
                value=None, 
                interactive=True, 
                scale=14
            )

            SoVITS_dropdown = gr.Dropdown(
                label="SoVITS weight list", 
                choices=sorted(sovits_names, key=custom_sort_key), 
                value=None, 
                interactive=True, 
                scale=14
            )
            refresh_button = gr.Button("Refresh Model Weight Root", variant="primary", scale=14)
            refresh_button.click(fn=b_refresh_weight, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])

        with gr.Row():
            inp_ref = gr.Audio(label="Reference Audio", type="filepath", scale=13)

            with gr.Column():
                auto_asr = gr.Checkbox(label="Auto ASR", value=False, interactive=True, show_label=True,scale=1)
                prompt_text = gr.Textbox(label="Reference Text", value="", lines=5, max_lines=5,scale=1)
                prompt_language = gr.Dropdown(
                    label="Prompt Language", choices=list(SUPPORT_LANGUAGE), value=SUPPORT_LANGUAGE[0],
                )
                asr_button = gr.Button("ASR", variant="primary", interactive=False, scale=1)
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input Text", value="how are you today? how about your famliy?", lines=26, max_lines=26)
            with gr.Column():
                text_language = gr.Dropdown(
                        label="Input Language", choices=list(SUPPORT_LANGUAGE), value=SUPPORT_LANGUAGE[0], scale=1
                    )
    
                top_k = gr.Slider(minimum=1,maximum=100,step=1,label="top_k",value=15,interactive=True, scale=1)
                top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label="top_p",value=1,interactive=True, scale=1)
                temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label="temperature",value=1,interactive=True,  scale=1) 
        with gr.Row():
            inference_button = gr.Button("Synthesis", variant="primary", size='lg', scale=25)
            output = gr.Audio(label="Output Audio", scale=14)

        inference_button.click(
            get_tts_wav,
            inputs = [
                inp_ref, 
                prompt_text, 
                prompt_language, 
                text, 
                text_language, 
                top_k, 
                top_p, 
                temperature, 
            ],
            outputs= [output],
        )

        SoVITS_dropdown.change(
            change_sovits_weights, 
            inputs=[
                SoVITS_dropdown,
                ], 
            outputs=[]
            )
        
        GPT_dropdown.change(
            change_gpt_weights, 
            inputs=[GPT_dropdown], 
            outputs=[]
            )
        
        auto_asr.change(
            change_auto_asr,
            inputs=[auto_asr],
            outputs=[asr_button],
        )

        asr_button.click(
            b_asr,
            inputs=[inp_ref, prompt_language],
            outputs=[prompt_text],
        )

        app.queue().launch(#concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=8080,
        quiet=True,
        )

if __name__ == '__main__':
    main()