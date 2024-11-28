import os
from text import symbols, cleaned_text_to_sequence
from text import english
special = [
    # ("%", "zh", "SP"),
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
    # ('@', 'zh', "SP4")#不搞鬼畜了，和第二版保持一致吧
]

def english_clean_text(text, language):
    symbols_table = symbols.SYMBOLS_TABLE
    norm_text = english.text_normalize(text)
    phones = english.g2p(norm_text)
    if len(phones) < 4:
        phones = [','] + phones
    word2ph = None
    phones = ['UNK' if ph not in symbols_table else ph for ph in phones]
    return phones, word2ph, norm_text

def clean_text(text, language):
    symbols_table = symbols.SYMBOLS_TABLE
    language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean","yue":"cantonese"}

    if language not in language_module_map:
        language = "en"
        text = " "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    language_module = __import__("text."+language_module_map[language],fromlist=[language_module_map[language]])
    if hasattr(language_module,"text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text=text
    if language == "zh" or language=="yue":##########
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ['UNK' if ph not in symbols_table else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    symbols = symbols.symbols
    language_module_map = {"zh": "chinese2", "ja": "japanese", "en": "english", "ko": "korean","yue":"cantonese"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text."+language_module_map[language],fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text

def text_to_sequence(text, language, version=None):
    version = os.environ.get('version',version)
    if version is None:version='v2'
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones, version)