import os
from text import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols.SYMBOLS_TABLE)}

def cleaned_text_to_sequence(cleaned_text):
  return [_symbol_to_id[symbol] for symbol in cleaned_text]
