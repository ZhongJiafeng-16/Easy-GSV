import numpy as np

def remove_parameter_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key =key.replace(prefix, "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def batch_sequences(sequences, axis=0, pad_val=0):
    seq = sequences[0]
    ndim = seq.ndim
    if axis < 0:
        axis += ndim
    dtype = seq.dtype
    pad_val = dtype.type(pad_val)

    seq_lengths = [seq.shape[axis] for seq in sequences]
    max_length = np.max(seq_lengths)

    padded_sequences = []
    for seq, length in zip(sequences, seq_lengths):
        padding = ([(0, 0)] * axis + [(0, max_length - length)] + [(0, 0)] * (ndim - axis - 1))
        padded_seq = np.pad(seq, padding, mode="constant", constant_values=pad_val)
        padded_sequences.append(padded_seq)
        
    batch = np.stack(padded_sequences)
    return batch


