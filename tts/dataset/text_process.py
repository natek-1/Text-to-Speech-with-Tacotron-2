eos = '~'
pad = '_'
chars = pad + eos + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? '
unk_idx = len(chars)

char_to_id = {char: i for i, char in enumerate(chars)}
id_to_char = {i: char for i, char in enumerate(chars)}


def text_to_sequence(text):
    text += eos
    return [char_to_id.get(char, unk_idx) for char in text]


def sequence_to_text(sequence):
    return "".join(id_to_char.get(i, unk_idx) for i in sequence)
