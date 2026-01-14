import torch


class Tokenizer:
    
    
    def __init__(self):
        
        self.eos_token = "<EOS>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        self.chars = [self.pad_token, self.eos_token, self.unk_token] + \
                    list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? ')
                    
        self.char2id = {char: idx for idx, char in enumerate(self.chars)}
        self.id2char = {idx: char for idx, char in enumerate(self.chars)}
        
        self.eos_token_id = self.char2id[self.eos_token]
        self.pad_token_id = self.char2id[self.pad_token]
        self.unk_token_id = self.char2id[self.unk_token]
        self.vocab_size = len(self.chars)
        
    def encode(self, text, return_tensors=True):
        '''
        Converts a text string into a list of token IDs.
        
        Inputs:
            text (str): Input text string.
            return_tensors (bool): Whether to return a PyTorch tensor.
        Outputs:
            token_ids (list or torch.Tensor): List of token IDs or PyTorch tensor.
        '''
        tokens = [self.char2id.get(char, self.unk_token_id) for char in text]
        tokens.append(self.eos_token_id)
        
        if return_tensors:
            return torch.LongTensor(tokens)
        return tokens

    def decode(self, token_ids, include_special_tokens=False):
        '''
        Converts a list of token IDs back into a text string.
        
        Inputs:
            token_ids (list or torch.Tensor): List of token IDs or PyTorch tensor.
            include_special_tokens (bool): Whether to include special tokens in the output text.
        Outputs:
            text (str): Decoded text string.
        '''
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        chars = []
        for token_id in token_ids:
            if not include_special_tokens and token_id in [self.eos_token_id, self.pad_token_id]:
                continue
            chars.append(self.id2char.get(token_id, self.unk_token))
        
        return ''.join(chars)

if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.vocab_size)
        
        