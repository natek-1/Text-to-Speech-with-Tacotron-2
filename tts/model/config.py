from dataclasses import dataclass


@dataclass
class Tacotron2Config:

    num_mels: int = 80

    # character embedding
    character_embedding_size: int = 512
    num_chars: int = 67
    pad_token_id: int = 0

    # encoder config
    encoder_kernel_size: int = 5
    encoder_num_convolutions: int = 3
    encoder_embed_dim: int = 512
    encoder_dropout: float = 0.5 


    # decoder config
    decoder_embed_dim: int = 1024
    decoder_prenet_dim: int = 256
    decoder_prenet_num_layers: int = 2
    decoder_prenet_dropout: float = 0.1
    decoder_postnet_num_conv: int = 5
    decoder_postnet_num_filters: int = 512
    decoder_postnet_kernel_size: int = 5
    decoder_postnet_dropout: float = 0.5
    decoder_dropout: float = 0.1


    # attention config
    attention_dim: int = 128
    attention_location_num_filters: int = 32
    attention_location_kernel_size: int = 31
    attention_dropout: float = 0.1

    



