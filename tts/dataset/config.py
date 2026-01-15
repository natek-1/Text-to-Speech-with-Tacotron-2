from dataclasses import dataclass

@dataclass
class TTSDatasetConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    n_mels: int = 80
    hop_length: int = 256
    win_length: int = 1024
    fmin: int = 0
    fmax: int = 8000
    center: bool = False
    min_db: float = -100.
    max_abs_val: float = 4
    normalize: bool = True
    batch_size: int = 4
    num_workers: int = 0
    drop_last: bool = False
    