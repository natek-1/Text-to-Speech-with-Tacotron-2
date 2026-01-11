import os
import argparse

import librosa

import pandas as pd
from sklearn.model_selection import train_test_split


def random_split(dataset_dir, save_dir, random_state=42, test_size=0.1, sort=True):
    '''
    Splits a dataset into training and testing sets randomly and saves them as CSV files.
    
    Inputs:
        dataset_dir (str): Directory to the dataset CSV file.
        save_dir (str): Directory where the split CSV files will be saved.
        random_state (int): Random seed for reproducibility.
        test_size (float): Proportion of the dataset to include in the test split.
        sort (bool): Whether to sort the dataset by audio duration before splitting.
    Ouputs:
        None
    '''
    meta_dataset_path = os.path.join(dataset_dir, 'metadata.csv')
    df = pd.read_csv(meta_dataset_path, sep='|', header=None, names=['file_path', 'text', 'normalized_text'])
    
    df = df[['file_path', 'normalized_text']]
    # some rows might have empty text, we need to remove them
    df = df[~df['normalized_text'].isna()].reset_index(drop=True)
    
    audio_path_func = lambda x: os.path.join(dataset_dir, 'wavs', f"{x}.wav")
    df['file_path'] = df['file_path'].apply(audio_path_func)
    
    # asseert all audio files exist
    assert all([os.path.isfile(p) for p in df['file_path'].tolist()]), "Some audio files are missing!"
    
    # get audio duration
    duration_func = lambda x: librosa.get_duration(filename=x)
    df['duration'] = df['file_path'].apply(duration_func)
    
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    if sort:
        train_df = train_df.sort_values(by='duration', ascending=False)
    
    train_path = os.path.join(save_dir, 'train.csv')
    test_path = os.path.join(save_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Train and test sets saved to {save_dir}")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Randomly split a dataset into training and testing sets.")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory to the dataset CSV file.")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory where the split CSV files will be saved.")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--test_size', type=float, default=0.1, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--sort', action='store_true', help="Whether to sort the dataset by audio duration before splitting.")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    random_split(args.dataset_dir, args.save_dir, args.random_state, args.test_size, args.sort)
    
    