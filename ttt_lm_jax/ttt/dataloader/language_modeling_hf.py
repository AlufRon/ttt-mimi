"""
A data loader for Mimi audio tokens using the Token Offset Approach.
Modified to use pre-tokenized Mimi data with token offsets to distinguish between codebooks.
"""

import os
import os.path as osp
from itertools import chain
from pathlib import Path
import pickle
import glob

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from datasets import Dataset, Audio

from ttt.dataloader.lm_dataset import RandomFaultTolerantSampler, LMDataset
from ttt.infra.jax_utils import master_print

# Import necessary components from transformers
from transformers import AutoFeatureExtractor
from transformers import AutoConfig

class LMDataModule:
    def __init__(
        self,
        dataset_name="unused",  # not used since we load local data
        tokenizer_name="kyutai/mimi",  # use the Mimi model from Hugging Face
        dataset_config_name=None,
        raw_txt_path=None,
        max_length=1024,
        cache_dir=None,
        raw_json_path=None,  # path to raw data directory
        tokenized_data_dir=None,  # path to pre-tokenized data
        val_ratio=0.0005,
        val_split_seed=2357,
        add_eos=False,  # typically not needed for speech
        batch_size=32,
        batch_size_eval=None,
        num_workers=1,
        loader_workers=1,
        shuffle=True,
        pin_memory=False,
        drop_last=False,
        fault_tolerant=False,
        tokenization_device="cuda",  # kept for API compatibility
        num_codebooks=8,  # Number of Mimi codebooks to use
        codebook_size=2048,  # Size of each Mimi codebook
    ):
        super().__init__()
        master_print("Initializing LMDataModule with Mimi token support (Token Offset Approach)")
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name if tokenizer_name else "kyutai/mimi"
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.raw_json_path = raw_json_path
        self.tokenized_data_dir = tokenized_data_dir  # store path to tokenized data
        self.max_length = max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.add_eos = add_eos
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.loader_workers = loader_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        if fault_tolerant:
            assert self.shuffle, "Fault tolerant mode requires shuffle=True"
        self.fault_tolerant = fault_tolerant
        self.tokenization_device = tokenization_device  # kept for API compatibility
        
        # Mimi-specific parameters
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.total_vocab_size = codebook_size * num_codebooks
        
        master_print("LMDataModule initialized with parameters:")
        master_print(f"  tokenizer_name: {self.tokenizer_name}")
        master_print(f"  max_length: {self.max_length}")
        master_print(f"  batch_size: {self.batch_size}")
        master_print(f"  tokenized_data_dir: {self.tokenized_data_dir}")
        master_print(f"  num_codebooks: {self.num_codebooks}")
        master_print(f"  codebook_size: {self.codebook_size}")
        master_print(f"  total_vocab_size: {self.total_vocab_size}")
        master_print(f"  fault_tolerant: {self.fault_tolerant}")

    def prepare_data(self):
        """
        Check for the existence of pre-tokenized data and create a mapping for each split.
        """
        if self.tokenized_data_dir is None:
            master_print("No tokenized_data_dir provided, skipping preparation")
            return
            
        master_print(f"Checking pre-tokenized data directory: {self.tokenized_data_dir}")
        
        splits_map = {"train": "train", "validation": "val", "test": "test"}
        tokenized_files = {}
        
        for split_key, subdir in splits_map.items():
            split_dir = os.path.join(self.tokenized_data_dir, subdir)
            if not os.path.isdir(split_dir):
                master_print(f"Warning: Tokenized data directory not found: {split_dir}")
                continue
            
            pt_files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
            master_print(f"Found {len(pt_files)} tokenized files for split '{split_key}' in {split_dir}")
            
            if len(pt_files) == 0:
                master_print(f"Warning: No tokenized files found in {split_dir}")
                continue
                
            tokenized_files[split_key] = pt_files
        
        self.tokenized_files = tokenized_files
        master_print("Pre-tokenized data verification complete.")

    def setup(self, stage=None):
        master_print("Setting up the data module...")
        if stage == "test" and hasattr(self, "dataset_test"):
            return
            
        concat_ids, self.tokenizer = self.process_dataset()
        self.vocab_size = self.total_vocab_size
        master_print(f"Tokenizer loaded with total vocab_size: {self.vocab_size}")
        
        # Check the dataset format and shape
        for split in ["train", "validation", "test"]:
            if split not in concat_ids:
                master_print(f"Warning: {split} split not available")
                continue
                
            self._log_dataset_info(concat_ids[split], split)
        
        # Create LM datasets for each split
        self.dataset_train, self.dataset_val, self.dataset_test = [
            LMDataset(
                concat_ids[split] if split in concat_ids else np.array([0], dtype=np.int32), 
                seq_len=self.max_length, 
                llama2=False
            )
            for split in ["train", "validation", "test"]
        ]
        
        master_print(f"Train dataset created with length: {len(self.dataset_train)}")
        master_print(f"Validation dataset created with length: {len(self.dataset_val)}")
        master_print(f"Test dataset created with length: {len(self.dataset_test)}")
        master_print("Setup complete.")

    def _log_dataset_info(self, data, split):
        """Helper to log dataset information"""
        if isinstance(data, dict) and "input_ids" in data:
            inp_ids = data["input_ids"]
            if isinstance(inp_ids, list):
                inp_ids = np.array(inp_ids)
            master_print(f"[Setup] {split} dataset shape: {inp_ids.shape}")
            master_print(f"[Setup] {split} dataset sample: {inp_ids[:10].tolist() if inp_ids.size > 10 else inp_ids.tolist()}")
        else:
            master_print(f"[Setup] {split} dataset is a numpy array of shape: {data.shape}")
            master_print(f"[Setup] {split} dataset sample: {data[:10].tolist() if data.size > 10 else data.tolist()}")

    def process_dataset(self):
        """
        Process the dataset using the Token Offset Approach for Mimi tokens.
        """
        # Check if we can load from cache
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None and cache_dir.is_dir():
            master_print(f"Found cached data at {cache_dir}")
            return self._load_from_cache(cache_dir)
        
        master_print("Processing dataset with Token Offset Approach...")
        
        # Load feature extractor to get sampling rate
        master_print(f"Loading feature extractor from {self.tokenizer_name}...")
        feature_extractor = None
        sampling_rate = 24000  # Default for Mimi
        
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.tokenizer_name)
            sampling_rate = feature_extractor.sampling_rate
            master_print(f"Loaded feature extractor, sampling_rate: {sampling_rate}")
        except Exception as e:
            master_print(f"Failed to load feature extractor: {e}")
            master_print(f"Using default sampling rate: {sampling_rate}")
        
        # If we don't have tokenized files, create dummy data
        if not hasattr(self, 'tokenized_files') or not self.tokenized_files:
            master_print("No tokenized files available, creating dummy data")
            return self._create_dummy_data(sampling_rate)
        
        # Process each split
        concat_ids = {}
        for split, file_paths in self.tokenized_files.items():
            master_print(f"Processing {len(file_paths)} tokenized files for {split} split...")
            
            # Create a dataset with the file paths
            split_dataset = Dataset.from_dict({"file_path": file_paths})
            
            # Define a function to load a tokenized file with token offset approach
            def load_tokenized_file(example):
                file_path = example["file_path"]
                try:
                    tokens_tensor = torch.load(file_path)
                    master_print(f"Sample token tensor shape: {tokens_tensor.shape}")
                    
                    # Apply token offsets to distinguish different codebooks
                    combined_tokens = []
                    
                    for codebook_idx in range(min(self.num_codebooks, tokens_tensor.size(1))):
                        offset = codebook_idx * self.codebook_size
                        codebook_tokens = tokens_tensor[0, codebook_idx].tolist()
                        # Add offset to each token based on its codebook
                        combined_tokens.extend([t + offset for t in codebook_tokens])
                        
                    return {"input_ids": combined_tokens, "len": len(combined_tokens)}
                except Exception as e:
                    master_print(f"Error loading file {file_path}: {str(e)}")
                    return {"input_ids": [0], "len": 1}
            
            # Map the loading function to all files
            tokenized = split_dataset.map(
                load_tokenized_file,
                batched=False,
                num_proc=self.num_workers,
                desc=f"Loading tokenized data for {split} split"
            )
            
            # Concatenate all tokens for the split
            def tokenize_concat(examples):
                master_print(f"Concatenating tokens for {split} split...")
                all_tokens = list(chain(*examples["input_ids"]))
                if len(all_tokens) == 0:
                    master_print("Warning: No tokens found; creating dummy sequence.")
                    all_tokens = [0] * self.max_length
                master_print(f"Concatenated token length: {len(all_tokens)}")
                return np.array(all_tokens, dtype=np.int32)
            
            # Concatenate tokens and convert to numpy array
            all_tokens = tokenize_concat(tokenized)
            concat_ids[split] = all_tokens
            master_print(f"Processed {split} split with {len(all_tokens)} tokens")
        
        # Create a simple tokenizer info dict
        tokenizer_info = {
            "vocab_size": self.total_vocab_size, 
            "sampling_rate": sampling_rate,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size
        }
        
        # Save to cache if needed
        if cache_dir is not None:
            self._save_to_cache(concat_ids, tokenizer_info, cache_dir)
        
        master_print("Dataset processing complete.")
        return concat_ids, tokenizer_info

    def _create_dummy_data(self, sampling_rate):
        """Create dummy data when no real tokenized files are available"""
        master_print("Creating dummy data for testing purposes")
        dummy_length = 10000  # Length of dummy sequence
        
        # Create random token sequences
        concat_ids = {}
        for split in ["train", "validation", "test"]:
            # Random tokens with appropriate offsets based on codebook
            tokens = []
            seq_per_codebook = dummy_length // self.num_codebooks
            
            for codebook_idx in range(self.num_codebooks):
                offset = codebook_idx * self.codebook_size
                # Generate random tokens within the codebook range
                codebook_tokens = np.random.randint(0, self.codebook_size, size=seq_per_codebook)
                # Add offset
                tokens.extend(codebook_tokens + offset)
            
            concat_ids[split] = np.array(tokens, dtype=np.int32)
            master_print(f"Created dummy {split} split with {len(tokens)} tokens")
        
        tokenizer_info = {
            "vocab_size": self.total_vocab_size, 
            "sampling_rate": sampling_rate,
            "num_codebooks": self.num_codebooks,
            "codebook_size": self.codebook_size
        }
        
        return concat_ids, tokenizer_info

    def _save_to_cache(self, concat_ids, tokenizer, cache_dir):
        """Save processed data to cache"""
        cache_dir.mkdir(parents=True, exist_ok=True)
        master_print(f"Saving to cache at {str(cache_dir)}")
        
        for k, v in concat_ids.items():
            np.save(cache_dir / f"{k}.npy", v)
            
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
            
        master_print("Data successfully saved to cache.")

    def _load_from_cache(self, cache_dir):
        """Load processed data from cache"""
        master_print(f"Loading from cache at {str(cache_dir)}")
        
        concat_ids = {}
        for split in ["train", "validation", "test"]:
            file_path = cache_dir / f"{split}.npy"
            if file_path.exists():
                concat_ids[split] = np.load(file_path, mmap_mode="r")
                master_print(f"Loaded {split} split with shape {concat_ids[split].shape}")
            else:
                master_print(f"Warning: {split}.npy not found in cache")
        
        with open(cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
            
        # Update total vocab size from cache if available
        if "vocab_size" in tokenizer:
            self.total_vocab_size = tokenizer["vocab_size"]
            master_print(f"Updated total_vocab_size from cache: {self.total_vocab_size}")
            
        if "num_codebooks" in tokenizer:
            self.num_codebooks = tokenizer["num_codebooks"]
            master_print(f"Updated num_codebooks from cache: {self.num_codebooks}")
            
        if "codebook_size" in tokenizer:
            self.codebook_size = tokenizer["codebook_size"]
            master_print(f"Updated codebook_size from cache: {self.codebook_size}")
            
        return concat_ids, tokenizer

    @property
    def _cache_dir_name(self):
        """Generate a unique cache directory name"""
        return (
            f"mimi-tokenizer-{self.tokenizer_name}-codebooks-{self.num_codebooks}-"
            f"codebooksize-{self.codebook_size}-val_ratio-{self.val_ratio}-"
            f"val_split_seed-{self.val_split_seed}-add_eos-{self.add_eos}"
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        """The train dataloader"""
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            sampler = RandomFaultTolerantSampler(self.dataset_train)
        else:
            shuffle = self.shuffle
            sampler = None

        return self._data_loader(self.dataset_train, batch_size=self.batch_size, shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        """The val dataloader"""
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        """The test dataloader"""
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset, batch_size: int, shuffle: bool = False, sampler=None) -> DataLoader:
        """Create a dataloader with the given parameters"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.loader_workers,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
