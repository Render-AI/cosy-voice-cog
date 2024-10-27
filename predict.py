from cog import BasePredictor, Input, Path
import torch
import torchaudio
import os
import numpy as np
import random
import logging
import time
from typing import Optional

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and set up required components"""
        from cosyvoice.cli.cosyvoice import CosyVoice
        
        print("Starting model loading...")
        total_start = time.time()
        
        # Use local model paths from pretrained_models directory
        base_model_path = "pretrained_models/CosyVoice-300M"
        sft_model_path = "pretrained_models/CosyVoice-300M-SFT"
        instruct_model_path = "pretrained_models/CosyVoice-300M-Instruct"
        
        # Initialize models using local paths
        print("\nLoading base model...")
        start = time.time()
        self.base_model = CosyVoice(base_model_path)
        print(f"Base model loaded in {time.time() - start:.2f} seconds")
        
        print("\nLoading SFT model...")
        start = time.time()
        self.sft_model = CosyVoice(sft_model_path)
        print(f"SFT model loaded in {time.time() - start:.2f} seconds")
        
        print("\nLoading instruct model...")
        start = time.time()
        self.instruct_model = CosyVoice(instruct_model_path)
        print(f"Instruct model loaded in {time.time() - start:.2f} seconds")
        
        self.prompt_sr = 16000
        self.target_sr = 22050
        
        # Get available speakers from SFT model
        print("\nGetting available speakers...")
        start = time.time()
        self.available_speakers = self.sft_model.list_avaliable_spks()
        print(f"Got available speakers in {time.time() - start:.2f} seconds")
        print('available speakers: ', self.available_speakers)

        total_time = time.time() - total_start
        print(f"\nTotal setup time: {total_time:.2f} seconds")

    def set_random_seed(self, seed):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def predict(
        self,
        mode: str = Input(
            choices=["sft", "zero_shot", "cross_lingual", "instruct"],
            default="zero_shot",
            description="Voice synthesis mode - sft: use pretrained voices, zero_shot: 3-second voice cloning, cross_lingual: cross-language cloning, instruct: personality-driven synthesis"
        ),
        text: str = Input(
            description="Text to be synthesized",
            default="Hello, this is a test."
        ),
        reference_audio: Path = Input(
            description="Reference audio file for voice cloning (required for zero_shot and cross_lingual modes)",
            default=None
        ),
        reference_text: str = Input(
            description="Text corresponding to the reference audio (required for zero_shot mode)",
            default=None
        ),
        speaker: str = Input(
            description="Speaker ID for SFT mode (ignored in other modes)",
            default=None
        ),
        instruct_text: str = Input(
            description="Personality description for instruct mode (e.g., 'passionate, energetic speaker')",
            default=None
        ),
        seed: int = Input(
            description="Random seed for reproducibility",
            default=None
        )
    ) -> Path:
        """Run voice synthesis prediction based on selected mode"""
        
        if seed is not None:
            self.set_random_seed(seed)
        else:
            seed = random.randint(1, 100000000)
            self.set_random_seed(seed)

        # Input validation based on mode
        if mode in ["zero_shot", "cross_lingual"] and reference_audio is None:
            raise ValueError(f"Reference audio is required for {mode} mode")
        
        if mode == "zero_shot" and not reference_text:
            raise ValueError("Reference text is required for zero_shot mode")
            
        if mode == "sft" and not speaker:
            raise ValueError("Speaker ID is required for sft mode")
            
        if mode == "instruct" and not instruct_text:
            raise ValueError("Instruct text is required for instruct mode")

        # Process reference audio if provided
        if reference_audio is not None:
            info = torchaudio.info(str(reference_audio))
            if info.sample_rate < self.prompt_sr:
                raise ValueError(f"Reference audio sample rate {info.sample_rate} is lower than required {self.prompt_sr}")
            
            from cosyvoice.utils.file_utils import load_wav
            reference_speech = load_wav(str(reference_audio), self.prompt_sr)

        # Generate audio based on mode
        output_path = Path("/tmp/output.wav")
        
        if mode == "sft":
            output = self.sft_model.inference_sft(text, speaker)
        elif mode == "zero_shot":
            output = self.base_model.inference_zero_shot(text, reference_text, reference_speech)
        elif mode == "cross_lingual":
            output = self.base_model.inference_cross_lingual(text, reference_speech)
        else:  # instruct
            output = self.instruct_model.inference_instruct(text, speaker, instruct_text)

        # Save output
        torchaudio.save(
            str(output_path),
            output['tts_speech'],
            self.target_sr
        )
        
        return output_path