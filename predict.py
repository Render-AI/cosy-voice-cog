from cog import BasePredictor, Input, Path
import torch
import torchaudio
import os
import shutil
import numpy as np
from typing import Optional
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from modelscope import snapshot_download
from audio_processor import process_audio_with_timeout

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and set up directories"""
        print("Starting setup...")

        # Get current directory and list its contents
        current_dir = os.getcwd()
        print(f"\nCurrent working directory: {current_dir}")
        print("\nListing all directories and their contents:")
        
        for root, dirs, files in os.walk(current_dir):
            level = root.replace(current_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 1)
            for d in dirs:
                print(f"{subindent}{d}/")
            for f in files:
                print(f"{subindent}{f}")
        
        # Create necessary directories
        directories = [
            'pretrained_models',
            '/tmp',
            'pretrained_models/CosyVoice-300M'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                print(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        
        # Download base model if it doesn't exist or is empty
        model_path = 'pretrained_models/CosyVoice-300M'
        if not os.path.exists(model_path) or len(os.listdir(model_path)) == 0:
            print("\nDownloading CosyVoice-300M...")
            # If directory exists but is empty, remove it first
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                os.makedirs(model_path)
            snapshot_download('iic/CosyVoice-300M', local_dir=model_path)
            print("Download completed")
        else:
            print("\nModel already exists in pretrained_models/CosyVoice-300M")
        
        # Initialize model
        print("\nLoading model...")
        self.model = CosyVoice(model_path)
        self.prompt_sr = 16000
        self.target_sr = 22050
        print("Model loaded successfully")

    def predict(
        self,
        mode: str = Input(
            choices=["zero_shot", "cross_lingual", "voice_conversion"],
            default="zero_shot",
            description="Voice synthesis mode"
        ),
        text: str = Input(
            description="Text to be synthesized (for zero_shot and cross_lingual modes)",
            default=None
        ),
        prompt_text: str = Input(
            description="Prompt text corresponding to the prompt audio (for zero_shot mode only)",
            default=None
        ),
        prompt_audio: Path = Input(
            description="Prompt audio file (for zero_shot and cross_lingual modes)",
            default=None
        ),
        source_audio: Path = Input(
            description="Source audio file for voice conversion",
            default=None
        ),
        target_audio: Path = Input(
            description="Target audio file for voice conversion",
            default=None
        ),
        speed: float = Input(
            description="Speech speed factor",
            default=1.0,
            ge=0.2
        ),
        max_chunk_time: int = Input(
            description="Maximum time in seconds for processing each chunk",
            default=30
        ),
        use_cpu: bool = Input(
            description="Force CPU usage instead of GPU",
            default=False
        ),
        use_half_precision: bool = Input(
            description="Enable FP16 precision for faster processing",
            default=True
        ),
        optimize_memory: bool = Input(
            description="Enable memory optimizations",
            default=True
        ),
    ) -> Path:
        """Run voice synthesis prediction"""
        
        # Create output directory if it doesn't exist
        os.makedirs("/tmp", exist_ok=True)
        output_path = Path("/tmp/output.wav")
        
        # Input validation
        if mode in ["zero_shot", "cross_lingual"] and not prompt_audio:
            raise ValueError(f"Prompt audio is required for {mode} mode")
        
        if mode == "zero_shot" and (not text or not prompt_text):
            raise ValueError("Text and prompt text are required for zero_shot mode")
        
        if mode == "cross_lingual" and not text:
            raise ValueError("Text is required for cross_lingual mode")
            
        if mode == "voice_conversion" and (not source_audio or not target_audio):
            raise ValueError("Source and target audio are required for voice conversion mode")

        try:
            # Load audio inputs based on mode
            if mode in ["zero_shot", "cross_lingual"]:
                prompt_speech_16k = load_wav(str(prompt_audio), self.prompt_sr)
            
            if mode == "voice_conversion":
                source_speech_16k = load_wav(str(source_audio), self.prompt_sr)
                prompt_speech_16k = load_wav(str(target_audio), self.prompt_sr)

            # Process audio with timeout handling
            return process_audio_with_timeout(
                model=self.model,
                mode=mode,
                text=text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
                source_speech_16k=source_speech_16k if mode == "voice_conversion" else None,
                speed=speed,
                max_chunk_time=max_chunk_time,
                output_path=str(output_path),
                target_sr=self.target_sr,
                use_cpu=use_cpu,
                use_half_precision=use_half_precision,
                optimize_memory=optimize_memory
            )
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise e