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

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and set up directories"""
        print("Starting setup...")
        
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
    ) -> Path:
        """Run voice synthesis prediction"""
        
        # Create output directory if it doesn't exist
        os.makedirs("/tmp", exist_ok=True)
        
        # Input validation
        if mode in ["zero_shot", "cross_lingual"] and not prompt_audio:
            raise ValueError(f"Prompt audio is required for {mode} mode")
        
        if mode == "zero_shot":
            if not text or not prompt_text:
                raise ValueError("Text and prompt text are required for zero_shot mode")
        
        if mode == "cross_lingual" and not text:
            raise ValueError("Text is required for cross_lingual mode")
            
        if mode == "voice_conversion":
            if not source_audio or not target_audio:
                raise ValueError("Source and target audio are required for voice conversion mode")

        # Process audio based on mode
        output_path = Path("/tmp/output.wav")
        
        try:
            if mode == "zero_shot":
                prompt_speech = load_wav(str(prompt_audio), self.prompt_sr)
                output = self.model.inference_zero_shot(
                    text,
                    prompt_text,
                    prompt_speech,
                    stream=False
                )
            
            elif mode == "cross_lingual":
                prompt_speech = load_wav(str(prompt_audio), self.prompt_sr)
                output = self.model.inference_cross_lingual(
                    text,
                    prompt_speech,
                    stream=False
                )
            
            else:  # voice_conversion
                source_speech = load_wav(str(source_audio), self.prompt_sr)
                target_speech = load_wav(str(target_audio), self.prompt_sr)
                output = self.model.inference_vc(
                    source_speech,
                    target_speech,
                    stream=False
                )

            # Save output
            # The models return a generator, so we need to get the first item
            output_audio = next(output)
            torchaudio.save(
                str(output_path),
                output_audio['tts_speech'],
                self.target_sr
            )
            
            return output_path
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise e