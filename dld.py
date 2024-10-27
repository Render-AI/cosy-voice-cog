import os
from modelscope import snapshot_download

def download_models():
    print("Starting model downloads...")
    models = [
        'CosyVoice-300M',
        'CosyVoice-300M-SFT',
        'CosyVoice-300M-Instruct'
    ]
    
    os.makedirs('pretrained_models', exist_ok=True)
    
    for model in models:
        model_path = f'pretrained_models/{model}'
        if not os.path.exists(model_path):
            print(f"\nDownloading {model}...")
            snapshot_download(f'iic/{model}', local_dir=model_path)
            print(f"Downloaded {model}")
        else:
            print(f"\n{model} already exists, skipping download")

if __name__ == "__main__":
    download_models()