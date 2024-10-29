import threading
import time
import torch
import torchaudio
import queue
import os
from typing import Optional, List, Generator

class ChunkTimeoutError(Exception):
    pass

def process_audio_with_timeout(
    model,
    mode: str,
    text: Optional[str] = None,
    prompt_text: Optional[str] = None,
    prompt_speech_16k: Optional[torch.Tensor] = None,
    source_speech_16k: Optional[torch.Tensor] = None,
    speed: float = 1.0,
    max_chunk_time: int = 30,
    output_path: str = "output.wav",
    target_sr: int = 22050,
    use_cpu: bool = False,
    use_half_precision: bool = True,  # Enable FP16 by default
    optimize_memory: bool = True      # Enable memory optimizations
):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        print(f"Running with {max_chunk_time} second timeout per chunk")
        all_chunks: List[torch.Tensor] = []
        
        # Performance optimizations
        if optimize_memory:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Move model to CPU if requested
        if use_cpu:
            if hasattr(model, 'cpu'):
                model.cpu()
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if hasattr(model, 'to'):
                model = model.to(device)
        
        # Optimize model settings
        if hasattr(model, 'eval'):
            model.eval()
        
        # Context managers for optimization
        inference_context = (
            torch.cuda.amp.autocast() if use_half_precision and not use_cpu
            else torch.no_grad()
        )
        
        with inference_context:
            print("Setting up generator...")
            if mode == "cross_lingual":
                generator = model.inference_cross_lingual(
                    tts_text=text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=False,
                    speed=speed
                )
            elif mode == "zero_shot":
                generator = model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=prompt_text,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=False,
                    speed=speed
                )
            else:  # voice_conversion
                generator = model.inference_vc(
                    source_speech_16k=source_speech_16k,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=False,
                    speed=speed
                )
            print("Generator setup complete")

            chunk_idx = 0
            while True:
                print(f"\nProcessing chunk {chunk_idx + 1}")
                chunk_start = time.time()
                
                try:
                    result_queue = queue.Queue()
                    def process_next():
                        try:
                            print(f"Thread: Starting chunk {chunk_idx + 1}")
                            chunk_output = next(generator)
                            chunk_tensor = chunk_output['tts_speech']
                            
                            # Optional: Move tensor to CPU to free GPU memory
                            if optimize_memory and not use_cpu:
                                chunk_tensor = chunk_tensor.cpu()
                                
                            result_queue.put(('success', chunk_tensor))
                        except StopIteration:
                            result_queue.put(('stop', None))
                        except Exception as e:
                            result_queue.put(('error', e))

                    thread = threading.Thread(target=process_next)
                    thread.daemon = True
                    thread.start()
                    
                    # Wait for result with timeout
                    start_wait = time.time()
                    while time.time() - start_wait < max_chunk_time:
                        try:
                            status, result = result_queue.get_nowait()
                            if status == 'stop':
                                print("Generation complete")
                                if all_chunks:
                                    final_audio = torch.cat(all_chunks, dim=1)
                                    torchaudio.save(output_path, final_audio, target_sr)
                                    print(f"Successfully generated {len(all_chunks)} chunks")
                                    print(f"Saved to: {os.path.abspath(output_path)}")
                                    return output_path
                                return None
                                
                            elif status == 'error':
                                raise result
                            else:
                                processed_chunk = result
                                break
                        except queue.Empty:
                            time.sleep(0.1)
                    else:
                        raise ChunkTimeoutError(f"Processing exceeded {max_chunk_time} seconds limit")

                    all_chunks.append(processed_chunk)
                    chunk_time = time.time() - chunk_start
                    print(f"Chunk {chunk_idx + 1} completed in {chunk_time:.2f} seconds")
                    chunk_idx += 1
                    
                except ChunkTimeoutError as e:
                    error_msg = str(e)
                    if all_chunks:
                        print(f"Timeout occurred. Saving {chunk_idx} chunks...")
                        final_audio = torch.cat(all_chunks, dim=1)
                        torchaudio.save(output_path, final_audio, target_sr)
                        error_msg += f". Partial output ({chunk_idx} chunks) saved to {output_path}"
                    raise RuntimeError(error_msg)

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

    finally:
        if optimize_memory:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

# Example usage:
"""
result = process_audio_with_timeout(
    model=model,
    mode="cross_lingual",
    text=your_text,
    prompt_speech_16k=your_prompt_speech,
    max_chunk_time=50,
    output_path="path/to/output.wav",
    use_half_precision=True,  # Enable FP16
    optimize_memory=True      # Enable memory optimizations
)
"""