#!/usr/bin/env python3
"""
Demo script for aitllm streaming inference. 
"""
import argparse
import torch
from aitllm import StreamingEngine

def main():
    parser = argparse.ArgumentParser(description="aitllm inference demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in simple terms.",
                        help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    
    args = parser.parse_args()
    
    print(f"[aitllm] Initializing streaming engine...")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    engine = StreamingEngine(
        model_path=args.model,
        max_length=512,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print(f"\n[aitllm] Prompt: {args.prompt}")
    print(f"[aitllm] Generating...\n")
    print("-" * 80)
    
    output = engine.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        stream=True
    )
    
    print(output)
    print("-" * 80)
    
    # Print statistics
    stats = engine.get_stats()
    print(f"\n[aitllm] Statistics:")
    print(f"  Tokens Generated: {stats['tokens_generated']}")
    print(f"  Time Elapsed: {stats['time_elapsed']:.2f}s")
    print(f"  Tokens/sec: {stats['tokens_per_second']:.2f}")
    print(f"  Peak VRAM: {stats['peak_vram_gb']:.2f} GB")
    print(f"  Avg VRAM: {stats['avg_vram_gb']:.2f} GB")


if __name__ == "__main__": 
    main()