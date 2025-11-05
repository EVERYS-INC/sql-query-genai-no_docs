"""
GPU環境診断スクリプト
"""

import sys
import platform

print("=" * 60)
print("GPU環境診断")
print("=" * 60)

# システム情報
print("\n【システム情報】")
print(f"Python バージョン: {sys.version}")
print(f"プラットフォーム: {platform.platform()}")

# PyTorchの確認
try:
    import torch
    print(f"\n【PyTorch情報】")
    print(f"PyTorch バージョン: {torch.__version__}")
    print(f"CUDA利用可能: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA バージョン: {torch.version.cuda}")
        print(f"cuDNN バージョン: {torch.backends.cudnn.version()}")
        print(f"GPU数: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  名前: {torch.cuda.get_device_name(i)}")
            print(f"  メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    else:
        print("\n⚠️ CUDAが利用できません。")
        print("\n考えられる原因:")
        print("1. NVIDIA GPUが搭載されていない")
        print("2. NVIDIA GPUドライバがインストールされていない")
        print("3. PyTorchのCPU版がインストールされている")
        
        # CPU版かGPU版かを確認
        if "cpu" in torch.__version__ or "+cpu" in torch.__version__:
            print("\n❌ PyTorchのCPU版がインストールされています")
            print("GPU版をインストールする必要があります")
        
except ImportError:
    print("\n❌ PyTorchがインストールされていません")

# NVIDIA SMIの確認
print("\n【NVIDIA GPU情報】")
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 
                          capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print("nvidia-smi出力:")
        print(result.stdout)
    else:
        print("⚠️ nvidia-smiコマンドが実行できません")
        print("NVIDIAドライバがインストールされていない可能性があります")
except Exception as e:
    print(f"nvidia-smi実行エラー: {e}")

print("\n" + "=" * 60)
print("診断完了")
print("=" * 60)