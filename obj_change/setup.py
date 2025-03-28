#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動的物体検出とマスク処理システムの環境セットアップスクリプト
"""

import os
import sys
import subprocess
import pkg_resources

def check_and_install_packages():
    """
    必要なパッケージがインストールされているか確認し、不足しているものをインストールする
    """
    required_packages = [
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "transformers>=4.20.0",
        "tqdm>=4.60.0",
        "faiss-gpu>=1.7.0",  # GPUが使用できない場合は faiss-cpu に変更
        "Pillow>=8.0.0",
        "ultralytics>=8.0.0",  # FastSAMのために必要
        "timm>=0.6.0",  # Depth Anything V2のために必要
    ]
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = []
    
    for package in required_packages:
        pkg_name = package.split(">=")[0]
        if pkg_name not in installed:
            missing.append(package)
    
    if missing:
        print("以下のパッケージがインストールされていません。インストールを開始します：")
        for pkg in missing:
            print(f"  - {pkg}")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("パッケージのインストールが完了しました。")
        except Exception as e:
            print(f"パッケージのインストールに失敗しました: {e}")
            print("手動でインストールを行ってください：")
            print(f"pip install {' '.join(missing)}")
            sys.exit(1)
    else:
        print("すべての必要なパッケージがインストールされています。")

def check_cuda_availability():
    """
    CUDAが利用可能かどうかを確認する
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA利用可能: {torch.cuda.get_device_name(0)}")
            print(f"GPUメモリ: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("警告: CUDAが利用できません。CPUモードで実行されます（処理速度が遅くなります）。")
    except:
        print("警告: PyTorchからCUDA情報を取得できませんでした。")

def check_models():
    """
    必要なモデルファイルが存在するか確認する
    """
    # FastSAM-x.ptのパス
    fastsam_paths = [
        "FastSAM-x.pt",
        "../FastSAM-x.pt",
        "../FastSAM/FastSAM-x.pt",
        "FastSAM/FastSAM-x.pt",
    ]
    
    for path in fastsam_paths:
        if os.path.exists(path):
            print(f"FastSAMモデルが見つかりました: {os.path.abspath(path)}")
            break
    else:
        print("警告: FastSAM-x.ptが見つかりません。プログラム実行時に自動的にダウンロードされます。")
    
    # Depth Anything V2は初回実行時に自動ダウンロードされる
    print("Depth Anything V2モデルは初回実行時に自動的にダウンロードされます。")

def create_output_dir():
    """
    出力ディレクトリを作成する
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"出力ディレクトリを作成しました: {output_dir}")

def main():
    print("動的物体検出とマスク処理システムのセットアップを開始します...")
    
    # パッケージのチェックとインストール
    check_and_install_packages()
    
    # CUDAの確認
    check_cuda_availability()
    
    # モデルの確認
    check_models()
    
    # 出力ディレクトリの作成
    create_output_dir()
    
    print("\nセットアップが完了しました。以下のコマンドでテストを実行できます：")
    print("python obj_change/run_test.py")

if __name__ == "__main__":
    main() 