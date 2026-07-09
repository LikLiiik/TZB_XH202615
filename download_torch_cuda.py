#!/usr/bin/env python3
"""
Robust file downloader with resume support for large files.
"""
import os
import sys
import time
import hashlib
import urllib.request

def download_with_resume(url, dest, max_retries=100, chunk_size=1024*1024):
    """
    Download a file with resume support and retry logic.
    Returns True on success.
    """
    os.makedirs(os.path.dirname(dest) if os.path.dirname(dest) else '.', exist_ok=True)

    existing_size = 0
    if os.path.exists(dest):
        existing_size = os.path.getsize(dest)

    retry = 0
    while retry < max_retries:
        try:
            # Get remote file size
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as resp:
                total_size = int(resp.headers.get('Content-Length', 0))

            if existing_size >= total_size and total_size > 0:
                print(f"文件已完整下载: {dest} ({total_size/1024**3:.2f} GB)")
                return True

            # Download with range request for resume
            req = urllib.request.Request(url)
            if existing_size > 0:
                req.add_header('Range', f'bytes={existing_size}-')
                print(f"续传: {existing_size/1024**3:.2f}GB / {total_size/1024**3:.2f}GB")
            else:
                print(f"开始下载: {total_size/1024**3:.2f}GB")

            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = 'ab' if existing_size > 0 else 'wb'
                with open(dest, mode) as f:
                    downloaded = existing_size
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = 100 * downloaded / total_size
                            print(f"\r  进度: {downloaded/1024**3:.2f}/{total_size/1024**3:.2f} GB ({pct:.1f}%)", end='', flush=True)

            print()

            # Verify
            final_size = os.path.getsize(dest)
            if total_size > 0 and final_size == total_size:
                print(f"下载完成! ({final_size/1024**3:.2f} GB)")
                return True
            else:
                print(f"大小不匹配: {final_size} vs {total_size}, 重试...")
                existing_size = final_size
                retry += 1

        except Exception as e:
            retry += 1
            existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"\n[重试 {retry}/{max_retries}] 错误: {e}")
            time.sleep(min(30, retry * 2))

    print(f"下载失败，已重试 {max_retries} 次")
    return False


if __name__ == "__main__":
    # Download PyTorch CUDA wheel
    url = "https://download.pytorch.org/whl/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl"
    dest = "E:/qwen-asr/tmp/torch-2.5.1+cu121-cp311-cp311-win_amd64.whl"

    print(f"目标: {dest}")
    success = download_with_resume(url, dest)

    if success:
        print("\n接下来运行:")
        print(f"  pip install {dest}")
    else:
        print("\n下载失败，请检查网络连接")
    sys.exit(0 if success else 1)
