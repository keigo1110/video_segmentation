from setuptools import setup, find_packages
import os

# requirements.txt から依存関係を読み込むヘルパー関数
def parse_requirements(filename):
    """ requirements.txt ファイルを解析して依存関係リストを返す """
    lines = (line.strip() for line in open(filename))
    # コメント行と空行を除外し、torch関連は一旦除外 (環境依存のため)
    # ultralytics や mediapipe は含める
    requirements = [line for line in lines if line and not line.startswith("#") and not line.startswith("torch")]
    return requirements

# README.md の内容を long_description として読み込む
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "DynaMask2: Dynamic object masking tool considering camera movement."

setup(
    name="dynamask",
    version="0.1.0", # 初期バージョン
    author="Your Name / Lab Name", # あなたの名前または研究室名に変更してください
    author_email="your.email@example.com", # あなたのメールアドレスに変更してください
    description="Dynamic object masking tool considering camera movement, using FastSAM, Optical Flow, and Human Detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/DynaMask2", # あなたのGitHubリポジトリURLなどに変更してください
    packages=find_packages(where=".", include=["dynamask", "dynamask.*"], exclude=["tests", "docs", "examples"]),
    # package_dir={'': '.'}, # ルートをパッケージディレクトリとして指定 (通常不要)
    include_package_data=False, # パッケージデータは今のところ無し
    install_requires=parse_requirements("requirements.txt"),
    entry_points={
        'console_scripts': [
            'dynamask=run_dynamask_cli:main', # dynamask コマンドで run_dynamask_cli.py の main を実行
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # ライセンスに合わせて変更してください (例: Apache-2.0)
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8", # 必要なPythonバージョン
)
