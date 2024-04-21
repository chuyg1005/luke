conda create -n chuyg-luke python==3.8.13
source activate chuyg-luke # 使用source而不是使用 conda，原因是source可以在shell脚本中使用

pip install poetry==1.1.11

poetry install
# 需要安装 nltk的一个工具
python -m nltk.downloader omw-1.4