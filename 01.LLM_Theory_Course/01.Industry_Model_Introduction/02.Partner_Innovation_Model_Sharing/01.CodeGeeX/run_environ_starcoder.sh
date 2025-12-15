#!/bin/bash  
# Create Python3.9 virtual environment
/home/ma-user/anaconda3/bin/conda create -n python-3.9.0 python=3.9.0 -y --override-channels --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
# Activate
source /home/ma-user/anaconda3/bin/activate python-3.9.0
# Install mindspore
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.14/MindSpore/unified/x86_64/mindspore-2.2.14-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

# Clone repository
git clone https://github.com/mindspore-lab/mindnlp

# Install dependencies
cd mindnlp/llm/inference/starcoder
pip install -r requirements.txt
wget https://mindspore-demo.obs.cn-north-4.myhuaweicloud.com/mindnlp_install/mindnlp-0.3.1-py3-none-any.whl
pip install mindnlp-0.3.1-py3-none-any.whl
pip install tokenizers==0.15.2