windows环境安装：
```
conda install libuv
pip install dpcpp-cpp-rt==2024.2 mkl-dpcpp==2024.2
```
### For Intel® Arc™ A-Series Graphics, use the commands below:
```
python -m pip install torch==2.1.0.post3 torchvision==0.16.0.post3 torchaudio==2.1.0.post3 intel-extension-for-pytorch==2.1.40+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```
### For Intel® Core™ Ultra Processors with Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (MTL-H), use the commands below:
```
python -m pip install torch==2.1.0.post3 torchvision==0.16.0.post3 torchaudio==2.1.0.post3 intel-extension-for-pytorch==2.1.40+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/mtl/cn/
pip install funasr #  我这边是1.0.27版本
```

env\Lib\site-packages\funasr\auto\auto_model.py 需要注释一下
![image](https://github.com/user-attachments/assets/2e4b75d7-47b0-47c8-add7-2f6da5294296)


测试代码（模型需要先提前下载好，放在models目录下）：
python funasr_gpu.py
