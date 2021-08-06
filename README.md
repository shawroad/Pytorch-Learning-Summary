# Pytorch-Learning-Summary
### 一些报错的解决方案
1. 加载模型的权重时，有时会发现加载报错。显示当前模型是有这些参数的，但是你的权重没有这些参数。这里只需要加一个参数: strict=False。
```python
model.load_state_dict(torch.load('xxx.bin'), strict=False) 
```
2. 高版本的torch训练的模型用低版本的torch加载时，报错。说是什么from_tf啥的。 不用理。 直接用高版本的torch重新加载模型，然后保存的时候加一个参数: _use_new_zipfile_serialization=False
```python
import torch
from transformers import AutoConfig, AutoModel

config = AutoConfig.from_pretrained('./config.json')
model = AutoModel.from_pretrained('./pytorch_model.bin', config=config)

torch.save(model.state_dict(), 'pytorch_model_new.bin', _use_new_zipfile_serialization=False)
```

### 一些技巧的总结
1. 在pytorch中借助sklearn实现k折交叉验证


