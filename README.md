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
    ```python
        from sklearn.model_selection import StratifiedKFold
        train_path = os.path.join("../data", 'train.csv')
        train_input_ids, train_attention_mask, train_token_type_ids, y_train = load_data(train_path, tokenizer)
        # print(train_input_ids.size())    # torch.Size([24999, 512])   # (样本个数, input_dim)
        # print(y_train.size())      # torch.Size([24999])
    
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
        for train_index, val_index in kfold.split(train_input_ids, y_train):
            # print(len(train_index))   # 19999
            # print(len(val_index))   # 5000
            train_data = TensorDataset(train_input_ids[train_index], train_attention_mask[train_index], train_token_type_ids[train_index], y_train[train_index])
            train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
            dev_data = TensorDataset(train_input_ids[val_index], train_attention_mask[val_index], train_token_type_ids[val_index], y_train[val_index])
            dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)
    
            for epoch in range(args.num_epochs):
                model.train()
                for step, (cur_input_ids, cur_attention_mask, cur_token_type_ids, cur_y) in enumerate(train_loader):
                    # print(cur_input_ids.size())    # torch.Size([24, 512])
                    output = model(cur_input_ids, cur_attention_mask, cur_token_type_ids)
                    loss = loss.fct(output, cur_y)
                    loss.backward()
                    optimizer.step()    
                evaluate(model, dev_loader)
    ```

2. 计算模型的计算量和参数量。首先安装库thop: pip install thop。
    ```python
    import torch
    from thop import profile
    from thop import clever_format
    from torchvision.models import resnet50
    
    
    def get_pf(model,input):
        flops, params = profile(model, inputs=(input,))
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    
    
    if __name__ == '__main__':
        model = resnet50()
        inputs = torch.rand(4, 3, 256, 128)
        flops, params = get_pf(model, inputs)
        print(flops)    # 浮点运算数: 10.743G
        print(params)     # 参数量: 25.557M
    
    ```
    - flops通常代表的是浮点运算数，衡量的是算法或者模型的复杂度，计算量小代表计算过程中需要的计算空间较小。
    - 参数量就是指模型各个部分的参数数量之和，参数小指模型的参数少，比较轻量级


3. CrossEntropyLoss和NLLLoss  
    - 最常见的错误是损失函数和输出激活函数之间的不匹配。nn.CrossEntropyLoss中的损失模块执行两个操作：nn.LogSoftmax和nn.NLLLoss。因此nn.CrossEntropyLoss的输入应该是最后一个线性层的输出。不要在nn.CrossEntropyLossPyTorch之前应用Softmax。 否则将对Softmax输出计算log-softmax，将会降低模型精度。
    
    - 如果使用nn.NLLLoss模块，则需要自己应用log-softmax。nn.NLLLoss需要对数概率，而不是普通概率。因此确保应用nn.LogSoftmax 或者 nn.functional.log_softmax，而不是nn.Softmax。

4. Softmax的计算维度  
    注意Softmax的计算维度。通常是输出张量的最后一个维度，例如nn.Softmax(dim=-1)。如果混淆了维度，模型最终会得到随机预测。

5. model.train()和model.eval()以及torch.no_grad()的区别  
    1. model.train()的作用是使Batch Normalization、Dropout生效。Batch Normalization采用的时一个批次的均值方差，Dropout按照概率随机选取神经元连接。

    2. model.eval()的作用是让Batch Normalization、Dropout无效。即:Batch Normalization使用全局的均值和方差，Dropout不丢弃神经元，使用全部的神经元。所以，model.eval()常用在模型进行验证或测试阶段。但是，这个模式不会影响梯度的计算(梯度照样计算)。
    
    3. torch.no_grad()则是停止梯度计算，以起到加速和节省显存的作用。但是，这个模式不会影响Normalization和Dropout。所以，通常是eval()和no_grad()搭配使用。

6. BCELoss和BCEWithLogitsLoss  
    简而言之，BCEWithLogitsLoss = Sigmoid + BCELoss  
    举个例子: 假设一个batch中有两个样本，预测的logits=[[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]], 真实的label=[[0, 1, 1],[0, 0, 1]]  
    
    则BCELoss的计算如下:    
    第一个样本:  
    0 × In 0.3992 + (1-0) × In(1-0.3992) = -0.5095  
    1 × In 0.2232 + (1-1) × In(1-0.2232) = -1.4997  
    1 × In 0.6435 + (1-1) × In(1-0.6435) = -0.4408  
    第二个样本:   
    0 × In 0.3800 + (1−0) × In(1−0.3800) = −0.4780  
    0 × In 0.3044 + (1−0) × In(1−0.3044) = −0.3630     
    1 × In 0.3241 + (1−1) × In(1−0.3241) = −1.1267   
    去掉符号对每个样本求均值: 第一个样本: (0.5095 + 1.4997 + 0.4408) / 3 = 0.8167； 第二个样本: (0.4780 + 0.3630 + 1.1267) / 3 = 0.6559  
    最后再平均: (0.8167 + 0.6559) / 2 = 0.7363  
    
    使用代码验证:  
    ```python
    from torch import nn
    loss_fct = nn.BCELoss()
    input = torch.tensor([[0.3992, 0.2232, 0.6435],[0.3800, 0.3044, 0.3241]])
    label = torch.tensor([[0, 1, 1],[0, 0, 1]], dtype=torch.float)
    loss = loss_fct(input, label)
    print(loss)
    # 0.7363
    ```
    
    同理，可验证BCEWithLogitsLoss  
7. 安装apex    
    - git clone https://github.com/NVIDIA/apex
    - cd apex
    - python3 setup.py install
