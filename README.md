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

