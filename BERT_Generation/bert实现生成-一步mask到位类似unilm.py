"""

@file  : bert实现生成-一步mask到位类似unilm.py

@author: xiaolu

@time  : 2020-03-24

"""
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
import torch


input_text = "[CLS] I go to school by bus [SEP] "
target_text = "我搭公車上學"

tokenizer = BertTokenizer.from_pretrained('./vocab.txt')
model = BertForMaskedLM.from_pretrained('Bert_Generation.tar.gz')

tokenized_text = tokenizer.tokenize(input_text)

for i in target_text:
    tokenized_text.append('[MASK]')
for _ in range(128-len(tokenized_text)):
    tokenized_text.append('[MASK]')
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)


tokens_tensor = torch.tensor([indexed_tokens])

loss_ids = [-1] * (len(tokenizer.tokenize(input_text)))
# loss_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_text)))
for i in target_text:
    loss_ids.append(tokenizer.convert_tokens_to_ids(i)[0])
loss_ids.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])

for _ in range(128-len(loss_ids)):
    loss_ids.append(-1)
loss_tensors = torch.tensor([loss_ids])

print(tokens_tensor, loss_tensors)
print(tokenizer.convert_ids_to_tokens(indexed_tokens))

optimizer = torch.optim.Adamax(model.parameters(), lr = 5e-5)

model.train()
for i in range(0, 10):
    loss = model(tokens_tensor,masked_lm_labels=loss_tensors)
    eveloss = loss.mean().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("step "+ str(i) + " : " + str(eveloss))


model.eval()
with torch.no_grad():
    predictions = model(tokens_tensor)
    start = len(tokenizer.tokenize(input_text))
    while start < len(predictions[0]):
        predicted_index = torch.argmax(predictions[0,start]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        if '[SEP]' in predicted_token:
            break
        print(predicted_token)
        start += 1

