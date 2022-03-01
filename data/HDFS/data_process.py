import pandas as pd 
import torch
from transformers import BertTokenizer, BertForPreTraining
import json

device = 'cuda'
seq_length = 60
cls_hidden = 64
train_ratio = 0.3
valid_ratio = 0.1

df_template = pd.read_csv('HDFS.log_templates.csv')
df_sequence = pd.read_csv('hdfs_sequence.csv')
df_label = pd.read_csv('anomaly_label.csv')
df_sequence = df_sequence.merge(df_label, on = 'BlockId')

hashed_index = {}
hashed_index["[SEQ]"] = 1
hashed_index["[PAD]"] = -1

for i in range(df_template.shape[0]):
	hashed_index[df_template.values[i][0]] = len(hashed_index)

json.dump(hashed_index,open('hashes.txt', 'w'))

df_sequence['EventSequence'] = df_sequence['EventSequence'].str[1:-1].str.split(", ")
df_sequence['EventSequence'] = df_sequence['EventSequence'].apply(lambda row: row[:seq_length] + ['[PAD]']*max(0,seq_length - len(row)))

print(df_sequence['EventSequence'][0])

df_normal_sequence = df_sequence[df_sequence['Label'] == 'Normal'].sample(frac = 1, random_state = 42)
df_anomaly_sequence = df_sequence[df_sequence['Label'] == 'Anomaly'].sample(frac = 1, random_state = 42)

train_size = int(df_normal_sequence.shape[0] * train_ratio)
valid_size = int(df_normal_sequence.shape[0] * (train_ratio + valid_ratio))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertForPreTraining.from_pretrained('bert-base-cased', output_hidden_states = True)

template_content = {}
template_embedding = {}
for i in range(df_template.values.shape[0]):
	template_content[df_template.values[i][0]] = tokenizer("".join(df_template.values[i][1].split("<*>")), padding="max_length", truncation=True).input_ids

seq_token = torch.rand(cls_hidden)
bert_model = bert_model.to(device)

bert_model.eval()

template_embedding['[SEQ]'] = seq_token
template_embedding['[PAD]'] = torch.zeros(cls_hidden)
for i,x in enumerate(template_content):
	cls_output = bert_model.forward(torch.tensor(template_content[x], dtype = torch.long).unsqueeze(0).to(device))['hidden_states'][-1][:,0,:].squeeze(0).detach().to('cpu')
	template_embedding[x] = cls_output
torch.save(template_embedding, 'template_embeddings.pt')


# template_embedding = torch.load('template_embeddings.pt')
# print(template_embedding.keys(), len(template_embedding))

# normal_cls_sequence = [torch.zeros(seq_length+1, cls_hidden) for x in range(len(df_normal_sequence.shape[0]))]
# for i in range(len(normal_cls_sequence)):
# 	print(i)
# 	normal_cls_sequence[i][0,:] = seq_token
# 	e = torch.tensor([template_content[y] for y in df_normal_sequence['EventSequence'].values[i][2:-2].split("', '")], dtype = torch.long).to(device)
# 	# e = normal_data_sequence[i].to(device)
# 	for idx in range(seq_length):
# 		if (idx >= e.size()[0]):
# 			continue
# 		cls_input = e[idx,:].unsqueeze(0)
# 		cls_output = bert_model.forward(cls_input)
# 		normal_cls_sequence[i][idx+1,:] = cls_output['hidden_states'][-1][:,0,:].squeeze(0).detach().to('cpu')

# anomaly_cls_sequence = [torch.zeros(seq_length+1, cls_hidden) for x in range(len(df_anomaly_sequence.shape[0]))]
# for i in range(len(anomaly_cls_sequence)):
# 	print(i)
# 	anomaly_cls_sequence[i][0,:] = seq_token
# 	e = torch.tensor([template_content[y] for y in df_anomaly_sequence['EventSequence'].values[i][2:-2].split("', '")], dtype = torch.long).to(device)
# 	# e = anomaly_data_sequence[i].to(device)
# 	for idx in range(seq_length):
# 		if (idx >= e.size()[0]):
# 			continue
# 		cls_input = e[idx,:].unsqueeze(0)
# 		cls_output = bert_model.forward(cls_input)
# 		anomaly_cls_sequence[i][idx+1,:] = cls_output['hidden_states'][-1][:,0,:].squeeze(0).detach().to('cpu')

print(train_size, valid_size)
df_normal_sequence[:train_size].to_csv('train_sequence.csv', index = False)
df_normal_sequence[train_size:valid_size].to_csv('valid_sequence.csv', index = False)
df_normal_sequence[valid_size:].to_csv('test_normal_sequence.csv', index = False)
df_anomaly_sequence.to_csv('test_anomaly_sequence.csv', index = False)

# print(len(normal_cls_sequence), len(anomaly_cls_sequence))
# torch.save(normal_cls_sequence[:train_size], 'train_cls_sequence.pt')
# torch.save(normal_cls_sequence[train_size:valid_size], 'valid_cls_sequence.pt')
# torch.save(normal_cls_sequence[valid_size:], 'test_normal_cls_sequence.pt')
# torch.save(anomaly_cls_sequence, 'test_anomaly_cls_sequence.pt')
# a = torch.load('train_cls_sequence.pt')
# print(len(a), a[0].size())
# print(cls_sequence[0].size())