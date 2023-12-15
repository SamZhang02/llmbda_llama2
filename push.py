from transformers import AutoModel, AutoTokenizer
import torch
torch.device('cuda')


print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("./models/fine_tuned_llama2_7b")
print("tokenizer loaded")

print("loading model")
model = AutoModel.from_pretrained("./models/fine_tuned_llama2_7b")
print("model loaded")

model.push_to_hub('llama2-7b-mcgill-requisites')
