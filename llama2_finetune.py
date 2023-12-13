import pandas as pd 
import argparse 

from transformers import Autotokenizer
import transformers
import torch
torch.device('cuda')



# - Experiments

parser = argparse.ArgumentParser()

parser.add_argument('--path')
parser.add_argument('--model')

args = parser.parse_args()

test_data_path = "./test.csv" if not args.path else args.path
model = "meta-llama/Llama-2-7b-chat-hf" if not args.model else args.model

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
            "text-generation",
                model=model,
                    torch_dtype=torch.float16,
                        device_map="auto",
                        )

data = load_data(test_data_path)

indices = data["index"].to_list()
requisites = data["requisite"].to_list()

pred = [test(req) for req in requisites]

pred_df = pd.DataFrame({
        "index": indices,
            "prediction": pred
            })

pred_df.to_csv("./llama2_chat_pred.csv")


