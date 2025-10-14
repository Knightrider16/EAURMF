from transformers import BertModel, BertTokenizer

model_name = "bert-base-uncased"

# Download and save both model and tokenizer
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.save_pretrained("./prebert")
tokenizer.save_pretrained("./prebert")