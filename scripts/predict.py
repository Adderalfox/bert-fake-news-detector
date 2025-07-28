import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "checkpoints/best_model"
TOKENIZER_DIR = "checkpoints/best_tokenizer"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)

model.eval()

label_map = {0: "FAKE", 1: "REAL"}

def predict(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    return label_map[predicted_class]

if __name__ == "__main__":
    print("Fake News Detector")
    print("--------------------")
    while True:
        input_text = input("\nEnter news article (or type 'exit' to quit):\n> ")
        if input_text.lower() == "exit":
            break
        result = predict(input_text)
        print(f"\nPrediction: {result}")
