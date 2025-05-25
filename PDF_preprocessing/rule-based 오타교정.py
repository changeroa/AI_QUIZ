from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
model = BartForConditionalGeneration.from_pretrained("hyunwoongko/kobart")

input_text = "나는 어제 도서관에 가따."
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("=== 출력 시작 ===")
print(output)
print("=== 출력 끝 ===")
