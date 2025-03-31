from Dataset import test_data, tokenizer
from Train import model
import json

submission = []
for example in test_data:
    paper_id = example["paper_id"]
    intro = example["introduction"]
    prompt = f"Introduction:\n{intro}\nAbstract:\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to(model.device)
    output_tokens = model.generate(**inputs, max_length=300, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    if "Abstract:" in generated_text:
        pred_abstract = generated_text.split("Abstract:")[1].strip()
    else:
        pred_abstract = generated_text
    submission.append({
        "paper_id": paper_id,
        "abstract": pred_abstract 
    })

output_file = f"313552049.json"

with open(output_file, "w", encoding="utf-8") as f:
    for entry in submission:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")