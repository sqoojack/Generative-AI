import evaluate
from Dataset import training_dataset, tokenizer
from Train import model

""" Initialize metrics """
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# prepare predictions and references for evaluation
predictions = []
references = []

""" Use a subset of training data for evaluation (first 50 examples) """
for example in training_dataset[:50]:
    intro = example["introduction"]
    ref_abstract = example["abstract"]

    prompt = f"Introduction:\n{intro}\nAbstract:\n"     # generate prediction using the fine-tuned model 
    inputs = tokenizer(prompt, return_tensors="pt")     # use tokenizer to turn prompt into Pytorch tensor
    inputs = inputs.to(model.device)

    output_tokens = model.generate(**inputs, max_length=300, num_beams=4, early_stopping=True)
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)   # this generated_text will include the prompt + predicted abstract
    if "Abstract:" in generated_text:
        pred_abstract = generated_text.split("Abstract:")[1].strip()
    else:
        pred_abstract = generated_text
    predictions.append(pred_abstract)
    references.append(ref_abstract)

""" Compute ROUGE scores """
rouge_result = rouge.compute(predictions=predictions, references=references)
print("Rouge Score: ", rouge_result)

""" Comput Bert Score """
bertscore_result = bertscore.comput(predictions=predictions, references=references, lang="en")
print("Bert Score: (P, R, F) ",
    sum(bertscore_result["precision"]) / len(predictions),
    sum(bertscore_result["recall"]) / len(predictions),
    sum(bertscore_result["f1"]) / len(predictions))