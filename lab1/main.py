import csv
from google import genai
from google.genai import types
import os 
from dotenv import load_dotenv
import time
import random   # used to refer the random example 
from collections import Counter

def get_prompt_from_file(file_path, delimiter):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    sections = data.split(delimiter)
    prompt = sections[1].strip()
    return prompt

def read_csv_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:  # add to data row by row
            data.append(row)
    return data

def call_API(prompt, api_key):
    client=genai.Client(api_key=api_key)
    response=client.models.generate_content(
        model='gemini-2.0-flash', 
        # model='gemini-2.0-lite',
        
        contents=prompt,
        
        config=types.GenerateContentConfig(
            max_output_tokens= 10,
            top_k= 1,   # The lower the model, the more stable it is.
            top_p= 0.1, # The more closed to zero, the more stable it is. 
            temperature= 0.1,   # The same to top_p
            response_mime_type= 'text/plain',
            stop_sequences=['\n', ' ', '.', ':'],
        ),
    )
    return response.text

def get_answer(prompt, api_key, num_calls):
    response = []   # used to store the all answer
    for i in range(num_calls):
        answer = call_API(prompt, api_key)
        response.append(answer)
        # time.sleep(1)
    counter = Counter(response) # summerize the frequency of each answer
    most_common_answer, count = counter.most_common(1)[0]   # pick the most 1 common element in Counter result, and return a list (tuple [element, times])
    return most_common_answer

def output_predictions_to_csv(Qids, answers, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:     # newline: avoid to write extra newline(換行) in different system
        writer = csv.writer(f)  # create a CSV write tool 
        writer.writerow(["ID", "Target"])  # write header row
        for Qid, answer in zip(Qids, answers):
            writer.writerow([Qid, answer])

def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")

    base_prompt = get_prompt_from_file("prompt.txt", delimiter = '---')

    csv_data = read_csv_data("mmlu_submit.csv")

    Qids = []   # used to store the output
    answers = []

    # batch_size = 10
    batch_size = 20
    count = 0   # to display the current iteration
    for i in range(0, len(csv_data), batch_size):
        batch = csv_data[i:i+batch_size]
        for row in batch:    # address the data row by row
            qid = row.get("Unnamed: 0", "")
            question = row.get("input", "")
            option_a = row.get("A", "")
            option_b = row.get("B", "")
            option_c = row.get("C", "")
            option_d = row.get("D", "")
            task = row.get("task", "")

            complete_prompt = (base_prompt
                    .replace("{{question}}", question)
                    .replace("{{A}}", option_a)
                    .replace("{{B}}", option_b)
                    .replace("{{C}}", option_c)
                    .replace("{{D}}", option_d)
                    .replace("{{task}}", task))

            count += 1
            print(f"Now is at {count} iterative.")
            answer = get_answer(complete_prompt, api_key, 5)
            Qids.append(qid)
            answers.append(answer)
            print(answer)
        # time.sleep(52)   # to avoid resource exhausted
        time.sleep(8)
    
    output_predictions_to_csv(Qids, answers, "submit_format.csv")
    print("Output succeed!")

if __name__ == "__main__":
    main()