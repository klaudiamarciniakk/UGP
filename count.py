import json

import torch

from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, \
    AutoModelForSeq2SeqLM

MAP_LABEL_TRANSLATION = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


def generate_text_simple(model_pipeline, text: str, max_new_tokens: int = 20, is_prompt: bool = False):
    generated_text = model_pipeline(text, do_sample=False, max_new_tokens=max_new_tokens)[0]["generated_text"]
    if is_prompt and generated_text.startswith(text):
        generated_text = generated_text[len(text):].strip()
    return generated_text


def get_pipeline(pipeline_type: str, model_name: str, model_type: str, torch_dtype: torch.dtype = "auto",
                 device_map="auto"):
    if model_type == 'clm':
        class_type = AutoModelForCausalLM
    elif model_type == 'mlm':
        class_type = AutoModelForMaskedLM
    elif model_type == 's2s':
        class_type = AutoModelForSeq2SeqLM
    model = class_type.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch_dtype,
                                       device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return pipeline(pipeline_type, model=model, tokenizer=tokenizer)


def count_json_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return len(lines)


def main(json_file):
    equal_count = 0
    lm_pipeline = get_pipeline('text2text-generation', 'google/flan-t5-large', 's2s')
    with open(json_file, 'r') as file:

        for line in file:
            json_object = json.loads(line)
            question = "Categorize this sentence with this options : joy, sadness, love, anger, fear, surprise: "
            text = question + json_object['text']
            label = json_object['label']
            emotion = MAP_LABEL_TRANSLATION[label]
            generated_result = generate_text_simple(lm_pipeline, text)
            if generated_result == emotion:
                equal_count += 1

    print(f"Ilość poprawnych odpowiedzi: {equal_count}")
    print(f"Accuracy: {equal_count / count_json_lines(json_file)}")


def example(question):
    lm_pipeline = get_pipeline('text2text-generation', 'google/flan-t5-large', 's2s')
    print(generate_text_simple(lm_pipeline, question))


if __name__ == "__main__":
    json_file_path = "data/test-5k.json"
    main(json_file_path)
    example(
        "Categorize this sentence with this options : joy, sadness, love, anger, fear, surprise: i woke up feeling fine")
    example(
        "Categorize this sentence with this options : joy, sadness, love, anger, fear, surprise: i now feel compromised and skeptical of the value of every unit of work i put in")
    example(
        "Categorize this sentence with this options : joy, sadness, love, anger, fear, surprise: i have seen heard and read over the past couple of days i am left feeling impressed by more than a few companies")
