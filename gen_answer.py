import json
import os
import time
import concurrent.futures

import tiktoken
import shortuuid
import tqdm

from utils import (
    load_questions,
    load_model_answers,
    make_config,
    get_endpoint,
    chat_completion_openai,
    chat_completion_yandex,
    chat_completion_gigachat,
    chat_completion_anthropic,
    chat_completion_openai_azure,
    chat_completion_mistral,
    chat_completion_gemini,
    chat_completion_cohere,
    reorg_answer_file,
    OPENAI_MODEL_LIST,
    temperature_config,
)


class AnswerGenerator:
    def __init__(self, generation_path, endpoint_file):
        self.settings = make_config(os.path.join(generation_path, "config", "gen_answer_config.yaml"))
        self.endpoint_list = make_config(endpoint_file)
        self.generation_path = generation_path
        self.existing_answer = load_model_answers(generation_path)
        self.questions = load_questions(os.path.join("data", self.settings["bench_name"], "question.jsonl"))

    def get_answer(self, question: dict, model: str, endpoint_info: dict, max_tokens: int, answer_file: str, api_dict: dict):
        temperature = self.settings["temperature"],
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]

        api_type = endpoint_info["api_type"]

        conv = []

        if "system_prompt" in endpoint_info.keys():
            conv.append({"role": "system", "content": endpoint_info["system_prompt"]})
        elif model in OPENAI_MODEL_LIST:
            conv.append({"role": "system", "content": "You are a helpful assistant."})

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        choices = []
        for i in range(self.settings["num_choices"]):
            turns = []
            for j in range(len(question["turns"])):
                conv.append({"role": "user", "content": question["turns"][j]["content"]})
                if api_type == "anthropic":
                    output = chat_completion_anthropic(model=endpoint_info["model_name"],
                                                    messages=conv,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens)
                elif api_type == "mistral":
                    output = chat_completion_mistral(model=endpoint_info["model_name"],
                                                    messages=conv,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens)
                elif api_type == "yandex":
                    output = chat_completion_yandex(model=endpoint_info["model_name"],
                                                    messages=conv,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens,
                                                    api_dict=api_dict)
                elif api_type == "gigachat":
                    output = chat_completion_gigachat(model=endpoint_info["model_name"],
                                                    messages=conv,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens,
                                                    api_dict=api_dict)
                elif api_type == "gemini":
                    output = chat_completion_gemini(model=endpoint_info["model_name"],
                                                    messages=question["turns"][j]["content"],
                                                    temperature=temperature,
                                                    max_tokens=max_tokens)
                elif api_type == "azure":
                    output = chat_completion_openai_azure(model=endpoint_info["model_name"],
                                                        messages=conv,
                                                        temperature=temperature,
                                                        max_tokens=max_tokens,
                                                        api_dict=api_dict)
                elif api_type == "cohere":
                    output = chat_completion_cohere(model=endpoint_info["model_name"],
                                                    messages=conv,
                                                    temperature=temperature,
                                                    max_tokens=max_tokens)
                else:
                    output = chat_completion_openai(model=endpoint_info["model_name"], 
                                                    messages=conv, 
                                                    temperature=temperature, 
                                                    max_tokens=max_tokens, 
                                                    api_dict=api_dict)
                conv.append({"role": "assistant", "content": output})

                turns.append({"content": output, "token_len": len(encoding.encode(output))})
            choices.append({"index": i, "turns": turns})

        # Dump answers
        ans = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model,
            "choices": choices,
            "tstamp": time.time(),
        }

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(answer_file, "a") as fout:
            fout.write(json.dumps(ans) + "\n")

    def generate(self):
        for model in self.settings["model_list"]:
            assert model in self.endpoint_list

            endpoint_info = self.endpoint_list[model]
            answer_file = os.path.join(self.generation_path, f"{model}.jsonl")

            print(f"Output to {answer_file}")

            if "parallel" in endpoint_info:
                parallel = endpoint_info["parallel"]
            else:
                parallel = 1

            # We want to maximizes the number of tokens generate per answer: max_tokens = specified token # - input tokens #
            if "tokenizer" in endpoint_info:
                question_list = [question["turns"][0]["content"] for question in self.questions]
                if model in OPENAI_MODEL_LIST:
                    tokenizer = tiktoken.encoding_for_model(endpoint_info["model_name"])
                    tokens = [tokenizer.encode(prompt) for prompt in question_list]
                    max_tokens = [(self.settings["max_tokens"] - len(token) - 100) for token in tokens]
                else:
                    from transformers import AutoTokenizer
                    
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                    tokenizer = AutoTokenizer.from_pretrained(endpoint_info["tokenizer"])

                    tokens = tokenizer(question_list)
                    max_tokens = [(self.settings["max_tokens"] - len(prompt) - 300) for prompt in tokens["input_ids"]]
            else:
                max_tokens = [self.settings["max_tokens"]] * len(self.questions)

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = []
                count = 0
                for index, question in enumerate(self.questions):
                    if model in self.existing_answer and question["question_id"] in self.existing_answer[model]:
                        count += 1
                        continue
                    future = executor.submit(
                        self.get_answer,
                        question,
                        model,
                        endpoint_info,
                        max_tokens[index],
                        answer_file,
                        get_endpoint(endpoint_info["endpoints"]),
                    )
                    futures.append(future)
                if count > 0:
                    print(f"{count} number of existing answers")
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures), total=len(futures)
                ):
                    future.result()

            reorg_answer_file(answer_file)

    def __str__(self):
        return f"\n{self.settings}\n"
