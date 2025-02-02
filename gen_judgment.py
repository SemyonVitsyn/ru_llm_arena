import json
import os
import re
import concurrent.futures

from tqdm import tqdm
from pathlib import Path

from utils import (
    load_questions,
    load_questions,
    load_model_answers,
    make_config,
    get_score,
    get_answer
)


class Judge:
    def __init__(self, judgment_path, generation_path, endpoint_file):
        self.settings = make_config(os.path.join(judgment_path, "config", "judge_config.yaml"))
        self.endpoint_list = make_config(endpoint_file)
        if self.settings["regex_pattern"]:
            self.pattern = re.compile(self.settings["regex_pattern"])

        self.questions = load_questions(os.path.join("data", self.settings["bench_name"], "question.jsonl"))
        self.model_answers = load_model_answers(generation_path)

        self.models = [model for model in self.settings["model_list"]]
        
        self.ref_answers = None
        if self.settings["reference"]:
            self.ref_answers = load_model_answers(os.path.join("data", self.settings["bench_name"], "reference_answer"))
            self.ref_answers = [self.ref_answers[model] for model in self.settings["ref_model"]]
    
        self.output_dir = judgment_path
        self.existing_judgments = load_model_answers(self.output_dir)
        self.endpoint_info = self.endpoint_list[self.settings["judge_model"]]

    def judgment(self, question, answer, reference, baseline, output_file):
        settings = self.settings
        model = settings["judge_model"]

        num_games = 2 if settings["pairwise"] else 1

        output = {
            "question_id":question["question_id"],
            "model":answer["model_id"],
            "judge": model,
            "games":[]
            }

        for game in range(num_games):
            conv = [{"role": "system", "content": settings["system_prompt"]}]

            for template in settings["prompt_template"]:
                prompt_args = {}

                for i, turn in enumerate(question["turns"]):
                    prompt_args[f"question_{i+1}"] = turn["content"]
                base = 1

                if baseline:
                    if game % 2 == 1: # swap position
                        temp = baseline
                        baseline = answer
                        answer = temp

                    for i, turn in enumerate(baseline["choices"][0]["turns"]):
                        prompt_args[f"answer_{i+1}"] = turn["content"]
                        base += 1
                if answer:
                    for i, turn in enumerate(answer["choices"][0]["turns"]):
                        prompt_args[f"answer_{i+base}"] = turn["content"]

                if reference:
                    for j, ref_answer in enumerate(reference):
                        for i, turn in enumerate(ref_answer["choices"][0]["turns"]):
                            prompt_args[f"ref_answer_{i+j+1}"] = turn["content"]
                
                user_prompt = template.format(**prompt_args)
                conv.append({"role": "user", "content": user_prompt})

            judgment = ""
            for _ in range(2):
                new_judgment = get_answer(
                    model,
                    conv,
                    settings["temperature"],
                    settings["max_tokens"],
                    self.endpoint_info,
                )

                judgment += ("\n" + new_judgment)

                score, try_again = get_score(judgment, self.pattern)

                conv.append({"role": "assistant", "content": new_judgment})

                if not try_again:
                    break

                conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

            result = {
                "user_prompt": conv[1]["content"],
                "judgment": judgment,
                "score":score
            }
            output["games"].append(result)

        with open(output_file, "a") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")


    def judge(self):
        output_files = {}
        for model in self.models:
            output_files[model] = os.path.join(
                self.output_dir,
                f"{model}.jsonl",
            )

        for output_file in output_files.values():
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.endpoint_info["parallel"]) as executor:
            futures = []
            for model in self.models:
                count = 0
                for question in self.questions:
                    question_id = question["question_id"]

                    if model in self.model_answers and not question_id in self.model_answers[model]:
                        print(f"Warning: {model} answer to {question['question_id']} cannot be found.")
                        continue

                    if model in self.existing_judgments and question_id in self.existing_judgments[model]:
                        count += 1
                        continue

                    answer = self.model_answers[model][question_id]
                    if self.ref_answers:
                        reference = [ref_answer[question_id] for ref_answer in self.ref_answers]
                        assert len(reference) == len(self.settings["ref_model"])
                    else:
                        reference = None
                    if self.settings["baseline"]:
                        baseline_answer = self.model_answers[self.settings["baseline_model"]][question_id]
                    else:
                        baseline_answer = None

                    future = executor.submit(
                        self.judgment, 
                        question, 
                        answer, 
                        reference, 
                        baseline_answer,
                        output_files[model]
                    )
                    futures.append(future)

                if count > 0:
                    print(f"{count} number of existing judgments")

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                future.result()

    def __str__(self):
        return (
            f'\njudge model: {self.settings["judge_model"]}, baseline: {self.settings["baseline"]}, baseline model: {self.settings["baseline_model"]},' 
            f'reference: {self.settings["reference"]}, reference models: {self.settings["ref_model"]}, temperature: {self.settings["temperature"]}, '
            f'max tokens: {self.settings["max_tokens"]}, pairwise: {self.settings["pairwise"]}\n'
        )
