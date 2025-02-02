import datetime
import os
from glob import glob

import numpy as np
import pandas as pd
from evalica import bradley_terry, Winner
from scipy.special import expit
from tqdm import tqdm

from utils import (
    load_questions, 
    load_model_answers, 
    get_models_answers_lengths, 
    get_win_rate_column,
    make_config
)

from rich.console import Console
from rich.table import Table


class Result:
    def __init__(self, generation_path, judgment_path, load_battles=False, load_bootstrap=False, show_elo=False, 
                 length_control=False, weight=3, num_rounds=100, output=False, first_game_only=False):
        self.judgment_path = judgment_path
        self.judge_config = make_config(os.path.join(judgment_path, "config", "judge_config.yaml"))
        self.bench_name = self.judge_config["bench_name"]
        self.judge_name = self.judge_config["judge_model"]
        self.baseline = self.judge_config["baseline_model"]
        self.load_battles = load_battles
        self.load_bootstrap = load_bootstrap
        self.show_elo = show_elo
        self.length_control = length_control
        self.weight = weight
        self.num_rounds = num_rounds
        self.output = output
        self.first_game_only = first_game_only 

        self.questions_df = pd.DataFrame(load_questions(os.path.join("data", self.bench_name, "question.jsonl"))).set_index('question_id')
        self.model_answers_df = pd.DataFrame(load_model_answers(generation_path)).T
        self.models_answers_lengths = get_models_answers_lengths(self.questions_df, self.model_answers_df)

    def compute_ratings(self, df: pd.DataFrame, initial: float = 1000., base: float = 10.,
                    scale: float = 400.) -> 'pd.Series[str]':
        df = df.copy()

        df['winner'] = df['winner'].map({
            'model_a': Winner.X,
            'model_b': Winner.Y,
            'tie': Winner.Draw,
            'tie (bothbad)': Winner.Draw,
        })

        result = bradley_terry(
            df['model_a'],
            df['model_b'],
            df['winner'],
            weights=df['answer_len_delta'] * 2,
            tolerance=1e-8
        )

        scores = initial + np.log(result.scores) / np.log(base) * scale

        if self.baseline in scores.index:
            scores += initial - scores[self.baseline]

        return scores.sort_values(ascending=False, kind="stable")

    def get_battles_from_judgment(self):
        arena_hard_battles = pd.DataFrame()

        print("Turning judgment results into battles...")

        directory = self.judgment_path
        assert os.path.exists(directory)
        for file in tqdm(glob(f"{directory}/*jsonl")):
            df = pd.read_json(file, lines=True)

            for _, row in df.iterrows():
                if self.length_control:
                    _model_name = row["model"].split('/')[-1]
                    answers_length_deltas = (self.answers_lengths.loc[self.baseline] - self.answers_lengths.loc[_model_name])
                    answer_length_delta = (self.answers_lengths.loc[self.baseline][row["question_id"]] -
                                        self.answers_lengths.loc[_model_name][row["question_id"]])
                    normalized_answer_delta_weight = expit(answer_length_delta / answers_length_deltas.std())
                else:
                    normalized_answer_delta_weight = 0.5

                # game 1
                output = {
                    "question_id": row["question_id"],
                    "model_a": self.baseline,
                    "model_b": row["model"],
                    "answer_len_delta": 0.5
                }

                game = row["games"][0]

                weight = 1
                if game["score"] == "A=B":
                    output["winner"] = "tie"
                elif game["score"] == "A>B":
                    output["winner"] = "model_a"
                elif game["score"] == "A>>B":
                    output["winner"] = "model_a"
                    weight = self.weight
                elif game["score"] == "B>A":
                    output["winner"] = "model_b"
                    output['answer_len_delta'] = normalized_answer_delta_weight
                elif game["score"] == "B>>A":
                    output["winner"] = "model_b"
                    output['answer_len_delta'] = normalized_answer_delta_weight
                    weight = self.weight
                else:
                    weight = 0

                if weight:
                    arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])

                if not self.first_game_only:
                    # game 2
                    output = {
                        "question_id": row["question_id"],
                        "model_a": self.baseline,
                        "model_b": row["model"],
                        "answer_len_delta": 0.5
                    }

                    game = row["games"][1]

                    weight = 1
                    if game["score"] == "A=B":
                        output["winner"] = "tie"
                    elif game["score"] == "A>B":
                        output["winner"] = "model_b"
                        output['answer_len_delta'] = normalized_answer_delta_weight
                    elif game["score"] == "A>>B":
                        output["winner"] = "model_b"
                        output['answer_len_delta'] = normalized_answer_delta_weight
                        weight = self.weight
                    elif game["score"] == "B>A":
                        output["winner"] = "model_a"
                    elif game["score"] == "B>>A":
                        output["winner"] = "model_a"
                        weight = self.weight
                    else:
                        weight = 0

                    if weight:
                        arena_hard_battles = pd.concat([arena_hard_battles, pd.DataFrame([output] * weight)])
        arena_hard_battles.to_json("data/arena_hard_battles.jsonl", lines=True, orient="records")
        return arena_hard_battles
    
    def get_bootstrap_result(self, battles, num_round):
        rows = []
        for i in tqdm(range(num_round), desc="bootstrap"):
            rows.append(self.compute_ratings(battles.sample(frac=1.0, replace=True, random_state=i)))
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]

    def compute(self):
        if self.load_battles:
            assert os.path.exists("data/arena_hard_battles.jsonl")
            battles = pd.read_json("data/arena_hard_battles.jsonl", lines=True)
        else:
            battles = self.get_battles_from_judgment()

        bootstrap_ratings = self.compute_ratings(battles)

        models_names = bootstrap_ratings.index

        if self.load_bootstrap:
            bootstrap_ratings_lu = pd.read_json("data/bootstrapping_results.jsonl", lines=True)
        else:
            bootstrap_ratings_lu = self.get_bootstrap_result(battles, self.num_rounds)
            bootstrap_ratings_lu.to_json("data/bootstrapping_results.jsonl", lines=True, orient="records")

        stats = pd.DataFrame()
        stats["results"] = None
        stats["results"] = stats['results'].astype('object')
        
        for i, model in enumerate(models_names):
            assert model in bootstrap_ratings_lu.columns

            stats.at[i, "model"] = model
            stats.at[i, "score"] = bootstrap_ratings[model]
            stats.at[i, "lower"] = np.percentile(bootstrap_ratings_lu[model], 2.5)
            stats.at[i, "upper"] = np.percentile(bootstrap_ratings_lu[model], 97.5)

            stats.at[i, "avg_tokens"] = self.models_answers_lengths.loc[model.split('/')[-1]].mean()
            stats.at[i, "std_tokens"] = self.models_answers_lengths.loc[model.split('/')[-1]].std()

            stats.at[i, "results"] = bootstrap_ratings_lu[model].tolist()

        if not self.show_elo:
            stats.sort_values(by="model", inplace=True)
            stats["score"] = get_win_rate_column(stats, "score", self.baseline).tolist()
            stats["lc_score"] = get_win_rate_column(stats, "score", self.baseline).tolist()
            stats["lower"] = get_win_rate_column(stats, "lower", self.baseline).tolist()
            stats["upper"] = get_win_rate_column(stats, "upper", self.baseline).tolist()
            decimal = 1
        else:
            decimal = 0
            stats = stats.astype({"score": int, "lower": int, "upper": int})

        return stats, decimal

    def show(self):
        console = Console()
    
        # Create a Rich Table
        table = Table(show_header=True, header_style="bold magenta", show_lines=False)
        table.add_column("Model", width=40)
        table.add_column("Score", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("Avg. #Tokens", justify="right")
        
        stats, decimal = self.compute()

        # Sort values by 'score' as per your existing code
        stats.sort_values(by="score", ascending=False, inplace=True)
        
        # Add rows to the table
        for _, row in stats.iterrows():
            interval = f"({round(row['lower'] - row['score'], decimal)}, {round(row['upper'] - row['score'], decimal)})"
            table.add_row(
                row['model'],
                f"{round(row['score'], decimal)}",
                interval,
                f"{int(row['avg_tokens'])}"
            )
        
        # Print the table using Rich
        console.print(table)

        if self.output:
            cur_date = datetime.datetime.now()
            date_str = cur_date.strftime("%Y%m%d")
            stats.to_json(f"arena_hard_leaderboard_{date_str}.json", orient="records", indent=4)

    def __str__(self):
        return '\n' + str({"bench_name": self.bench_name, "judge_name": self.judge_name, "baseline": self.baseline, "load_battles": self.load_battles,
                "load_bootstrap": self.load_bootstrap, "show_elo": self.show_elo, "length_control": self.length_control, "weight": self.weight,
                "num_rounds": self.num_rounds, "output": self.output, "first_game_only": self.first_game_only}) + '\n'