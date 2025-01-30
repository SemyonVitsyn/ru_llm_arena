import argparse
from gen_answer import AnswerGenerator
from gen_judgment import Judge
from show_result import Result


TASK_TYPES = [
    "generate",
    "judge",
    "show",
    "full_pipeline"
]

BASELINE_MODEL_NAME = "gpt-3.5-turbo-0125"

def generate(generation_config, endpoint_file):
    print("\n\n----------------------------GENERATION----------------------------")
    generator = AnswerGenerator(generation_config, endpoint_file)
    print(generator)
    generator.generate()

def judge(judgement_config, endpoint_file):
    print("\n\n----------------------------JUDGEMENT----------------------------")
    judge = Judge(judgement_config, endpoint_file)
    print(judge)
    judge.judge()

def result(bench_name, judge_name, baseline, load_battles, load_bootstrap, show_elo,
               length_control, weight, num_rounds, output, first_game_only):
    print("\n\n----------------------------RESULT----------------------------")
    result = Result(bench_name, judge_name, baseline, load_battles, load_bootstrap, show_elo,
                    length_control, weight, num_rounds, output, first_game_only)
    print(result)
    result.show()

def full_pipeline(generation_config, judgement_config, endpoint_file, bench_name, judge_name, 
                  baseline, load_battles, load_bootstrap, show_elo, length_control, weight, 
                  num_rounds, output, first_game_only):
    
    generate(generation_config, endpoint_file)
    judge(judgement_config, endpoint_file)
    result(bench_name, judge_name, baseline, load_battles, load_bootstrap, show_elo,
               length_control, weight, num_rounds, output, first_game_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", type=str, default="full_pipeline")
    parser.add_argument("--generation-config", type=str, default="config/gen_answer_config.yaml")
    parser.add_argument("--endpoint-file", type=str, default="config/api_config.yaml")
    parser.add_argument("--judgement-config", type=str, default="config/judge_config.yaml")
    parser.add_argument("--bench-name", type=str, default="arena-hard-v0.1")
    parser.add_argument("--judge-name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--baseline", type=str, default=BASELINE_MODEL_NAME)
    parser.add_argument("--load-battles", action="store_true")
    parser.add_argument("--load-bootstrap", action="store_true")
    parser.add_argument("--show-elo", action="store_true")
    parser.add_argument("--length-control", action="store_true")
    parser.add_argument("--weight", type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--first-game-only", action="store_true")

    args = parser.parse_args()

    assert args.task_type in TASK_TYPES
    assert not args.load_bootstrap or (
            args.load_battles and args.load_bootstrap), "If loading prexisting bootstrapping data, you must also load preexisting battles."


    if args.task_type == "generate":
        generate(args.generation_config, args.endpoint_file)
    elif args.task_type == "judge":
        judge(args.judgement_config, args.endpoint_file)
    elif args.task_type == "show":
        result(args.bench_name, args.judge_name, args.baseline, args.load_battles, args.load_bootstrap, args.show_elo,
               args.length_control, args.weight, args.num_rounds, args.output, args.first_game_only)
    else:
        full_pipeline(args.generation_config, args.judgement_config, args.endpoint_file, args.bench_name, args.judge_name, 
                      args.baseline, args.load_battles, args.load_bootstrap, args.show_elo, args.length_control, 
                      args.weight, args.num_rounds, args.output, args.first_game_only)

