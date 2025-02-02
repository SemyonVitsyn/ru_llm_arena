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

def generate(generation_path, endpoint_file):
    print("\n\n----------------------------GENERATION----------------------------")
    generator = AnswerGenerator(generation_path, endpoint_file)
    print(generator)
    generator.generate()

def judge(judgment_path, generation_path, endpoint_file):
    print("\n\n----------------------------JUDGMENT----------------------------")
    judge = Judge(judgment_path, generation_path, endpoint_file)
    print(judge)
    judge.judge()

def result(generation_path, judgment_path, load_battles, load_bootstrap, show_elo,
                    length_control, weight, num_rounds, output, first_game_only):
    print("\n\n----------------------------RESULT----------------------------")
    result = Result(generation_path, judgment_path, load_battles, load_bootstrap, show_elo,
                    length_control, weight, num_rounds, output, first_game_only)
    print(result)
    result.show()

def full_pipeline(generation_path, judgment_path, endpoint_file, load_battles, load_bootstrap, 
                  show_elo, length_control, weight, num_rounds, output, first_game_only):
    
    generate(generation_path, endpoint_file)
    judge(judgment_path, generation_path, endpoint_file)
    result(generation_path, judgment_path, load_battles, load_bootstrap, show_elo,
                    length_control, weight, num_rounds, output, first_game_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", type=str, default="full_pipeline")
    parser.add_argument("--generation-path", type=str, default="data/arena-hard-v0.1/model_answer/default")
    parser.add_argument("--endpoint-file", type=str, default="default_config/api_config.yaml")
    parser.add_argument("--judgment-path", type=str, default="data/arena-hard-v0.1/model_judgment/gpt-4-1106-preview_default")
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
        generate(args.generation_path, args.endpoint_file)
    elif args.task_type == "judge":
        judge(args.judgment_path, args.generation_path, args.endpoint_file)
    elif args.task_type == "show":
        result(args.generation_path, args.judgment_path, args.load_battles, args.load_bootstrap, args.show_elo,
               args.length_control, args.weight, args.num_rounds, args.output, args.first_game_only)
    else:
        full_pipeline(args.generation_path, args.judgment_path, args.endpoint_file,args.load_battles, args.load_bootstrap,
                       args.show_elo, args.length_control, args.weight, args.num_rounds, args.output, args.first_game_only)

