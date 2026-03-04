import argparse

from commands.commands import execute_command

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", choices=["gainpro"], required=True)
    parser.add_argument("-command", choices=["train", "evaluate"], required=True)
    parser.add_argument("-dataset", type=str, required=True)
    args = parser.parse_args()

    execute_command(
        command_name = args.command, 
        model_name = args.model, 
        dataset_path = args.dataset
    )

if __name__ == "__main__":
    main()