import argparse
import src.phonetisaurus
import src.wav2vec2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help="path to json config with settings")

    subparsers = parser.add_subparsers(help='sub-command help')
    src.phonetisaurus.add_command(subparsers)
    src.wav2vec2.add_command(subparsers)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

