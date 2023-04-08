import json
import os
import re
import subprocess
import sys
from .phoneme_remapper import PhonemeRemapper
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS


def add_command(subparsers):
    parser = subparsers.add_parser('phonetisaurus', help='prepare files for training')
    parser.add_argument('-d', '--data', required=True, help='path to json lines wiktionary dump')
    parser.add_argument('-o', '--out', required=True, help='model output dir')
    parser.set_defaults(func=lambda args: train(args.config, args.data, args.out))


def train(config_path, data_path, output_path):
    with open(config_path) as file:
        config = json.load(file)

    remapper = PhonemeRemapper(phonemes=config["phonemes"], mapping=config["mapping"])
    langs = config["langs"]


    def clean_word(word):
        word = re.sub(r"^\s+", "", word)
        word = re.sub(r"\s+$", "", word)
        word = re.sub(r"\s+", " ", word)
        return word


    backends = {lang["wiktionary"]:BACKENDS["espeak"](lang["espeak"]) for lang in langs}
    batches = {lang["wiktionary"]:[] for lang in langs}

    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    dict_path = os.path.join(output_path, "dict.txt")
    print("Preparing dict...", file=sys.stderr)
    with open(data_path, "r", encoding="utf-8") as f:
        with open(dict_path, "w") as out:
            for line in f:
                data = json.loads(line)
                if "lang_code" not in data or data["lang_code"] not in backends:
                    continue
                lang_code = data["lang_code"]

                if "word" in data:
                    batches[lang_code].append(clean_word(data["word"]))

                    if len(batches[lang_code]) > 100:
                        res = backends[lang_code].phonemize(
                            batches[lang_code],
                            separator=Separator(phone='|'),
                            strip=True,
                        )
                        for word, phones in zip(batches[lang_code], res):
                            word_split = word.split(" ")
                            phones_split = phones.split(" ")
                            if len(word_split) == len(phones_split):
                                for w, t in zip(word_split, phones_split):
                                    t = " ".join(remapper.remap(t.split("|")))
                                    print(f"{w}\t{t}", file=out)
                        batches[lang_code] = []

    subprocess.check_call(["phonetisaurus-train", "--lexicon", dict_path, "--dir_prefix", output_path])



