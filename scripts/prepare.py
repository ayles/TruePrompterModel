import json
import re
import sys
import argparse
from phonemizer import phonemize
from phonemizer.separator import Separator
from phonemizer.backend import BACKENDS


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="path to json config with phonemes, mapping and other settings")
parser.add_argument("-d", "--data", help="path to wiktionary line-json dump")
args = parser.parse_args()


with open(args.config) as file:
    config = json.load(file)


phonemes = set(config["phonemes"])
mapping = config["mapping"]
langs = config["langs"]


def clean_word(word):
    word = re.sub(r"^\s+", "", word)
    word = re.sub(r"\s+$", "", word)
    word = re.sub(r"\s+", " ", word)
    return word


not_found = set()
used = set()
used_final = set()


def remap_phone(phoneme):
    mapped = phoneme
    if phoneme not in phonemes and phoneme in mapping:
        mapped = mapping[phoneme]
        used.add(phoneme)

    mapped = [m if m in phonemes else None for m in mapped]

    for m in mapped:
        if m is None:
            not_found.add(phoneme)
        else:
            used_final.add(m)

    return [m for m in mapped if m is not None]


backends = {lang["wiktionary"]:BACKENDS["espeak"](lang["espeak"]) for lang in langs}
batches = {lang["wiktionary"]:[] for lang in langs}

with open(args.data, "r", encoding="utf-8") as f:
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
                            t = " ".join(map(lambda phoneme: " ".join(remap_phone(phoneme)), t.split("|")))
                            print(f"{w}\t{t}")
                batches[lang_code] = []


print("Not found in remappings: ", not_found, file=sys.stderr)
print("Not used remappings: ", set(mapping.keys()).difference(used), file=sys.stderr)
print("Not used from vocab: ", phonemes.difference(used_final), file=sys.stderr)

