import json
import re
from phonemizer import phonemize
from phonemizer.separator import Separator

def clean(s):
    s = re.sub(r"^\s+", "", s)
    s = re.sub(r"\s+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

with open("/Users/ayles/Downloads/raw-wiktextract-data.json", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if "lang_code" not in data or data["lang_code"] != "en":
            continue

        if "sounds" in data:
            for sound in data["sounds"]:
                if "ipa" in sound:
                    word = clean(data["word"])
                    phn = phonemize(
                        [word],
                        language='en-us',
                        backend='espeak',
                        separator=Separator(phone='|', word='/'),
                        strip=True,
                        preserve_punctuation=True)
                    print(phn[0].replace('/', ' ').replace('|', ' '))
                    transcript = clean("|".join([str(c) for c in ipapy.ipastring.IPAString(unicode_string=sound["ipa"], ignore=True)]))
                    word_split = word.split(" ")
                    transcript_split = transcript.split(" ")
                    if len(word_split) == len(transcript_split):
                        for w, t in zip(word_split, transcript_split):
                            print(f"{w}\t{t.replace('|', ' ')}")

