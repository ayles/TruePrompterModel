import itertools
import json
import re
from .phonemizer import EpitranPhonemizer
from .remapper import PhonemeRemapper


def _iterate_wiktionary_dump(data_path, languages, count=float("inf")):
    cnt = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "lang_code" not in data or data["lang_code"] not in languages:
                continue
            if "word" in data:
                yield (data["lang_code"], data["word"])
                cnt += 1
                if cnt > count:
                    return


def _process(language, sentences, phonemizer, phoneme_stats):
    result = phonemizer.phonemize(language, sentences)
    for sentence, words in zip(sentences, result):
        for phonemes in words:
            for phoneme in phonemes:
                if phoneme not in phoneme_stats:
                    phoneme_stats[phoneme] = 0
                phoneme_stats[phoneme] += 1


def build_remapper(lang_sentences, phonemizer, matcher, tokens_max=32):
    stats_by_lang = dict()
    batches = dict()
    for language, sentence in lang_sentences:
        if language not in batches:
            batches[language] = []
        if language not in stats_by_lang:
            stats_by_lang[language] = dict()

        batch = batches[language]
        stats = stats_by_lang[language]

        def cleanup(text):
            text = re.sub(r"^\s+", "", text)
            text = re.sub(r"\s+$", "", text)
            text = re.sub(r"\s+", " ", text)
            return text

        batch.append(cleanup(sentence))

        if len(batch) > 100:
            _process(language, batch, phonemizer, stats)
            batch.clear()

    for language, batch in batches.items():
        if len(batch) > 0:
            _process(language, batch, phonemizer, stats)

    stats = dict()
    for language, counts in stats_by_lang.items():
        total_count = 0
        for _, count in counts.items():
            total_count += count
        for phoneme in counts:
            counts[phoneme] /= total_count
            if phoneme not in stats:
                stats[phoneme] = 0
            stats[phoneme] += counts[phoneme] / len(stats_by_lang)

    srtd = sorted(stats.items(), key=lambda s: s[1], reverse=True)
    vocabulary = {phoneme for phoneme, value in itertools.islice(srtd, tokens_max)}
    mapping = dict()

    for phoneme in stats:
        if phoneme not in vocabulary:
            if phoneme not in phonemizer.vectors:
                continue
            best_fit = None
            for a in vocabulary:
                if best_fit is None:
                    best_fit = a
                    continue
                a_similarity = matcher.match(phoneme, a)
                b_similarity = matcher.match(phoneme, best_fit)
                if a_similarity > b_similarity or (a_similarity == b_similarity and stats[a] > stats[best_fit]):
                    best_fit = a
            mapping[phoneme] = [best_fit]

    return PhonemeRemapper(vocabulary=vocabulary, mapping=mapping), stats, stats_by_lang


def build(languages, count, tokens_max, data_path, output_path):
    phonemizer = EpitranPhonemizer()
    matcher = phonemizer
    remapper, _, _ = build_remapper(_iterate_wiktionary_dump(data_path, languages, count), phonemizer, matcher, tokens_max)
    with open(output_path, "w") as f:
        remapper.save(f)


def add_command(subparsers):
    parser = subparsers.add_parser('vocabulary', help='generate vocabulary ')
    parser.add_argument('-l','--language', required=True, nargs='+', help='language to build vocabulary to, space-separated if many')
    parser.add_argument('-t','--tokens', type=int, default=32, help='maximum number of tokens in resulting vocabulary')
    parser.add_argument('-c','--count', type=int, default=float("inf"), help='maximum number of rows from dataset to consume')
    parser.add_argument('-d', '--data', required=True, help='path to wiktionary dump')
    parser.add_argument('-o', '--out', required=True, help='remapper output path')
    parser.set_defaults(func=lambda args: build(args.language, args.count, args.tokens, args.data, args.out))


"""
matplotlib.use('TkAgg')

def do_bar(plt, s):
    x = [(k, v) for k, v in sorted(s.items(), key=lambda x: x[1])]
    return x, plt.bar(list(map(lambda x: x[0], x)), list(map(lambda x: x[1], x)))


fig, ax = plt.subplots()
srt, barlist = do_bar(ax, stats)
for kv, b in zip(srt, barlist):
    color = (1 * (stats_by_lang["ru"][kv[0]] / kv[1] / 2 if kv[0] in stats_by_lang["ru"] else 0), 0, 1 * (stats_by_lang["en"][kv[0]] / kv[1] / 2 if kv[0] in stats_by_lang["en"] else 0))
    #color = (1, 0, 0)
    b.set_color((1 if color[0] > 1 else color[0], color[1], 1 if color[2] > 1 else color[2]))
fig.show()

fig, ax = plt.subplots()
do_bar(ax, stats_by_lang["en"])
do_bar(ax, stats_by_lang["ru"])

plt.show()
"""

