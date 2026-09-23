import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

root = Path(__file__).resolve().parent
sys.path.append(str(root.parent))

import pipeline.generation_quiz as gq
import pipeline.generation_summary as gs
from pipeline import make_summary, make_quiz, release_llm

ORCA = 'orca-mini:3b'
LLAMA = 'llama3.2:3b'

# display names
MODELS = {'Orca-Mini 3B': ORCA, 'Llama 3.2 3B': LLAMA}

# rouge variants
ROUGE_TYPES = [
    'rouge1', # unigram overlap
    'rouge2', # bigram overlap
    'rougeL' # longest common subsequence
]

SOURCE_DIR = root / 'test_text_sources'
OUT_DIR = root / 'results' / 'text_outputs'

def use_model(model):
    """
    generation modules read TEXT_MODEL constant defined in config
    so it is replaced with the models used here
    """
    gs.TEXT_MODEL = model
    gq.TEXT_MODEL = model


class AnswerCounter:
    """
    counts every validation performed and the ones that pass
    """

    def __init__(self):
        self.real = gq._valid_answer
        self.checked = 0
        self.valid = 0

    def __enter__(self):
        gq._valid_answer = self
        return self

    def __exit__(self, *exc):
        gq._valid_answer = self.real
        return False

    def __call__(self, ques_type, ques, ans, opt, avoid=None):
        ok = self.real(ques_type, ques, ans, opt, avoid)
        self.checked += 1
        self.valid += bool(ok)
        return ok


def load_sources():
    """
    read the source text
    """
    sources = []

    for txt in sorted(SOURCE_DIR.glob('*.txt')):
        text = txt.read_text(encoding='utf-8').strip()
        if not text:
            print(f'Empty source: {txt.name}')
            continue

        sources.append((txt.stem, text))

    return sources

def run_text(text, model):
    """
    run summary and quiz generation functions
    """
    use_model(model)

    # make summary
    summary = make_summary(text).strip()

    # make quiz
    with AnswerCounter() as counter:
        quiz = make_quiz(text)

    return {
        'summary': summary,
        'quiz': quiz,
        'checked': counter.checked,
        'valid': counter.valid
    }

def generate(sources, cache, regen=False):
    """
    run both models over each piece of text
    """

    # only generate again if not already in cache
    gens = {} if regen or not cache.exists() else json.loads(cache.read_text(encoding='utf-8'))

    for name, model in MODELS.items():
        gens.setdefault(name, {})
        todo = [(stem, text) for stem, text in sources if stem not in gens[name]]

        if not todo:
            print(f'{name} - all {len(sources)} generations cached')
            continue

        print(f'\n{name} - generating for {len(todo)} texts')
        for stem, text in todo:
            gens[name][stem] = run_text(text, model)
            print(f'{stem} - DONE')
            cache.write_text(json.dumps(gens, indent=2, ensure_ascii=False), encoding='utf-8')

        # unload model once done with every text
        release_llm(model=model, end_of_phase=False)

    return gens

def rouge(summary, source, scorer):
    """
    score summary against source text
    """
    scores = scorer.score(source, summary)
    return {t: scores[t].fmeasure for t in ROUGE_TYPES}


def get_scores(sources, gens):
    stems = [stem for stem, _ in sources]
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    res = pd.DataFrame({'text': stems})

    for name in MODELS:
        outputs = [gens[name][stem] for stem in stems]
        res[f'{name}_summary'] = [o['summary'] for o in outputs]

        # rouge f-measure for source text
        scored = [rouge(o['summary'], text, scorer) for o, (_, text) in zip(outputs, sources)]
        for t in ROUGE_TYPES:
            res[f'{name}_{t}'] = [s[t] for s in scored]

        # quiz answer validity
        res[f'{name}_checked'] = [o['checked'] for o in outputs]
        res[f'{name}_valid'] = [o['valid'] for o in outputs]

    return res


def valid_ratio(res, name):
    """
    ratio across every text
    """
    checked = res[f'{name}_checked'].sum()
    return res[f'{name}_valid'].sum() / checked if checked else 0.0


def plot(res):
    """
    mean rouge scores and valid answer ratios
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # mean rouge measure per model
    x = np.arange(len(ROUGE_TYPES))
    w = 0.34

    for i, name in enumerate(MODELS):
        offset = (i - (len(MODELS) - 1) / 2) * w
        means = [res[f'{name}_{t}'].mean() for t in ROUGE_TYPES]
        stds = [res[f'{name}_{t}'].std() for t in ROUGE_TYPES]

        bars = ax1.bar(x + offset, means, w*0.94, yerr=stds, capsize=4, label=name)
        ax1.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)

    ax1.set_xticks(x, ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    ax1.set_ylabel('Mean F-measure vs source text')
    ax1.set_title('Summary overlap with source')
    ax1.legend(frameon=False)

    # ratio of valid question shapes
    names = list(MODELS)
    ratios = [valid_ratio(res, name) for name in names]
    bars = ax2.bar(names, ratios, 0.5)
    ax2.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)

    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('Valid responses / responses generated')
    ax2.set_title('Quiz shape validity')

    for ax in (ax1, ax2):
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'text_comparison.png', dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regenerate', action='store_true', help='ignore cached outputs and rerun both models')
    ap.add_argument('--skip-generation', action='store_true', help='only re-score and re-plot from cached outputs')
    ap.add_argument('--no-show', action='store_true', help='save plots without opening them')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # get source texts
    sources = load_sources()
    if not sources:
        raise SystemExit(f'No source found in {SOURCE_DIR}')
    print(f'{len(sources)} sources')

    cache = OUT_DIR / 'generations.json'
    if args.skip_generation:
        gens = json.loads(cache.read_text(encoding='utf-8'))
    else:
        gens = generate(sources, cache, regen=args.regenerate)

    missing = [(name, stem) for name in MODELS for stem, _ in sources if stem not in gens.get(name, {})]
    if missing:
        raise SystemExit(f'Missing gens: {missing}')

    res = get_scores(sources, gens)
    res.to_csv(OUT_DIR / 'text_scores.csv', index=False)

    plot(res)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
