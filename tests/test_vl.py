import argparse, io, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ollama
from PIL import Image
from bert_score import score

root = Path(__file__).resolve().parent
sys.path.append(str(root.parent))

from pipeline import release_llm

LLAVA = 'llava:7b'
QWEN = 'qwen2.5vl:latest'

# display names
MODELS = {'LLaVA 7B': LLAVA, 'Qwen2.5-VL': QWEN}

PROMPT = """
You are helping create study notes from lecture material.
Describe the figure, chart, or diagram in the uploaded image clearly including all key data,
trends, labels, axes, or relationships, so it can be understood without seeing the image.
If the image contains no figure or diagram, reply with 'None'.
"""

MAX_TOKENS = 150
IMAGE_DIR = root / 'test_vl_images'
TEXT_DIR = root / 'test_vl_text'
OUT_DIR = root / 'results' / 'vl_outputs'


def load_image(source: str) -> Image.Image:
    img = Image.open(source)

    # change transparent parts to white
    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg

    return img.convert('RGB')


def run_vl(image, model):
    buf = io.BytesIO()
    image.save(buf, format='PNG')

    # generate description
    description = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': PROMPT, 'images': [buf.getvalue()]}],
        options={'num_predict': MAX_TOKENS},
    )['message']['content']

    return description.strip()


def load_pairs():
    """
    match each image file to .txt file of the 
    same name containing its descriptive text
    """
    pairs = []

    for img in sorted(IMAGE_DIR.iterdir()):
        if not img.is_file():
            continue

        txt = TEXT_DIR / f'{img.stem}.txt'
        if not txt.exists():
            print(f'No descriptive text found for {img.name}')
            continue

        description = txt.read_text(encoding='utf-8').strip()
        if not description:
            print(f'Empty descriptive text file for {img.name}')
            continue

        pairs.append((img.stem, img, description))

    return pairs


def generate(pairs, cache, regen=False):
    """
    run both models over each image
    cache descriptions so rerunning file does not result in rerunning models
    """
    gens = {} if regen or not cache.exists() else json.loads(cache.read_text(encoding='utf-8'))

    for name, model in MODELS.items():
        gens.setdefault(name, {})
        todo = [(stem, path) for stem, path, _ in pairs if stem not in gens[name]]

        if not todo:
            print(f'{name} - all {len(pairs)} descriptions cached')
            continue

        print(f'\n{name} - generating {len(todo)} descriptions')
        for stem, path in todo:
            gens[name][stem] = run_vl(load_image(path), model)
            print(f'{stem} - DONE')
            cache.write_text(json.dumps(gens, indent=2, ensure_ascii=False), encoding='utf-8')

        # unload model once done with every image
        release_llm(model=model, end_of_phase=False)

    return gens

def bertscore(gens, refs):
    # spread scores out
    precision, recall, f1 = score(gens, refs, lang='en', rescale_with_baseline=True, verbose=False)
    return precision.numpy(), recall.numpy(), f1.numpy()

def get_scores(pairs, gens):
    stems = [stem for stem, _, _ in pairs]
    refs = [ref for _, _, ref in pairs]

    res = pd.DataFrame({'image': stems, 'reference': refs})

    for name in MODELS:
        outputs = [gens[name][stem] for stem in stems]
        p, r, f = bertscore(outputs, refs)
        res[f'{name}_output'] = outputs
        res[f'{name}_P'], res[f'{name}_R'], res[f'{name}_F1'] = p, r, f

    return res

def plot_means(res):
    """
    plot mean precision, recall, f1 scores per model
    """
    metrics = ['P', 'R', 'F1']
    x = np.arange(len(metrics))
    w = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(MODELS):
        offset = (i - (len(MODELS) - 1) / 2) * w
        means = [res[f'{name}_{m}'].mean() for m in metrics]
        stds = [res[f'{name}_{m}'].std() for m in metrics]
        bars = ax.bar(x + offset, means, w * 0.94, yerr=stds, capsize=4, label=name)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)

    ax.set_xticks(x, ['Precision', 'Recall', 'F1'])
    ax.set_ylabel('BERTScore (rescaled)')
    ax.set_title('Mean BERTScore')
    ax.axhline(0, lw=0.8)
    ax.legend(frameon=False)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'vl_mean_scores.png', dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regenerate', action='store_true', help='ignore cached descriptions and re-run both models')
    ap.add_argument('--skip-generation', action='store_true', help='only re-score and re-plot from cached descriptions')
    ap.add_argument('--no-show', action='store_true', help='save plots without opening them')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # get pairs
    pairs = load_pairs()
    if not pairs:
        raise SystemExit('No image/text pairs found')
    print(f'{len(pairs)} image/text pairs')

    cache = OUT_DIR / 'generations.json'
    if args.skip_generation:
        gens = json.loads(cache.read_text(encoding='utf-8'))
    else:
        gens = generate(pairs, cache, regen=args.regenerate)

    missing = [(name, stem) for name in MODELS for stem, _, _ in pairs if stem not in gens.get(name, {})]
    if missing:
        raise SystemExit(f'missing descriptions: {missing}')

    res = get_scores(pairs, gens)

    # plot the graph
    plot_means(res)

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
