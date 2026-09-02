import ollama
import instrumentation as inst
from .config import TEXT_MODEL

@inst.timed('summarisation')
def make_summary(combined):
    system = (
        'You are a careful academic note-taker. You turn lecture material into '
        'structured study notes. You never invent facts, examples, or figures '
        'that are not present in the source text. You output ONLY the notes '
        'themselves - never a preamble, never a note to the person who asked, '
        'and never a sign-off.'
    )

    prompt = f"""Summarise the lecture material below into structured study notes.

Follow this exact structure:
1. A single '# Overview' section: 2-3 sentences describing what the material covers.
2. One '## <Topic>' heading for EACH distinct topic, in the order it first appears.
3. Under each topic heading, 3-6 concise bullet points with the key facts, definitions, and relationships for that topic.
4. A final '## Key Terms' section: a bulleted list formatted as '**Term** - one-line definition'.

Rules:
- Use ONLY information present in the TEXT below. Do not add outside knowledge or examples not present in the text.
- Mention every topic covered in the TEXT; do not omit or merge distinct topics.
- Be concise and factual. No filler like 'In this lecture' or 'This section discusses'.
- Use consistent markdown: '#' for the title, '##' for headings, '-' for bullets.
- Do NOT include any preamble, meta-commentary, or sign-off - no 'Here is a summary', no 'Based on the text provided', no 'I hope this helps'. Output must start immediately with '# Overview' and end after the Key Terms section.

TEXT:
{combined}
"""

    summary = ollama.chat(
        model=TEXT_MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ],
        options={'temperature': 0.2, 'seed': 42},
    )['message']['content']

    return summary