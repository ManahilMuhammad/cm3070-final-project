import ollama
import instrumentation as inst
import json
from .config import TEXT_MODEL

# words that cannot be used as blank words in fill-in-the-blank
_BLANK_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "and", "or", "but", "it", "its",
    "this", "that", "these", "those", "as", "by", "with", "from", "not",
}

# words that cannot appear at the beginning of a true-false question 
# (because they would make it a question rather than a statement that is true or false)
_QUESTION_STARTERS = {
    "which", "what", "who", "whom", "whose", "how", "why", "when", "where",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "will",
    "would", "should", "has", "have", "had",
}

def _question_words(text):
    """
    strips the question to only important words therein
    """
    return {w.strip(".,?!:;'\"()_").lower() for w in text.split() if len(w.strip(".,?!:;'\"()_")) > 2}

def _too_similar(question, asked, threshold=0.5):
    """
    checks for duplicates
    high word overlap with any previously asked question
    """
    q_words = _question_words(question) # simplify question to root words

    if not q_words:
        return False

    # search previously asked questions and see if any of them 
    # majorly overlap with this one
    for prev in asked:
        prev_words = _question_words(prev)
        if not prev_words:
            continue
        overlap = len(q_words & prev_words) / len(q_words | prev_words)
        if overlap >= threshold:
            return True
        
    return False

def _valid_answer(ques_type, question, answer, options, avoid=None):
    """
    retries bad generation of each type of question instead of returning it
    """

    ans = str(answer).strip()

    # if detected to be a duplicate then invalid
    if _too_similar(question, avoid or []):
        return False

    # MCQ
    if ques_type == "multiple-choice":

        # check for correct type and length of options list
        if not isinstance(options, list) or len(options) != 4:
            return False
        
        cleaned = [str(o).strip() for o in options if str(o).strip()]

        # check that each MCQ option is distinct
        if len(cleaned) != 4 or len({o.lower() for o in cleaned}) != 4:
            return False
        
        return ans in cleaned

    # fill-in-the-blank
    if ques_type == "fill-in-the-blank":

        # check for missing blank
        if "_____" not in question:
            return False

        # check that answer is not present within question
        if ans.lower() in question.lower():
            return False

        # check that answer is a valid word
        return len(ans) >= 3 and ans.lower() not in _BLANK_STOPWORDS

    # true-false
    if ques_type == "true-false":

        # check that answer is either true or false
        if ans.lower() not in ("true", "false"):
            return False

        stripped = question.strip()

        # check that the sentence is not a question
        if stripped.endswith("?"):
            return False

        # check that the first word is not one of the question starters
        first_word = stripped.split(" ", 1)[0].lower().strip(",.:;\"'")
        return first_word not in _QUESTION_STARTERS

    # short-answer
    if ques_type == "short-answer":
        # check that answer is between 1 and 6 words, inclusive (short)
        return 1 <= len(ans.split()) <= 6

    return True

def generate_ques(content, spec, topics, retries=4, avoid=None):
    """
    generates one question per topic, requesting all of them in a single
    LLM call; on retry only re-requests topics that failed validation
    """
    avoid = list(avoid or [])
    ques_type = spec["type"]
    remaining = list(topics)
    accepted = []

    for retry in range(retries):
        if not remaining:
            break

        count = len(remaining)
        topics_desc = "\n".join(
            f'{i + 1}. "{t}"' if t else f"{i + 1}. Any topic from the content."
            for i, t in enumerate(remaining)
        )

        avoid_block = ""
        if avoid:
            already = "\n".join(f"- {a}" for a in avoid)
            avoid_block = f"""
Do NOT repeat or closely rephrase any of these questions already asked:
{already}
""" # avoid duplicates

        prompt = f"""You are a quiz generator. Read the content, then create exactly {count} DIFFERENT {ques_type} questions, one for each topic listed below, in the same order.

CONTENT:
{content}

Topics (one question per topic, in order):
{topics_desc}

{spec["instructions"]}
{avoid_block}
General rules:
- Base each question ONLY on facts stated in the content. Do not invent facts, numbers, or terms that are not in the content.
- Write original questions in your own words. Do NOT copy a sentence verbatim from the content as a question.
- Each question must be understandable and answerable on its own, without needing to see the original content.
- The {count} questions must all be different from each other.

Return ONLY a JSON object with this EXACT key: "questions" - a list of exactly {count} objects, each with keys "question", "answer", "options", in the same order as the topics above.
Example shape: {json.dumps({"questions": [spec["shape"]] * count})}
Do NOT return a summary. Do NOT use any other keys."""

        response = ollama.chat(model=TEXT_MODEL,
                               messages=[
                                   {"role": "system", "content": "You generate quiz questions as JSON. You never summarise."},
                                   {"role": "user", "content": prompt}
                                ],
                                format="json",
                                options={"temperature": 0.3},
                                )

        try:
            items = json.loads(response["message"]["content"])["questions"]
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Failed batch attempt {retry}")
            continue

        if not isinstance(items, list):
            continue

        # go back for next retry instead of redoing whole batch
        # if anything invalid or missing from response
        still_needed = []
        for topic, item in zip(remaining, items):
            ques = item.get("question") if isinstance(item, dict) else None
            ans = item.get("answer") if isinstance(item, dict) else None
            opts = item.get("options") if isinstance(item, dict) else None
            if ques and ans not in (None, "", []) and _valid_answer(ques_type, ques, ans, opts, avoid):
                accepted.append({"type": ques_type, "question": ques, "answer": ans, "options": opts, "topic": topic or "General"})
                avoid.append(ques)
            else:
                still_needed.append(topic)
        still_needed.extend(remaining[len(items):])
        remaining = still_needed

    return accepted

@inst.timed("create quiz")
def make_quiz(combined, per_type=2, topics=None):
    questions = []

    # extract topics if not already done
    if topics is None:
        topics = extract_topics(combined, n=8)

    # specifications for each question type
    qspecs = [
        {
            "type": "multiple-choice",
            "shape": {"question": "<question>", "answer": "<correct answer>", "options": ["<option1>","<option2>","<option3>","<option4>"]},
            "instructions": (
                "Ask about a specific fact, definition, or relationship from the content. "
                "Provide exactly 4 options. Exactly ONE must be correct and explicitly "
                "supported by the content; the other 3 must be plausible but clearly "
                "wrong - not just a reworded version of the correct answer, and not "
                "also technically correct. All 4 options must be distinct from each "
                "other and similar in length and style. 'answer' MUST be copied "
                "exactly, character-for-character, from one of the 'options'."
            )
        },
        {
            "type": "fill-in-the-blank",
            "shape": {"question": "<sentence with a missing word represented as _____>", "answer": "<missing word>", "options": None},
            "instructions": (
                "Write ONE original factual sentence based on the content (do not "
                "copy a sentence verbatim from the content). Replace ONE important, "
                "specific keyword or technical term that is central to that "
                "sentence's meaning with '_____'. NEVER blank a common word like "
                "'the', 'is', 'a', or 'this' - the blanked word must be something a "
                "student who understood the topic could infer from the rest of the "
                "sentence. The 'question' is the sentence WITH the blank. The "
                "'answer' is the exact word or short term you removed, and it must "
                "NOT appear anywhere else in the question."
            )
        },
        {
            "type": "true-false",
            "shape": {"question": "<statement>", "answer": "<correct answer>", "options": None},
            "instructions": (
                "Write ONE original declarative STATEMENT, in your own words, that "
                "is clearly true or clearly false according to the content - not a "
                "sentence copied verbatim from it. This must be a statement, NOT a "
                "question: it must NOT end with a question mark, and it must NOT "
                "start with a word like 'Which', 'What', 'Does', 'Is', 'Are', or "
                "'Can'. For example, write 'X causes Y' instead of 'Does X cause Y?'. "
                "'answer' is exactly 'true' or 'false'."
            )
        },
        {
            "type": "short-answer",
            "shape": {"question": "<question>", "answer": "<short answer>", "options": None},
            "instructions": (
                "Ask a specific factual question that has exactly ONE unambiguous, "
                "short correct answer: a single word, number, or short phrase of at "
                "most 3-4 words (e.g. a name, term, or figure from the content). Do "
                "NOT ask a question whose correct answer would require a full "
                "sentence or is open to interpretation. 'answer' must be that short "
                "phrase, taken from the content - not a full sentence."
            )
        }
    ]

    asked = []
    topic_idx = 0
    for spec in qspecs:

        # assign per_type target topics to this type's batch
        batch_topics = []
        for _ in range(per_type):
            batch_topics.append(topics[topic_idx % len(topics)] if topics else None)
            topic_idx += 1

        batch = generate_ques(combined, spec, batch_topics, avoid=asked)
        questions.extend(batch)
        asked.extend(q["question"] for q in batch) # keep track of asked questions to avoid duplicates

    return questions

@inst.timed("extract topics")
def extract_topics(text, n=8):
    """
    extract topics from source text
    """
    prompt = (
        "You are labelling the topics covered in a lecture.\n"
        f"List at most {n} distinct topics.\n"
        "Each topic must be 1-4 words and specific to the material.\n"
        'Respond with JSON only: {"topics": ["...", "..."]} \n\n'
        f"Lecture material:\n{text}"
    )

    for _ in range(3):
        raw = ollama.generate(
            model=TEXT_MODEL, prompt=prompt, format="json",
            options={"temperature": 0.1, "seed": 42},
        )["response"]

        # extract topics from generated text
        try:
            topics = [str(t).strip() for t in json.loads(raw)["topics"] if str(t).strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

        # dedupe the list of topics, keeping the order intact
        seen, out = set(), []
        for t in topics:
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)

        if out:
            return out[:n]

    return ["General"]