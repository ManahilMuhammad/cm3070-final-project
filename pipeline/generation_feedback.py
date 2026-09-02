import ollama
from collections import defaultdict
import instrumentation as inst
import statistics, math, json
from .config import TEXT_MODEL

@inst.timed('generate score')
def generate_score(results):

    # consider the length of the question 
    # when considering the time spent on it
    times_per_word = [r['time'] / r['num_words'] for r in results]
    mean = statistics.mean(times_per_word)
    stdev = statistics.stdev(times_per_word) if len(times_per_word) > 1 else 0

    # assign incorrect/correct and confidence to each answer
    for r, tpw in zip(results, times_per_word):
        correct = str(r['user_answer']).strip().lower() == str(r['correct_answer']).strip().lower()
        r['correct'] = correct
        z = (tpw - mean) / stdev if stdev > 0 else 0
        r['confidence'] = 0.0 if not correct else 1 / (1 + math.exp(z))

    # keep track of confidence and score across topics
    topic_data = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    for r in results:
        t = topic_data[r['topic']]
        t['total'] += 1
        t['correct'] += int(r['correct'])
        t['confidences'].append(r['confidence'])

    # keep track of student's performance
    performance = []
    for topic, d in topic_data.items():
        performance.append({
            'topic': topic,
            'score': f"{d['correct']}/{d['total']}",
            'confidence': round(sum(d['confidences']) / len(d['confidences']), 2),
        })

    performance.sort(key=lambda x: x['confidence'])
    return json.dumps(performance, indent=2)

@inst.timed('generate feedback')
def generate_feedback(performance, combined):
    system = (
        "You write quiz feedback directly to the student, addressing them as "
        "'you'. You output ONLY the feedback itself - never a preamble, never "
        "a note to whoever asked for it, and never a sign-off."
    ) # ensure that feedback does not address prompter but only user

    prompt = f"""A student completed a quiz. Their performance per topic is provided
    (confidence is 0-1 where low means they were wrong or hesitant)
    Write encouraging, specific feedback directly to the student (use 'you'):
    acknowledge strong topics and gently highlight weak ones. 2-3 short paragraphs.
    Stick strictly to the performance. Suggest improvements to weaknesses using the provided content.
    Do NOT include any preamble or meta-commentary such as 'Here is your feedback'
    or 'Based on the performance provided' - begin directly with the feedback.

    PERFORMANCE:
    {performance}

    CONTENT:
    {combined}
    """

    # generate the feedback according to performance
    feedback = ollama.chat(
        model=TEXT_MODEL,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ],
    )['message']['content']

    return feedback