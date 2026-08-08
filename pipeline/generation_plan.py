import ollama
import json
import instrumentation as inst
import streamlit as st
from .config import TEXT_MODEL

def _priority_bucket(confidence):
    """
    assign priority to topics based on confidence scores
    """
    if confidence < 0.4:
        return "high"
    if confidence < 0.7:
        return "medium"
    return "low"

def _allocate_days(topics, duration_days):
    """
    returns [(name, confidence, day_count), ...] with day_counts summing to
    duration_days, weighted towards the weakest topics.
    """
    n = len(topics)
    if n == 0 or duration_days <= 0:
        return []

    if duration_days <= n:
        return [(name, conf, 1) for name, conf in topics[:duration_days]]

    # deciding how many of the student's chosen study days go to each topic, 
    # weighted by how weak they are on it
    weights = [max(0.05, 1 - conf) for _, conf in topics]
    total_weight = sum(weights)
    quotas = [w / total_weight * duration_days for w in weights]
    days = [max(1, int(q)) for q in quotas]
    remainders = [q - int(q) for q in quotas]

    diff = duration_days - sum(days)
    if diff > 0:
        # hand out the leftover days to topics rounded down most
        order = sorted(range(n), key=lambda i: remainders[i], reverse=True)
        for i in order[:diff]:
            days[i] += 1
    else:
        # trim back starting from the topics closest to their rounded-down quota
        # since rounding up topics to 1-day minimum can exceed duration_days
        order = sorted(range(n), key=lambda i: remainders[i])
        i = 0
        while diff < 0:
            idx = order[i % n]
            if days[idx] > 1:
                days[idx] -= 1
                diff += 1
            i += 1

    return [(topics[i][0], topics[i][1], days[i]) for i in range(n)]

def _generate_topic_actions(performance, combined, retries=3):
    prompt = f"""A student completed a quiz. Their performance per topic is below
(confidence is 0-1, where low means they were wrong or hesitant).

PERFORMANCE:
{performance}

CONTENT:
{combined}

For EACH topic listed in PERFORMANCE, write 1 to 3 concrete, specific study
actions based ONLY on the CONTENT above (e.g. "re-read the section on X",
"practice deriving Y"). Order each topic's actions from most to least important.

Return ONLY JSON in this exact shape:
{{"actions": {{"<topic name exactly as given>": ["<action 1>", "<action 2>"]}}}}"""

    for retry in range(retries):
        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": "You write specific study actions as JSON. You never write a paragraph."},
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.3},
        ) # get content for the day for each topic in the learning plan

        try:
            data = json.loads(response["message"]["content"])
        except json.JSONDecodeError:
            print(f"Failed at attempt {retry}")
            continue

        actions = data.get("actions")
        if isinstance(actions, dict) and actions:

            # return the topic and its generated action
            return {
                str(topic): [str(a).strip() for a in items if str(a).strip()]
                for topic, items in actions.items()
                if isinstance(items, list) and items
            }

    return {}

@inst.timed("create plan")
def create_plan(performance, combined, duration_days=7):
    try:
        perf_list = json.loads(performance)
    except (json.JSONDecodeError, TypeError):
        perf_list = []

    # get each topic and its confidence
    topics = [(p["topic"], p["confidence"]) for p in perf_list if "topic" in p and "confidence" in p]
    if not topics:
        return []

    # get actions and allocate days
    actions_by_topic = _generate_topic_actions(performance, combined)
    allocation = _allocate_days(topics, duration_days)

    schedule = []
    day_num = 1
    for topic, confidence, day_count in allocation:

        # create the schedule organised according to the day and topic
        actions = actions_by_topic.get(topic) or ["Review your notes and the summary for this topic."]
        for i in range(day_count):
            schedule.append({
                "day": day_num,
                "topic": topic,
                "confidence": confidence,
                "priority": _priority_bucket(confidence),
                "action": actions[i % len(actions)],
            })
            day_num += 1

    return schedule

# for tagging topics in the learning plan
PRIORITY_STYLE = {
    "high": ("\U0001F534", "High priority"),
    "medium": ("\U0001F7E1", "Medium priority"),
    "low": ("\U0001F7E2", "Low priority"),
}

def _plan_to_markdown(plan):
    """
    converts learning plan to MD
    """

    lines = ["# Your Personalised Learning Plan", ""]

    for item in plan:
        icon, label = PRIORITY_STYLE.get(item['priority'], ("", ""))
        lines.append(f"## Day {item['day']} - {icon} {item['topic']} ({label})".strip())
        lines.append(f"- [ ] {item['action']}")
        lines.append("")

    return "\n".join(lines)

ss = st.session_state

def render_study_plan(plan):
    """
    visualises learning plan
    """

    # if no topics were found then return
    if not plan:
        st.success("Nice work — no weak topics found, so no study plan is needed.")
        return

    total = len(plan)
    done = sum(1 for item in plan if ss.get(f"plan_day_{item['day']}"))
    st.progress(done / total, text=f"{done}/{total} days completed")

    st.download_button(
        "Download learning plan",
        data=_plan_to_markdown(plan),
        file_name="learning_plan.md",
        mime="text/markdown",
        help="Save this plan so you can still access it after you close the app.",
    )

    # display topics according to priority
    for item in plan:
        icon, label = PRIORITY_STYLE.get(item['priority'], ("⚪", ""))
        title = f"Day {item['day']} · {icon} {item['topic']} ({label})"
        with st.expander(title, expanded=(item['day'] == 1)):
            st.checkbox(item['action'], key=f"plan_day_{item['day']}")