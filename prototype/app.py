import streamlit as st
import time
import pipeline

st.title('CM3070 - Prototype')

ss = st.session_state
if 'stage' not in ss:
    ss.stage = 'upload'
    ss.q_index = 0
    ss.results = []

# for tagging topics in the learning plan
PRIORITY_STYLE = {
    "high": ("\U0001F534", "High priority"),
    "medium": ("\U0001F7E1", "Medium priority"),
    "low": ("\U0001F7E2", "Low priority"),
}

def plan_to_markdown(plan):
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
        data=plan_to_markdown(plan),
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

# UPLOAD SCREEN
if ss.stage == 'upload':
    audio = st.file_uploader('Lecture audio', type=['mp3', 'wav', 'm4a'])
    slides = st.file_uploader('Slides (PDF or PPTX)', type=['pdf', 'pptx'])
    notes = st.file_uploader('Notes (image)', type=['png', 'jpg', 'jpeg'])
    figure = st.file_uploader('Figure / diagram (image)', type=['png', 'jpg', 'jpeg'])

    if st.button('Process'):
        for key in ('performance', 'feedback', 'plan'):
            ss.pop(key, None)
        for key in [k for k in ss.keys() if k.startswith('plan_day_')]:
            del ss[key]

        with st.status('Processing your materials...', expanded=True) as status:
            if audio:
                status.update(label='Transcribing audio...') # checkpoint 1: transcription
                transcript = pipeline.transcribe_audio(audio)
            else:
                transcript = ""

            if slides:
                status.update(label='Extracting slide text...') # checkpoint 2: slide text extraction
                slide_text = pipeline.extract_slides(slides)
            else:
                slide_text = ""

            if notes:
                status.update(label='Reading handwritten notes...') # checkpoint 3: OCR
                notes_text = pipeline.ocr_notes(notes)
            else:
                notes_text = ""

            pipeline.unload_all() # release models

            if figure:
                status.update(label='Describing figure...') # checkpoint 4: figure description
                figure_text = pipeline.describe_figure(figure)
            else:
                figure_text = ""
            pipeline.release_llm(model="qwen2.5vl:latest", end_of_phase=False)

            status.update(label='Combining extracted content...') # checkpoint 5: combining extracted text
            ss.combined = pipeline.fuse(transcript, slide_text, notes_text, figure_text)

            status.update(label='Generating summary...') # checkpoint 6: creating summary
            ss.summary = pipeline.make_summary(ss.combined)

            status.update(label='Generating quiz...') # checkpoint 7: creating quiz
            ss.quiz = pipeline.make_quiz(ss.summary)

            status.update(label='Processing complete!', state='complete') # checkpoint 8: complete

        pipeline.release_llm() # release llama3.2
        ss.stage = 'summary'
        st.rerun()

# SUMMARY SCREEN
elif ss.stage == 'summary':
    st.header('Summary')
    st.markdown(ss.summary)

    if st.button('Start quiz'):
        ss.stage = 'quiz'
        ss.q_start = time.time() 
        st.rerun()

# QUIZ SCREEN
elif ss.stage == 'quiz':
    q = ss.quiz[ss.q_index]
    st.header(f'Question {ss.q_index + 1} of {len(ss.quiz)}')
    st.write(q['question'])

    # true-false
    if q['type'] == 'true-false':
        answer = st.radio('Choose: ', ['True', 'False'], key=f"q{ss.q_index}")

    # MCQ
    elif q.get('options'):
        answer = st.radio('Choose:', q['options'], key=f'q{ss.q_index}')

    # fill-in-the-blank / short-answer
    else:
        answer = st.text_input('Your answer:', key=f'q{ss.q_index}')

    if st.button('Submit'):
        elapsed = time.time() - ss.q_start

        ss.results.append({
            'question': q['question'], 'type': q['type'], 'topic': q['topic'],
            'correct_answer': q['answer'], 'user_answer': answer,
            'time': elapsed, 'num_words': len(q['question'].split()),
        })

        ss.q_index += 1

        # if not last question then record time and show next one
        if ss.q_index < len(ss.quiz):
            ss.q_start = time.time()          
            st.rerun()

        # if last question then show result
        else:
            ss.stage = 'results'
            st.rerun()

# RESULTS SCREEN
elif ss.stage == 'results':
    st.header('Your Results')

    if 'performance' not in ss:
        with st.status('Preparing your results...', expanded=True) as status:
            status.update(label='Scoring your answers...') # checkpoint 1: scoring
            ss.performance = pipeline.generate_score(ss.results)

            status.update(label='Generating feedback...') # checkpoint 2: feedback generation
            ss.feedback = pipeline.generate_feedback(ss.performance, ss.summary)

            status.update(label='Done!', state='complete') # checkpoint 3: coplete

        pipeline.release_llm() # release llama3.2

    # FEEDBACK
    st.subheader('Feedback')
    st.write(ss.feedback)

    # LEARNING PLAN
    st.subheader('Your Personalised Learning Plan')

    # create plan if not already done
    if 'plan' not in ss:

        # ask student how long they want the plan to be
        duration = st.slider(
            'How many days would you like your learning plan to span?',
            min_value=2, max_value=31, value=15,
        )

        if st.button('Generate my learning plan'):
            with st.status('Generating your learning plan...', expanded=True) as status:
                status.update(label='Generating learning plan...')
                ss.plan = pipeline.create_plan(ss.performance, ss.summary, duration_days=duration)
                status.update(label='Done!', state='complete')

            pipeline.release_llm() # release llama3.2
            st.rerun()

    # display plan
    else:
        render_study_plan(ss.plan)