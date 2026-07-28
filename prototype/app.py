import streamlit as st
import time
import pipeline

st.title('CM3070 - Feature Prototype')

ss = st.session_state
if 'stage' not in ss:
    ss.stage = 'upload'
    ss.q_index = 0
    ss.results = []

# UPLOAD SCREEN
if ss.stage == 'upload':
    audio = st.file_uploader('Lecture audio', type=['mp3', 'wav', 'm4a'])
    slides = st.file_uploader('Slides (PDF or PPTX)', type=['pdf', 'pptx'])
    notes = st.file_uploader('Notes (image)', type=['png', 'jpg', 'jpeg'])
    figure = st.file_uploader("Figure / diagram (image)", type=["png", "jpg", "jpeg"])

    if st.button('Process'):
        with st.spinner('Running...'):
            transcript = pipeline.transcribe_audio(audio) if audio else ""
            slide_text = pipeline.extract_slides(slides) if slides else ""
            notes_text = pipeline.ocr_notes(notes) if notes else ""

            pipeline.unload_all()

            figure_text = pipeline.describe_figure(figure) if figure else ""
            pipeline.release_llm(model="qwen2.5vl:latest", end_of_phase=False)

            ss.combined = pipeline.fuse(transcript, slide_text, notes_text, figure_text)

            ss.summary = pipeline.make_summary(ss.combined)
            ss.quiz = pipeline.make_quiz(ss.summary)

            topics = pipeline.extract_topics(ss.summary)
            for item in ss.quiz:
                item['topic'] = pipeline.tag_topic(item['question'], topics)

        pipeline.release_llm()
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

    if q['type'] == 'true-false':
        answer = st.radio('Choose: ', ['True', 'False'], key=f"q{ss.q_index}")
    elif q.get('options'):
        answer = st.radio('Choose:', q['options'], key=f'q{ss.q_index}')
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
        if ss.q_index < len(ss.quiz):
            ss.q_start = time.time()          
            st.rerun()
        else:
            ss.stage = 'results'
            st.rerun()

# RESULTS SCREEN
elif ss.stage == 'results':
    st.header('Your Results')
    performance = pipeline.generate_score(ss.results)
    feedback = pipeline.generate_feedback(performance, ss.summary)
    plan = pipeline.create_plan(performance, ss.summary)
    pipeline.release_llm()
    st.subheader('Feedback')
    st.write(feedback)
    st.subheader('Your Personalised Study Plan')
    st.write(plan)