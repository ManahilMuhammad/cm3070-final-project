import streamlit as st
import time
from pipeline import (
    transcribe_audio, extract_slides, ocr_notes, describe_figure, # extraction 
    unload_all, release_llm, # model-related
    fuse, # utilty
    make_summary, # summary
    make_quiz, # quiz
    generate_score, generate_feedback, # feedback
    create_plan, render_study_plan # learning plan
)

st.title('CM3070 - Prototype')

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
    figure = st.file_uploader('Figure / diagram (image)', type=['png', 'jpg', 'jpeg'])

    if st.button('Process'):
        for key in ('performance', 'feedback', 'plan'):
            ss.pop(key, None)
        for key in [k for k in ss.keys() if k.startswith('plan_day_')]:
            del ss[key]

        with st.status('Processing your materials...', expanded=True) as status:
            if audio:
                status.update(label='Transcribing audio...') # checkpoint 1: transcription
                transcript = transcribe_audio(audio)
            else:
                transcript = ""

            if slides:
                status.update(label='Extracting slide text...') # checkpoint 2: slide text extraction
                slide_text = extract_slides(slides)
            else:
                slide_text = ""

            if notes:
                status.update(label='Reading handwritten notes...') # checkpoint 3: OCR
                notes_text = ocr_notes(notes)
            else:
                notes_text = ""

            unload_all() # release models

            if figure:
                status.update(label='Describing figure...') # checkpoint 4: figure description
                figure_text = describe_figure(figure)
            else:
                figure_text = ""
            release_llm(model="qwen2.5vl:latest", end_of_phase=False)

            status.update(label='Combining extracted content...') # checkpoint 5: combining extracted text
            ss.combined = fuse(transcript, slide_text, notes_text, figure_text)

            status.update(label='Generating summary...') # checkpoint 6: creating summary
            ss.summary = make_summary(ss.combined)

            status.update(label='Generating quiz...') # checkpoint 7: creating quiz
            ss.quiz = make_quiz(ss.summary)

            status.update(label='Processing complete!', state='complete') # checkpoint 8: complete

        release_llm(unload=False) # keep llama3.2 for scoring/feedback/plan
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
            ss.performance = generate_score(ss.results)

            status.update(label='Generating feedback...') # checkpoint 2: feedback generation
            ss.feedback = generate_feedback(ss.performance, ss.summary)

            status.update(label='Done!', state='complete') # checkpoint 3: coplete

        release_llm(unload=False) # keep llama3.2 warm for the learning plan

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
                ss.plan = create_plan(ss.performance, ss.summary, duration_days=duration)
                status.update(label='Done!', state='complete')

            release_llm() # release llama3.2
            st.rerun()

    # display plan
    else:
        render_study_plan(ss.plan)