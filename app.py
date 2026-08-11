import streamlit as st
import time
from pipeline import (
    transcribe_audio, extract_slides, ocr_notes, describe_figure, # ingestion 
    unload_all, release_llm, # model-related
    fuse, # utilty
    make_summary, # summary
    make_quiz, # quiz
    generate_score, generate_feedback, # feedback
    create_plan, render_study_plan, # learning plan
    extract_from_file, # extraction
    VL_MODEL
)
import tempfile
import cv2
import os

st.title('CM3070 - Prototype')

ss = st.session_state
if 'stage' not in ss:
    ss.stage = 'upload'
    ss.q_index = 0
    ss.results = []

# UPLOAD SCREEN
if ss.stage == 'upload':
    st.subheader("Upload your lecture material")
    st.write("Upload one or multiple files (video, audio, slides, images)")
    
    files = st.file_uploader(
        'Upload files',
        type=['mp4', 'mov', 'avi', 'mkv', 'mp3', 'wav', 'm4a', 'aac', 'flac',
              'pdf', 'pptx', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True  # allow multiple files
    )

    if st.button('Process'):
        if not files:
            st.warning('Please upload at least one file.')
        else:
            try:
                with st.status('Processing your materials...') as status:
                    
                    all_transcripts = []
                    all_slide_text = []
                    all_notes_text = []
                    all_figures = []

                    for file in files:
                        st.info(f"Processing {file.name}...")
                        
                        extracted = extract_from_file(file)

                        # process extracted audio
                        if extracted['audio']:
                            status.update(label='Transcribing audio...') # checkpoint 1: transcription
                            transcript = transcribe_audio(extracted['audio'])
                            all_transcripts.append(transcript)
                            unload_all()
                        
                        # use extracted text directly
                        if extracted['text']:
                            status.update(label='Extracting slide text...') # checkpoint 2: slide text extraction
                            all_slide_text.append(extracted['text'])
                        
                        # describe extracted images (figures/diagrams from PDF/PPTX)
                        if extracted['images']:
                            status.update(label='Describing figures...') # checkpoint 3: figure description
                            for img in extracted['images']:
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    img.save(tmp.name)
                                    description = describe_figure(tmp.name)
                                    all_figures.append(description)
                                    print("Almost done..")
                                os.remove(tmp.name)
                                print("Okay next one!")
                            release_llm(model=VL_MODEL, end_of_phase=False)

                        # apply OCR to extracted slide frames
                        if extracted['slide_frames']:
                            status.update(label='Reading slide text...') # checkpoint 4: OCR
                            for frame in extracted['slide_frames']:
                                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                                    cv2.imwrite(tmp.name, frame)
                                    notes_text = ocr_notes(tmp.name)
                                    all_notes_text.append(notes_text)
                                os.remove(tmp.name)
                            unload_all()

                    status.update(label='Combining extracted content...') # checkpoint 5: combining extracted text
                    ss.combined = fuse(
                        " ".join(all_transcripts),
                        " ".join(all_slide_text),
                        " ".join(all_notes_text),
                        " ".join(all_figures)
                    )
                    
                    if not ss.combined.strip():
                        st.warning('No content extracted. Check your files.')
                        st.stop()
                    
                    status.update(label='Generating summary...') # checkpoint 6: creating summary
                    ss.summary = make_summary(ss.combined)
        
                    status.update(label='Generating quiz...') # checkpoint 7: creating quiz
                    ss.quiz = make_quiz(ss.summary) or []

                status.update(label='Processing complete!', state='complete') # checkpoint 8: complete
                    
                release_llm(unload=False) # keep llama3.2 for scoring/feedback/plan
                ss.stage = 'summary'
                st.rerun()

            except Exception as e:
                st.error(f'Processing failed: {type(e).__name__}: {e}')
                st.exception(e)

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