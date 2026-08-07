import ollama
import instrumentation as inst

LLM_MODEL = "llama3.2"

def release_llm(model=LLM_MODEL, label="run", end_of_phase=True, unload=True):
    """
    free up GPU/RAM.
    unload=False keeps the model
    used between screens that reuse same model so it isn't reloaded each time
    """
    if unload:
        try:
            ollama.generate(model=model, prompt="", keep_alive=0)
        except Exception as exc:
            print(f"[release_llm] could not unload {model}: {exc}")
            return False

    if end_of_phase:
        inst.report()
        inst.save_run(label)
        inst.reset()

    return True