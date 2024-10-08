import ollama

class Ollama():
    def __init__(self) -> None:
        pass

    def generate(prompt, model='llama3.2:1b'):
        stream = ollama.generate(
            model=model,
            prompt=prompt,
            stream=True,
        )
        
        # Yield each chunk as it streams and print to console
        for chunk in stream:
            content = chunk['response']
            yield content

    def list_models():
        models = ollama.list()
        return models
    
    def chat(messages, model='llama3.2:1b'):
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            content = chunk['message']['content']
            yield content

    def info(model):
        info = ollama.show(model)
        return info
    
    def delete(model):
        result = ollama.delete(model)
        return result
    
    def pull(model):
        stream = ollama.pull(model, stream=True)

        for chunk in stream:
            content = chunk['response']
            yield content