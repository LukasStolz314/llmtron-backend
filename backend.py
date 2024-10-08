from flask import Flask, request, jsonify, stream_with_context, Response
from custom_ollama import *

app = Flask(__name__)
        
@app.route('/generate', methods=['POST'])
def generate(): 
    data = request.json
    model_type = data.get('model_type') or 'llama3.2:1b'
    prompt = data.get('prompt')

    stream = Ollama.generate(prompt, model_type)
    return Response(stream_with_context(stream), content_type='text/plain')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    model_type = data.get('model_type') or 'llama3.2:1b'
    messages = data.get('messages')
    stream = Ollama.chat(messages, model_type)
    return Response(stream_with_context(stream), content_type='text/plain')

@app.route('/pull', methods=['POST'])
def pull():
    data = request.json
    model_name = data.get('model')
    stream = Ollama.pull(model_name)
    return Response(stream_with_context(stream), content_type='text/plain')

@app.route('/list_models', methods=['GET'])
def list_models():
    models = Ollama.list_models()
    result = [model['name'] for model in models['models']]
    return jsonify({'models': result})

@app.route('/info', methods=['GET'])
def info():
    model_name = request.args.get('model', default=None, type=str)
    info = Ollama.info(model_name)
    return jsonify({'info': info})

@app.route('/delete', methods=['DELETE'])
def delete():
    model_name = request.args.get('model', default=None, type=str)
    delete = Ollama.delete(model_name)
    return delete

if __name__ == '__main__':
    app.run(debug=True)
