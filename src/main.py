import os
import sys

from flask import Flask
from flask_cors import CORS

from huggingface_hub import login
from llm_api import LLMAPI
from model_holder import ModelHolder

token = os.getenv("HUGGING_FACE_TOKEN")
if token is None:
    print("Error: HUGGING_FACE_TOKEN is not set", file=sys.stderr)
    sys.exit(1)

login(token)

app = Flask(__name__)
CORS(app)


if __name__ == '__main__':
    model_holder = ModelHolder()
    model_holder.generate_text([{"role": "user", "content": "Напиши короткую историю про землю на русском."}])

    llm_api_view = LLMAPI.as_view('llm_api', model_holder=model_holder)
    app.add_url_rule('/llm_api', view_func=llm_api_view, methods=['POST', 'OPTIONS'])

    app.run(host='0.0.0.0', port=8090)
