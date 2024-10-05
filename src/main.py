from flask import Flask
from flask_cors import CORS

from llm_api import LLMAPI

app = Flask(__name__)
CORS(app)

if __name__ == '__main__':
    llm_api_view = LLMAPI.as_view('llm_api')
    app.add_url_rule('/llm_api', view_func=llm_api_view, methods=['POST', 'OPTIONS'])
    app.run(host='0.0.0.0', port=8090)


