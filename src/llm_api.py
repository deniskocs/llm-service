from typing import Dict

from flask import jsonify, request, Response
from flask.views import MethodView

from model_holder import ModelHolder


class LLMAPI(MethodView):

    def __init__(self, model_holder: ModelHolder):
        self.model_holder = model_holder

    def options(self) -> tuple[str, int]:
        # You can include logic for handling OPTIONS requests here if needed
        return '', 204

    def post(self) -> tuple[Response, int]:
        json_data = request.json

        if isinstance(json_data, Dict):
            output = self.model_holder.generate_text(json_data["history"])
            return jsonify({"output": output}), 200

        return jsonify({}), 400
