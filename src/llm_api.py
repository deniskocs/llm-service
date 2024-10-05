from flask import jsonify, request
from flask.views import MethodView

from model_holder import model_holder


class LLMAPI(MethodView):

    def options(self):
        # You can include logic for handling OPTIONS requests here if needed
        return '', 204

    def post(self):
        json_data = request.json

        output = model_holder.generate_text(json_data["history"])

        return jsonify({"output": output}), 200