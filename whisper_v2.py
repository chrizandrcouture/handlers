import os
import sys
import time
import logging

from ts.torch_handler.base_handler import BaseHandler

import torch
import jsonpickle


class WhisperHandler(BaseHandler):
    def __init__(self, *base_args):
        super(WhisperHandler, self).__init__(*base_args)
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.alignment_heads = b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9"

    def install_packages(self):
        try:
            import whisper
        except ImportError:
            os.system(f'python3 -m pip install -r {os.path.join(self.model_dir, "requirements.txt")}')

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.manifest = context.manifest
        self.properties = context.system_properties
        self.model_dir = self.properties.get("model_dir")
        self.gpu_id = self.properties.get("gpu_id")
        self.install_packages()

        self.device = torch.device("cuda:" + str(self.gpu_id)
                                   if torch.cuda.is_available() else "cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(self.model_dir, serialized_file)

        # In case the model checkpoint is empty and weights are loaded in the init_function
        if os.path.getsize(model_pt_path) == 0 or not os.path.isfile(model_pt_path):
            model_pt_path = None

        import whisper
        self.model = whisper.load_model(model_pt_path, self.device)
        if self.alignment_heads is not None:
            self.model.set_alignment_heads(self.alignment_heads)

        self.initialized = True

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param data: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        logging.info(f"Processing batch of size: {len(data)}")
        data = [jsonpickle.decode(x.get("audio_file")) for x in data]
        return data

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_outputs = [self.model.transcribe(x, word_timestamps=True) for x in model_input]
        return model_outputs

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = []
        for data in inference_output:
            segments = data["segments"]
            for x in segments:
                x["word_timestamps"] = x.pop("words")
            postprocess_output.append({"result": segments})
        return postprocess_output

    def handle(self, data, context=None):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        processed_output =  self.postprocess(model_output)

        return processed_output


# torch-model-archiver -f --model-name whisper \
#                      --version 1 --handler handler_v2.py \
#                      --serialized-file medium.pt \
#                      --runtime python3 \
#                      --requirements-file requirements.txt
#
