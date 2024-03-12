import os
import sys
import json
import time
import pdb
import string


def install_packages():
    try:
        from dtw import dtw
    except ImportError:
        # os.system('python3 -m pip install -U pip')
        # os.system(f'python3 -m pip install -r {os.path.join(model_dir, "requirements.txt")}')
        os.system('python3 -m pip install -r requirements.txt')
        print('System path is as follows: ', sys.path)

install_packages()
import torch
import torch.nn.functional as F

def unpack_dependencies():
    if not os.path.exists('./whisper'):
        print('unpacking dependencies')
        os.system('tar -xzf files.tar.gz')
        time.sleep(2)

unpack_dependencies()

import whisper_time as whisper
# import whisper
from whisper_time.tokenizer import get_tokenizer
import jsonpickle
from dtw import dtw
from scipy.signal import medfilt


MODEL = "medium.pt"
FP16 = True
SAMPLE_RATE = whisper.audio.SAMPLE_RATE
LANGUAGE = "English"
AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / SAMPLE_RATE

medfilt_width = 7
qk_scale = 1.0


class Whisper:
    def __init__(self, model_, device):
        self.model = whisper.load_model(model_, device)
        # self.model.eval()
        self.device = device
        self.tokenizer = get_tokenizer(self.model.is_multilingual, language=LANGUAGE)
        options = dict(language=LANGUAGE, beam_size=5, best_of=5)
        self.transcribe_options = dict(task="transcribe", **options)
        self.QKs = [None] * self.model.dims.n_text_layer

        for i, block in enumerate(self.model.decoder.blocks):
            block.cross_attn.register_forward_hook(
                # lambda _, ins, outs, index=i: self.QKs.__setitem__(index, outs[-1])
                self.forward_hook(i)
            )
        # self.model = self.model.to(device)

    def forward_hook(self, i):
        def hook(module, input, output):
            self.QKs[i] = output[-1]
        return hook

    def stt_segment(self, audio):
        # audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        print(detected_language)

        options = whisper.DecodingOptions(language=detected_language, fp16=FP16)
        result = whisper.decode(self.model, mel, options)
        # print the recognized text
        return result.text

    def tokenize(self, text, duration):
        tokens = torch.tensor(
            [*self.tokenizer.sot_sequence, self.tokenizer.timestamp_begin,] +
            self.tokenizer.encode(text) +
            [self.tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN, self.tokenizer.eot,]
        ).cuda()
        return tokens

    def transcribe(self, audio_file):
        # result = self.stt_segment(audio_file)
        # transcription = self.model.transcribe(audio_file, **self.transcribe_options)
        # self.align_words(audio_file, transcription)
        result = self.model.transcribe(audio_file)
        transcription_segments = result["segments"]
        # aligned_transcript = self.align_words(transcription_segments, segments)
        # pdb.set_trace()
        return transcription_segments


def handle(data, context=None):
    if data is None:
        # manifest = context.manifest

        # properties = context.system_properties
        # model_dir = properties.get("model_dir")
        # install_packages(model_dir)
        return None

    audio_file = jsonpickle.decode(data[0]['audio_file']) #.decode('utf-8').strip()
    # segments = jsonpickle.decode(data[0]['segments'])
    output = _service.transcribe(audio_file)
    response = {'result': output}
    return [response]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
_service = Whisper(MODEL, device)


if __name__ == "__main__":
    import numpy as np
    import librosa
    import time
    audio_file = "sample.wav"
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    # segments = json.load(open("sample.json"))
    # data = {"audio_file": jsonpickle.encode(np.array(y)), "segments": jsonpickle.encode(segments)}
    data = {"audio_file": jsonpickle.encode(np.array(y))}
    begin = time.time()
    response = handle([data])
    pdb.set_trace()
    end = time.time()
    print(response, f"\nTime taken = {end-begin}")


# torch-model-archiver -f --model-name whisper --version 1 --handler handler.py:handle --serialized-file medium.pt --runtime python3 --requirements-file requirements.txt --extra-files files.tar.gz
