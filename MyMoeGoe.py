from scipy.io.wavfile import write
from mel_processing import spectrogram_torch
from text import text_to_sequence, _clean_text
from models import SynthesizerTrn
import utils
import commons
import sys
import re
import torch
from torch import no_grad, LongTensor
import logging

from random import randint
import os
import json
from torch import no_grad, LongTensor
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import socketserver
import http.server
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.getLogger('numba').setLevel(logging.WARNING)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


def voice_conversion():
    audio_path = input('Path of an audio file to convert:\n')
    print_speakers(speakers)
    audio = utils.load_audio_to_torch(
        audio_path, hps_ms.data.sampling_rate)

    originnal_id = get_speaker_id('Original speaker ID: ')
    target_id = get_speaker_id('Target speaker ID: ')
    out_path = input('Path to save: ')

    y = audio.unsqueeze(0)

    spec = spectrogram_torch(y, hps_ms.data.filter_length,
                             hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                             center=False)
    spec_lengths = LongTensor([spec.size(-1)]).to(device)
    sid_src = LongTensor([originnal_id]).to(device)

    with no_grad():
        sid_tgt = LongTensor([target_id]).to(device)
        audio = net_g_ms.voice_conversion(spec.to(device), spec_lengths, sid_src=sid_src.to(device),
                                          sid_tgt=sid_tgt.to(device))[
            0][0, 0].data.to(device).float().cpu().numpy()
    return audio, out_path


def generateWav(text, file_path, idx, length, noise, noisew):
    if n_symbols != 0:
        if not emotion_embedding:
            # choice = input('TTS or VC? (t/v):')
            choice = 't'
            if choice == 't':
                # length_scale, text = get_label_value(
                #     text, 'LENGTH', 1, 'length scale')
                # noise_scale, text = get_label_value(
                #     text, 'NOISE', 0.667, 'noise scale')
                # noise_scale_w, text = get_label_value(
                #     text, 'NOISEW', 0.8, 'deviation of noise')

                length_scale, text = get_label_value(
                    text, 'LENGTH', length, 'length scale')
                noise_scale, text = get_label_value(
                    text, 'NOISE', noise, 'noise scale')
                noise_scale_w, text = get_label_value(
                    text, 'NOISEW', noisew, 'deviation of noise')

                cleaned, text = get_label(text, 'CLEANED')

                stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                print_speakers(speakers, escape)
                # speaker_id = get_speaker_id('Speaker ID: ')
                # out_path = input('Path to save: ')

                speaker_id = idx
                out_path = file_path

                with no_grad():
                    x_tst = stn_tst.unsqueeze(0).to(device)
                    x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                    sid = LongTensor([speaker_id]).to(device)
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.to(
                        device).float().cpu().numpy()

            elif choice == 'v':
                audio, out_path = voice_conversion()

            write(out_path, hps_ms.data.sampling_rate, audio)
            print('Successfully saved!')
        else:
            import os
            import librosa
            import numpy as np
            from torch import FloatTensor
            import audonnx
            w2v2_folder = input('Path of a w2v2 dimensional emotion model: ')
            w2v2_model = audonnx.load(os.path.dirname(w2v2_folder))
            while True:
                choice = input('TTS or VC? (t/v):')
                if choice == 't':
                    text = input('Text to read: ')
                    if text == '[ADVANCED]':
                        text = input('Raw text:')
                        print('Cleaned text is:')
                        ex_print(_clean_text(
                            text, hps_ms.data.text_cleaners), escape)
                        continue

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.667, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                    print_speakers(speakers, escape)
                    speaker_id = get_speaker_id('Speaker ID: ')

                    emotion_reference = input('Path of an emotion reference: ')
                    if emotion_reference.endswith('.npy'):
                        emotion = np.load(emotion_reference)
                        emotion = FloatTensor(emotion).unsqueeze(0)
                    else:
                        audio16000, sampling_rate = librosa.load(
                            emotion_reference, sr=16000, mono=True)
                        emotion = w2v2_model(audio16000, sampling_rate)[
                            'hidden_states']
                        emotion_reference = re.sub(
                            r'\..*$', '', emotion_reference)
                        np.save(emotion_reference, emotion.squeeze(0))
                        emotion = FloatTensor(emotion).to(device)

                    out_path = input('Path to save: ')

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0).to(device)
                        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                        sid = LongTensor([speaker_id]).to(device)
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w,
                                               length_scale=length_scale, emotion_embedding=emotion)[0][0, 0].data.to(
                            device).float().cpu().numpy()

                elif choice == 'v':
                    audio, out_path = voice_conversion()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')
    else:
        model = input('Path of a hubert-soft model: ')
        from hubert_model import hubert_soft
        hubert = hubert_soft(model)

        while True:
            audio_path = input('Path of an audio file to convert:\n')

            if audio_path != '[VC]':
                import librosa
                if use_f0:
                    audio, sampling_rate = librosa.load(
                        audio_path, sr=hps_ms.data.sampling_rate, mono=True)
                    audio16000 = librosa.resample(
                        audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio16000, sampling_rate = librosa.load(
                        audio_path, sr=16000, mono=True)

                print_speakers(speakers, escape)
                target_id = get_speaker_id('Target speaker ID: ')
                out_path = input('Path to save: ')
                length_scale, out_path = get_label_value(
                    out_path, 'LENGTH', 1, 'length scale')
                noise_scale, out_path = get_label_value(
                    out_path, 'NOISE', 0.1, 'noise scale')
                noise_scale_w, out_path = get_label_value(
                    out_path, 'NOISEW', 0.1, 'deviation of noise')

                from torch import inference_mode, FloatTensor
                import numpy as np
                with inference_mode():
                    units = hubert.units(FloatTensor(audio16000).unsqueeze(
                        0).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                    if use_f0:
                        f0_scale, out_path = get_label_value(
                            out_path, 'F0', 1, 'f0 scale')
                        f0 = librosa.pyin(audio, sr=sampling_rate,
                                          fmin=librosa.note_to_hz('C0'),
                                          fmax=librosa.note_to_hz('C7'),
                                          frame_length=1780)[0].to(device)
                        target_length = len(units[:, 0])
                        f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                                     np.arange(0, len(f0)), f0)) * f0_scale
                        units[:, 0] = f0 / 10

                stn_tst = FloatTensor(units).to(device)
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0).to(device)
                    x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                    sid = LongTensor([target_id]).to(device)
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.to(
                        device).float().cpu().numpy()

            else:
                audio, out_path = voice_conversion()

            write(out_path, hps_ms.data.sampling_rate, audio)
            print('Successfully saved!')
            ask_if_continue()


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        parsed_data = json.loads(post_data.decode())
        print("Received POST request:")
        for key in parsed_data:
            print(key, "=", parsed_data[key])

        idx = int(parsed_data['index'])
        text = parsed_data['text']
        random_id = '-'.join(''.join('{:x}'.format(randint(0, 15)) for _ in range(y)) for y in [8, 4, 4, 4, 8])
        file_path = "temp/" + str(random_id) + ".wav"

        if 'length' in parsed_data.keys():
            length = (parsed_data['length'])
        else:
            length = 1

        if 'noise' in parsed_data.keys():
            noise = (parsed_data['noise'])
        else:
            noise = 0.667

        if 'noisew' in parsed_data.keys():
            noisew = (parsed_data['noisew'])
        else:
            noisew = 0.8

        # dCode = '[ZH]你好,旅行者，我是派蒙，很高兴见到你[ZH]'
        generateWav(text, file_path, idx, length, noise, noisew)
        file_path = "temp/" + str(random_id) + ".wav"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                file_data = f.read()
                self.send_response(200)
                self.send_header('Content-type', 'audio/wav')
                self.send_header('Content-Length', str(len(file_data)))
                self.end_headers()
                self.wfile.write(file_data)
                f.close()
                os.remove("temp/" + str(random_id) + ".wav")
        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    with socketserver.TCPServer(("", 8000), RequestHandler) as httpd:
        print("Server started at localhost:8000")
        httpd.serve_forever()


if __name__ == '__main__':
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    model = "genshi/G_953000.pth"
    config = "genshi/config.json"

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model).to(device)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    generateWav("开始", "temp/1.wav", 551, 1, 0.667, 0.8)

    # 起服务
    server_thread = threading.Thread(target=start_server)
    server_thread.start()
