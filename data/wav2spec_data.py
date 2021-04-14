import os, glob

from PIL import Image

from utils.get_audio import GetAudio


data_format = "*/*.flac"
sampling_rate = 16000
data_path = "/workspace/db/audio/Libri/LibriSpeech/train-clean-360/"
_, id_names, _ = next(os.walk(data_path))

get_audio = GetAudio(data_path)


def make_imgs(out_data_path="./audio/aimg/"):
    if os.path.exists(out_data_path) is False: os.mkdir(out_data_path)
    for i, id_name in enumerate(id_names):
        print(" "*70, end="\r")
        print("Creating log mel spectogramm imgs: %0.2f" % ((i+1)/len(id_names)) + chr(37), end="\r")
        if os.path.exists(out_data_path + f'{i:04}' + "/") is False: os.mkdir(out_data_path + f'{i:04}' + "/")
        file_paths = glob.glob(os.path.join(data_path, id_name, data_format))
        for j, file_path in enumerate(file_paths):
            mel = get_audio.wav2mel(file_path)
            img = Image.fromarray(mel*255.).convert("L")
            img.save(out_data_path + f'{i:04}' + "/" + f'{j:04}' + ".png")


if __name__ == "__main__":
    make_imgs()