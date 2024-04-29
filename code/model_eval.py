from torchmetrics.text import CharErrorRate, WordErrorRate
import os
import argparse
import soundfile as sf
import glob
import numpy as np
from speechbrain.inference.ASR import EncoderASR, EncoderDecoderASR

# wav2vec2 - https://huggingface.co/speechbrain/asr-wav2vec2-librispeech
# Conformer - https://huggingface.co/speechbrain/asr-conformer-transformerlm-librispeech

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--libri_dir', type=str, required=True,
                    help='Path to librispeech partition root directory')

def perf_metrics(preds, targets):
    cer = CharErrorRate()
    wer = WordErrorRate()
    return cer(preds, targets), wer(preds, targets)

def main(args):
    model_wav2vec2 = EncoderASR.from_hparams(
        "speechbrain/asr-wav2vec2-librispeech",  run_opts={"device":"cuda"}
    )
    model_conformer = EncoderDecoderASR.from_hparams(source="speechbrain/asr-conformer-transformerlm-librispeech", 
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech",  run_opts={"device":"cuda"})
    libri_dir = args.libri_dir
    dirs = [dir for dir in glob.glob(os.path.join(libri_dir, '**/**'), recursive=False) if os.path.isdir(dir)]

    cer_list_wav2vec2 = []
    wer_list_wav2vec2 = []
    cer_list_conformer = []
    wer_list_conformer = []
    for root_dir in dirs:
        sound_file_paths = glob.glob(os.path.join(root_dir, '*.flac'),
                            recursive=True)
        target_file_path = glob.glob(os.path.join(root_dir, '*.txt'),
                            recursive=True)
        targets = []
        file = open(target_file_path[0], 'r')
        for line in file:
            targets.append(" ".join(line.split()[1:-1]))
        file.close()

        preds_wav2vec2 = []
        preds_conformer = []
        for sound_file in np.sort(sound_file_paths):
            tscp_wav2vec2 = model_wav2vec2.transcribe_file(sound_file)
            preds_wav2vec2.append(tscp_wav2vec2)

            tscp_conformer = model_conformer.transcribe_file(sound_file)
            preds_conformer.append(tscp_conformer)
        cer_wav2vec2, wer_wav2vec2 = perf_metrics(preds_wav2vec2, targets)
        cer_conformer, wer_conformer = perf_metrics(preds_conformer, targets)
        cer_list_wav2vec2.append(cer_wav2vec2)
        wer_list_wav2vec2.append(wer_wav2vec2)
        cer_list_conformer.append(cer_conformer)
        wer_list_conformer.append(wer_conformer)
    return np.mean(cer_list_wav2vec2), np.mean(wer_list_wav2vec2), np.mean(cer_list_conformer), np.mean(wer_list_conformer)

        


if __name__ == "__main__":
    args = parser.parse_args()
    cer_mean_wav2vec2, wer_mean_wav2vec2, cer_mean_conformer, wer_mean_conformer = main(args)
    print("Mean CER on wham-noisy test-clean of LibriSpeech for wav2vec2 model :", cer_mean_wav2vec2, flush=True)
    print("\n")
    print("Mean WER on wham-noisy test-clean of LibriSpeech for wav2vec2 model :", wer_mean_wav2vec2, flush=True)
    print("\n")
    print("Mean CER on wham-noisy test-clean of LibriSpeech for conformer model :", cer_mean_conformer, flush=True)
    print("\n")
    print("Mean WER on wham-noisy test-clean of LibriSpeech for conformer model :", wer_mean_conformer, flush=True)
    


