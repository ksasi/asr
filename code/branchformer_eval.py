from torchmetrics.text import CharErrorRate, WordErrorRate
import os
import argparse
import soundfile as sf
import glob
import numpy as np
from espnet2.bin.asr_inference import Speech2Text

# https://huggingface.co/pyf98/librispeech_100_ctc_e_branchformer

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--libri_dir', type=str, required=True,
                    help='Path to librispeech partition root directory')

def perf_metrics(preds, targets):
    cer = CharErrorRate()
    wer = WordErrorRate()
    return cer(preds, targets), wer(preds, targets)

def main(args):
    model_branchformer = Speech2Text.from_pretrained(
        "pyf98/librispeech_100_ctc_e_branchformer"
    )
    libri_dir = args.libri_dir
    dirs = [dir for dir in glob.glob(os.path.join(libri_dir, '**/**'), recursive=False) if os.path.isdir(dir)]

    cer_list_branchformer = []
    wer_list_branchformer = []
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

        preds = []
        for sound_file in np.sort(sound_file_paths):
            speech, rate = sf.read(sound_file)
            tscp_branchformer, *_ = model_branchformer(speech)[0]
            preds.append(tscp_branchformer)
        cer_branchformer, wer_branchformer = perf_metrics(preds, targets)
        cer_list_branchformer.append(cer_branchformer)
        wer_list_branchformer.append(wer_branchformer)
    return np.mean(cer_list_branchformer), np.mean(wer_list_branchformer)

        


if __name__ == "__main__":
    args = parser.parse_args()
    cer_mean_branchformer, wer_mean_branchformer= main(args)
    print("Mean CER on wham-noisy test-clean of LibriSpeech for branchformer model :", cer_mean_branchformer, flush=True)
    print("\n")
    print("Mean WER on wham-noisy test-clean of LibriSpeech for branchformer model :", wer_mean_branchformer, flush=True)
    



