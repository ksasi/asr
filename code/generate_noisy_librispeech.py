import os
import argparse
import soundfile as sf
import glob
import tqdm.contrib.concurrent
import functools
from pysndfx import AudioEffectsChain
import numpy as np
import random


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--wham_dir', type=str, required=True,
                    help='Path to wham_noise root directory')
parser.add_argument('--libri_dir', type=str, required=True,
                    help='Path to librispeech root directory')


def main(args):
#def main():
    wham_noise_dir = args.wham_dir
    libri_dir = args.libri_dir
    # Get noise train dir
    subdir_tr = os.path.join(wham_noise_dir, 'tr')
    subdir_libritr = os.path.join(libri_dir, 'train-clean-100')
    # List files in that tr dir 
    sound_paths_tr = glob.glob(os.path.join(subdir_tr, '**/*.wav'),
                            recursive=True)

    sound_path_libritr = glob.glob(os.path.join(subdir_libritr, '**/**/*.flac'),
                            recursive=True)
    augment_noise(sound_path_libritr, sound_paths_tr)
    print("Completed the creation of noisy train-clean-100 partition of LibriSpeech")

    # Get noise test dir
    subdir_tt = os.path.join(wham_noise_dir, 'tt')
    subdir_libritt = os.path.join(libri_dir, 'test-clean')
    subdir_libridd = os.path.join(libri_dir, 'dev-clean')
    # List files in that tt dir 
    sound_paths_tt = glob.glob(os.path.join(subdir_tt, '**/*.wav'),
                            recursive=True)
    sound_path_libritt = glob.glob(os.path.join(subdir_libritt, '**/**/*.flac'),
                            recursive=True)
    sound_path_libridd = glob.glob(os.path.join(subdir_libridd, '**/**/*.flac'),
                            recursive=True)
    augment_noise(sound_path_libritt, sound_paths_tt)
    print("Completed the creation of noisy test-clean partition of LibriSpeech")
    augment_noise(sound_path_libridd, sound_paths_tt)
    print("Completed the creation of noisy dev-clean partition of LibriSpeech")
    
    #print(sound_paths_tt[0:4])
    #print("\n")
    #print(sound_path_libritt[0:4])


def augment_noise(sound_paths_source, sound_paths_noise):
    rand_idx = random.randint(0, len(sound_paths_noise)-1)
    sound_path2 = sound_paths_noise[rand_idx]
    tqdm.contrib.concurrent.process_map(
        functools.partial(apply_noise, sound_path2=sound_path2),
        sound_paths_source,
        chunksize=1000
    )

def apply_noise(sound_path1, sound_path2):
    s1, rate1 = sf.read(sound_path1)
    s2, rate2 = sf.read(sound_path2)
    s2 = s2[:, 0]
    target_shape = s1.shape[0]
    noisy_audio = np.add(np.resize(s1, target_shape), np.resize(s2, target_shape))
    #path_list = str.split(sound_path1, '.')
    #noisy_sound_path = path_list[0] + '_noisy' + path_list[1]
    os.remove(sound_path1)
    sf.write(sound_path1 , noisy_audio, rate1)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)