import torch
import torchaudio
import torchaudio.transforms as T
import argparse
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(path_file, target_sr=16000): # i had to load the audio using load() and then resampled it to 165kHz which was used in Whisper
    wfm, sr = torchaudio.load(path_file)
    if sr != target_sr:
        resample = T.Resample(sr, target_sr)
        wfm = resample(wfm)
    return wfm.to(device), target_sr

def spectral_gate_denoise(wfm, sr, n_fft=1024, hop_length=512):# this function is used to perform spectral gating noise reduction
    spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None).to(device) # it does this by calculating the STFT to get the spectrogram
    inverse_transform = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(device)
    # i've used the spectogram to reconstruct the main objectified audio signal
    spec = spec_transform(wfm)
    mag, phase = torch.abs(spec), torch.angle(spec) # by getting the values of magnitude and phase we'll be able to understand the audio better

    # this is optional - I added this so that it estimates the bg noise. so the first step the pipe does it to understand how much BG audio is present
    noise_frames = int((0.5 * sr) / hop_length)
    noise_profile = mag[:, :, :noise_frames].mean(dim=-1, keepdim=True)# I averaged the magnitude so we get an average noise level and that will be subtracted from our original track.

    # Spectral gating
    denoised_mag = torch.clamp(mag - noise_profile, min=0.0)

    # Reconstruct
    complex_spec = denoised_mag * torch.exp(1j * phase)
    denoised_wfm = inverse_transform(complex_spec)
    return denoised_wfm

def save_audio(wfm, path, sr):
    torchaudio.save(path, wfm.cpu(), sample_rate=sr)

def main():
    parser = argparse.ArgumentParser(description="1. Denoise 2. Enhance - The WAV File")
    parser.add_argument("input", type=str, help="Path : WAV file")
    parser.add_argument("output", type=str, help="Path : save output WAV file")
    args = parser.parse_args()

    # Load and process
    wfm, sr = load_audio(args.input)
    denoised = spectral_gate_denoise(wfm, sr)
    save_audio(denoised, args.output, sr)
    print(f"[✓] Saved enhanced audio to {args.output}")

if __name__ == "__main__":
    main()
