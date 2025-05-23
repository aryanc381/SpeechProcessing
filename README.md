# 🎧 Audio Denoising and Enhancement with Spectral Gating (PyTorch + Torchaudio)

This script removes background noise and enhances the quality of a `.wav` audio file using **spectral gating** in PyTorch and Torchaudio.

---

## 🚀 Features

* 📥 Takes a noisy `.wav` file as input
* 🧠 Applies **spectral gating** to suppress background noise
* 🎧 Outputs a **cleaned and enhanced** `.wav` file
* ⚡ Runs on **GPU (CUDA)** if available

---

## 🛠️ How It Works

1. **Load Audio:**
   Uses `torchaudio.load()` and resamples to 16kHz (default for models like Whisper).

2. **Spectral Gating:**
   Converts waveform to spectrogram → estimates background noise → subtracts it → reconstructs waveform.

3. **Save Clean Audio:**
   Stores the denoised waveform as a `.wav` file.

---

## 🧱 File Structure

```
audio_denoise_enhance.py   # Main script
input.wav                  # Example input (your noisy audio)
output.wav                 # Output (cleaned audio)
```

---

## ▶️ How to Run

### 1. Install dependencies:

```bash
pip install torch torchaudio
```

> Make sure your system has a GPU with CUDA if you want GPU acceleration.

---

### 2. Run the script:

```bash
python denoise.py input.wav output.wav
```

* `input.wav`: Path to your noisy audio file
* `output.wav`: Path where cleaned audio will be saved

---

## 📌 Notes

* The first **0.5 seconds** of the input audio are assumed to contain background noise only (used for noise estimation).
* Spectral gating helps suppress constant low-level noise like fan hum, air conditioning, etc.
* You can modify `n_fft` or `hop_length` in the `spectral_gate_denoise()` function for finer control.

---

## 🧪 Example

```bash
python denoise.py noisy_speech.wav cleaned_speech.wav
```

---

## 🧠 What is Spectral Gating?

> Spectral gating is a noise reduction technique that suppresses low-energy (noisy) frequencies in a spectrogram while preserving high-energy components (speech/music).

---
