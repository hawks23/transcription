import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, sosfiltfilt
from pydub import AudioSegment
from pydub.effects import normalize
import subprocess
import adaptfilt as adf
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

def convert_webm_to_wav(input_file, output_file):
    """
    Converts a WebM file to WAV format using FFmpeg.
    """
    print(f"Starting conversion of {input_file} to WAV format...")
    try:
        # Run FFmpeg command to convert WebM to WAV
        subprocess.run(
            ['ffmpeg', '-i', input_file, '-ar', '16000', '-ac', '1', output_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Conversion complete. Saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
        raise RuntimeError("FFmpeg conversion failed.")
    
    return output_file


def get_audio_data(file_path):
    print(f"Loading audio data from {file_path}...")
    data, rate = librosa.load(file_path, sr=None)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)  # Convert to mono if stereo
    print(f"Audio data loaded. Sampling rate: {rate} Hz")
    return data, rate

def normalize_audio(data):
    """
    Normalize the audio data to ensure consistent loudness.
    
    Args:
        data (numpy array): Audio data.
        
    Returns:
        numpy array: Normalized audio data.
    """
    peak = np.max(np.abs(data))
    if peak > 0:
        normalized_data = data / peak  # Scale data to [-1, 1]
        print("Audio normalized.")
        return normalized_data
    else:
        print("Audio normalization skipped (silent audio).")
        return data


def noise_reduction(data, rate):
    print("Starting noise reduction...")
    reduced_data = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
    
    # Normalize after noise reduction
    reduced_data = normalize_audio(reduced_data)
    
    # Save noise-reduced output (optional)
    noise_reduced_path = 'noise_reduced.wav'
    sf.write(noise_reduced_path, reduced_data, rate)
    print(f"Noise reduction complete. Saved as {noise_reduced_path}")
    
    return reduced_data


def bandpass_filter(data, rate, order=4):
    print("Starting bandpass filtering...")
    nyquist = 0.5 * rate
    lowcut = max(20, 0.02 * rate)  # Automatically determine lowcut (at least 20 Hz)
    highcut = min(0.5 * rate, 0.45 * rate)  # Automatically determine highcut
    sos = butter(order, [lowcut, highcut], btype='band', fs=rate, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    
    # Normalize after bandpass filtering
    filtered_data = normalize_audio(filtered_data)
    
    # Save filtered output (optional)
    output_path = 'bandpass_filtered.wav'
    sf.write(output_path, filtered_data, rate)
    print(f"Bandpass filtering complete. Saved as {output_path}")
    
    return filtered_data


def slow_down_audio(input_file, processed_dir, slowdown_factor=1.2):
    """
    Slows down the audio by a given factor without changing its pitch.
    
    Args:
        input_file (str): Path to the input audio file.
        processed_dir (str): Directory where the slowed-down audio will be saved.
        slowdown_factor (float): Factor by which to slow down the audio (e.g., 1.2 for 20% slower).
        
    Returns:
        str: Path to the saved slowed-down audio file.
    """
    print(f"Slowing down audio from {input_file} by {int((slowdown_factor - 1) * 100)}%...")
    
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    # Calculate new playback rate (slowing down factor is inverted for speedup)
    new_playback_rate = int(audio.frame_rate / slowdown_factor)
    
    # Apply new playback rate without affecting pitch
    slowed_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_playback_rate})
    
    # Set frame rate back to original to maintain pitch
    slowed_audio = slowed_audio.set_frame_rate(audio.frame_rate)
    
    # Create directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Save the slowed-down audio in the processed directory
    output_file = os.path.join(processed_dir, "slowed_audio.wav")
    slowed_audio.export(output_file, format="wav")
    
    print(f"Slowed-down audio saved as {output_file}")
    
    return output_file


def echo_cancellation(input_file, ref_signal, processed_dir, output_file_name="echo_canceled.wav", step_size=0.1, filter_taps=128):
    """
    Perform echo cancellation using the NLMS adaptive filter, normalize the result, and save it.

    Args:
        input_file (str): Path to the input audio file (microphone signal with echo).
        ref_signal (numpy array): The reference signal (e.g., speaker output causing the echo).
        processed_dir (str): Directory where the echo-canceled audio will be saved.
        output_file_name (str): Name of the output file for echo-canceled audio.
        step_size (float): Step size for the NLMS algorithm (controls adaptation speed).
        filter_taps (int): Number of filter taps for the adaptive filter.

    Returns:
        str: Path to the saved echo-canceled audio file.
    """
    print(f"Starting echo cancellation for {input_file}...")
    
    # Load microphone signal
    mic_signal, rate = librosa.load(input_file, sr=None)
    
    # Apply NLMS adaptive filter
    y, e, w = adf.nlms(ref_signal, mic_signal, M=filter_taps, step=step_size)  # Filtered output
    
    # Normalize the filtered output
    normalized_signal = normalize_audio(y)
    
    # Create directory if it doesn't exist
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Save the normalized and echo-canceled audio as a WAV file
    output_file_path = os.path.join(processed_dir, output_file_name)
    sf.write(output_file_path, normalized_signal, rate)  # Save normalized signal
    
    print(f"Echo cancellation complete. Saved as {output_file_path}")
    
    return output_file_path


def save_to_processed_directory(input_file_name, *file_paths):
    """
    Saves processed audio files into a directory named after the input file.
    
    Args:
        input_file_name (str): Name of the input audio file (e.g., '303.webm').
        file_paths (list): Paths of processed files to be moved.
        
    Returns:
        list: Paths of the files in the new processed directory.
    """
    # Create a directory named "<audio_name>_processed"
    base_name = os.path.splitext(os.path.basename(input_file_name))[0]
    processed_dir = f"{base_name}_processed"
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)  # Create directory if it doesn't exist
    
    new_paths = []
    
    print(f"Saving processed files into directory: {processed_dir}")
    
    for file_path in file_paths:
        new_file_path = os.path.join(processed_dir, os.path.basename(file_path))
        os.replace(file_path, new_file_path)  # Replace if file already exists
        new_paths.append(new_file_path)
        print(f"Saved: {new_file_path}")
    
    return new_paths


def process_audio(file_path):
    print(f"Processing started for {file_path}...")
    
    # Step 1: Convert WebM to WAV
    wav_file_path = "input_converted.wav"
    wav_file_path = convert_webm_to_wav(file_path, wav_file_path)

    # Step 2: Slow down the audio by 20%
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    processed_dir = f"{base_name}_processed"

    slowed_audio_path = slow_down_audio(wav_file_path, processed_dir, slowdown_factor=1.2)

    # Step 3: Simulate a reference signal (replace with actual speaker output in real scenarios)
    mic_signal, rate = get_audio_data(slowed_audio_path)
    ref_signal = np.copy(mic_signal) * 0.5  # Example reference signal for testing

    # Step 4: Perform Echo Cancellation and save results
    echo_canceled_path = echo_cancellation(slowed_audio_path, ref_signal, processed_dir)

    # Step 5: Load Echo-Canceled Audio for Bandpass Filtering
    echo_canceled_signal, rate = get_audio_data(echo_canceled_path)

    # Step 6: Apply Bandpass Filtering to Echo-Canceled Audio
    bandpass_filtered_path = bandpass_filter(echo_canceled_signal, rate)

    # Step 7: Save all processed files into the processed directory
    final_paths = save_to_processed_directory(
        file_path,
        slowed_audio_path,
        echo_canceled_path,
        'bandpass_filtered.wav'  # Only include existing files
    )
    
    print("Processing complete.")
    
    return final_paths


if __name__ == '__main__':
    file_path = '305_long.webm'
    
    try:
        processed_files = process_audio(file_path)
        print("Processed files saved at:")
        for path in processed_files:
            print(f" - {path}")
    
    except RuntimeError as e:
        print(f"Processing failed: {e}")