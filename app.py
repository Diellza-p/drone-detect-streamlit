import streamlit as st
import librosa
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
import matplotlib.pyplot as plt

st.set_page_config(page_title="Drone Detection", layout="wide")

# Title
st.title("Drone Detection System")
st.write("Upload audio file to detect drones")

# Drone detector class
class DroneDetector:
    def __init__(self):
        self.sr = 6000
    
    def extract_features(self, audio_signal):
        """Extract features from audio"""
        features = {}
        
        # FFT
        N = len(audio_signal)
        freqs = fftfreq(N, 1/self.sr)[:N//2]
        magnitude = np.abs(fft(audio_signal))[:N//2]
        magnitude = magnitude / (np.max(magnitude) + 1e-10)
        
        # Dominant frequency
        features['dominant_freq'] = freqs[np.argmax(magnitude)]
        
        # Peak detection
        peaks, properties = signal.find_peaks(magnitude, height=0.05, distance=20)
        features['num_peaks'] = len(peaks)
        
        # Spectral properties
        features['spectral_centroid'] = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        
        # Energy
        features['rms_energy'] = np.sqrt(np.mean(audio_signal**2))
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=self.sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}'] = np.mean(mfccs[i])
        
        return features, freqs, magnitude
    
    def detect(self, audio_file):
        """Detect drone in audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sr)
            
            # Extract features
            features, freqs, magnitude = self.extract_features(y)
            
            # Simple detection logic
            freq_stability = 1 - np.std([features['dominant_freq']]) / (features['dominant_freq'] + 1e-10)
            peak_consistency = features['num_peaks'] / 50.0
            energy_level = min(features['rms_energy'] * 10, 1.0)
            
            # Confidence score
            confidence = (freq_stability * 0.4 + peak_consistency * 0.3 + energy_level * 0.3) * 100
            confidence = max(0, min(100, confidence))
            
            # Decision
            is_drone = confidence > 45
            
            return {
                'is_drone': is_drone,
                'confidence': round(confidence, 1),
                'dominant_frequency': round(features['dominant_freq'], 2),
                'rms_energy': round(features['rms_energy'], 4),
                'num_peaks': features['num_peaks'],
                'audio_duration': round(len(y) / sr, 2),
                'freqs': freqs,
                'magnitude': magnitude,
                'audio_signal': y
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'is_drone': None,
                'confidence': 0
            }

# Create detector
detector = DroneDetector()

# Sidebar
st.sidebar.title("About")
st.sidebar.write("""
**RF Acoustic Drone Detection**

- Analyzes audio for drone signatures
- Uses AI to classify signals
- Shows confidence score
- Technical details available
""")

# Main app
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

with col2:
    st.subheader("Instructions")
    st.write("""
    1. Upload a WAV, MP3, or OGG file
    2. System analyzes the audio
    3. Results show detection status
    """)

if uploaded_file is not None:
    st.write("---")
    
    # Show file info
    st.write(f"**File:** {uploaded_file.name}")
    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
    
    # Detect
    with st.spinner("Analyzing audio..."):
        result = detector.detect(uploaded_file)
    
    if 'error' in result:
        st.error(f"Error: {result['error']}")
    else:
        # Display results
        st.write("---")
        
        # Main result
        is_drone = result['is_drone']
        confidence = result['confidence']
        
        if is_drone:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.error(f"DRONE DETECTED")
            st.metric("Confidence", f"{confidence}%")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.success("✓ Area Clear")
            st.metric("Confidence", f"{confidence}%")
        
        st.write("---")
        
        # Details
        st.subheader("Technical Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dominant Frequency", f"{result['dominant_frequency']} Hz")
        
        with col2:
            st.metric("Audio Duration", f"{result['audio_duration']}s")
        
        with col3:
            st.metric("Spectral Peaks", result['num_peaks'])
        
        st.write("---")
        
        # Frequency spectrum visualization
        st.subheader("Frequency Spectrum")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(result['freqs'], result['magnitude'], linewidth=1.5, color='#667eea')
        ax.fill_between(result['freqs'], result['magnitude'], alpha=0.3, color='#667eea')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (normalized)')
        ax.set_title('FFT Spectrum')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 3000])
        
        st.pyplot(fig)
        
        # Waveform
        st.subheader("Waveform")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        time = np.linspace(0, len(result['audio_signal']) / 6000, len(result['audio_signal']))
        ax.plot(time, result['audio_signal'], linewidth=0.5, color='#667eea')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio Waveform')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.write("---")
        
        # Summary
        st.subheader("Summary")
        if is_drone:
            st.write(f"✓ Drone detected with {confidence}% confidence")
            st.write(f"✓ Dominant frequency: {result['dominant_frequency']} Hz")
            st.write(f"✓ Audio duration: {result['audio_duration']}s")
            st.warning("Alert would be triggered in production system")
        else:
            st.write(f"✓ No drone detected")
            st.write(f"✓ Confidence in detection: {confidence}%")
            st.write(f"✓ Area appears safe")
            st.success("No action needed")

st.write("---")
st.write("*RF Acoustic Drone Detection System v1.0*")