import pandas as pd  
import numpy as np 
import librosa
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder


# Load the label encoder from the pickle file
with open('E:\My_Projects\Emotion_Classifier\p3_model\label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

model = tf.keras.models.load_model('E:\My_Projects\Emotion_Classifier\p3_model\prototype3.h5')

class emotion_prediction:
    def load_and_trim_audio_clip(self, audio_path, target_duration=2.5):
        audio_clips = []
        # Load audio clip using librosa
        audio_clip, _ = librosa.load(audio_path, sr=None)
        
        # Trim audio clip to the target duration (2.5 seconds)
        target_length = int(target_duration * _)
        audio_clip = audio_clip[:target_length]
        
        audio_clips.append(audio_clip)
        return np.array(audio_clips)

    # Function to preprocess data
    def preprocess_data(self, audio_path):
        test_audio = self.load_and_trim_audio_clip(audio_path)
        
        return test_audio

    def compute_mfccs(self, audio_clips, sample_rate=22050, n_mfcc=13):
        mfccs_list = []
        for audio_clip in audio_clips:
            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=audio_clip, sr=sample_rate, n_mfcc=n_mfcc)
            mfccs_list.append(mfccs)
        test_mfccs = np.array(mfccs_list)
        test_mfccs /= np.max(np.abs(test_mfccs))
        return test_mfccs

    def make_prediction(self, audio_path):
        test_audio = self.preprocess_data(audio_path)
        test_mfccs = self.compute_mfccs(test_audio)
        predictions = model.predict(test_mfccs)

        # Convert the numerical predictions back to original class labels using the loaded label encoder
        predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        predicted_label_str = np.array2string(predicted_label, separator=', ', formatter={'str_kind': lambda x: x})

        # Remove unwanted characters from the string
        cleaned_label = predicted_label_str.strip("[]'")

        return cleaned_label.upper()

def main():
    emotion_predictor = emotion_prediction()
    #audio_path = r"E:\My_Projects\Emotion_Classifier\test\Happy_emotion.wav"
    audio_path = r"E:\My_Projects\Emotion_Classifier\p3_model\Emotion_Dataset(Splitted)\Test\disgust\03-01-07-01-02-02-15.wav"
    cleaned_label = emotion_predictor.make_prediction(audio_path)
    print(cleaned_label)


if __name__ == "__main__":
    main()
