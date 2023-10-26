import speech_recognition as sr
import pyttsx3

recog = sr.Recognizer()

def speech_to_text(sentence):
    engine = pyttsx3.init()
    engine.say(sentence)
    engine.runAndWait()

with sr.Microphone() as source:
    recog.adjust_for_ambient_noise(source, duration = 0.2)
    audio = recog.listen(source)
    text_from_speech = recog.recognize_google(audio).lower()
    print("You said: " + text_from_speech)