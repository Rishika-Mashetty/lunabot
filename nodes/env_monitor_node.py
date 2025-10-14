import time, random, os
from gtts import gTTS
import playsound

def speak(text):
    print(f"🔊 {text}")
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    playsound.playsound("speech.mp3", True)
    os.remove("speech.mp3")

while True:
    temp = random.randint(18, 35)
    o2 = random.uniform(19, 23)

    if temp > 30:
        speak(f"Warning! Temperature too high: {temp} degrees Celsius.")
    if o2 < 20:
        speak(f"Warning! Oxygen level too low: {o2:.2f} percent.")

    time.sleep(5)
