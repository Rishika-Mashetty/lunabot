import time, random
from gtts import gTTS
from IPython.display import Audio, display

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("speech.mp3")
    display(Audio("speech.mp3", autoplay=True))

while True:
    temp = random.randint(18, 35)
    o2 = random.uniform(19, 23)
    if temp > 30:
        speak(f"Warning! Temperature too high: {temp}C")
    if o2 < 20:
        speak(f"Warning! O2 level low: {o2}%")
    time.sleep(5)
