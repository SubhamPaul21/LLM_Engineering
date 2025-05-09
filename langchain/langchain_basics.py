import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from groq import Groq
from kokoro import KPipeline
import soundfile as sf
from huggingface_hub import login
import numpy as np

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = input("Enter API key for Groq: ")

if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
    token = input("Enter API key for Hugging Face Hub: ")
    login(token)
else:
    login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

client = Groq()

llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.7)
user_prompt = ChatPromptTemplate.from_template("Write a 4 line funnypoem about {topic}. Just output the poem in plain english without formatting and nothing else. You can add expressions for the reader to dictate the poem better.")

chain = user_prompt | llm | StrOutputParser()
output = chain.invoke({"topic": "Dreams"})

# response = client.audio.speech.create(
#     model="playai-tts",
#     voice="Aaliyah-PlayAI",
#     response_format="wav",
#     input=output,
# )
# response.write_to_file(speech_file_path)

# KOKORO TTS

# Collect all audio chunks
audio_segments = []

pipeline = KPipeline(lang_code='a')
generator = pipeline(output, voice='am_adam')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    audio_segments.append(audio)

# Concatenate all audio into one
full_audio = np.concatenate(audio_segments)

# Write to a single file
sf.write('output.wav', full_audio, 24000)