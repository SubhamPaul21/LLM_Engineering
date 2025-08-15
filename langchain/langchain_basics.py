import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class Joke(BaseModel):
    category: str = Field(description="The category of the joke")
    joke: str = Field(description="The actual joke text")


def main():
    if not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = input("Enter API key for Groq: ")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Return a JSON object that matches the required schema for a Joke. "
                "Write a short joke for the given topic.",
            ),
            ("human", "Topic: {topic}"),
        ]
    )

    # If the model supports structured outputs, this returns a Runnable that emits Joke objects
    structured_llm = llm.with_structured_output(Joke)

    # Combine prompt with structured llm
    chain = prompt | structured_llm

    result: Joke = chain.invoke({"topic": "Dreams"})
    # Access fields
    print(result.category)
    print(result.joke)


if __name__ == "__main__":
    main()

# ----- Create audio with the generated LLM output and save it as a .wav file -----

# from groq import Groq
# from pathlib import Path
# from kokoro import KPipeline
# import soundfile as sf
# from huggingface_hub import login
# import numpy as np

# if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
#     token = input("Enter API key for Hugging Face Hub: ")
#     login(token)
# else:
#     login(os.getenv("HUGGINGFACE_HUB_TOKEN"))

# client = Groq()
# # response = client.audio.speech.create(
# #     model="playai-tts",
# #     voice="Aaliyah-PlayAI",
# #     response_format="wav",
# #     input=output,
# # )
# # response.write_to_file(speech_file_path)

# # KOKORO TTS

# # Collect all audio chunks
# audio_segments = []

# pipeline = KPipeline(lang_code='a')
# generator = pipeline(output, voice='am_adam')
# for i, (gs, ps, audio) in enumerate(generator):
#     print(i, gs, ps)
#     audio_segments.append(audio)

# # Concatenate all audio into one
# full_audio = np.concatenate(audio_segments)

# # Write to a single file
# sf.write('output.wav', full_audio, 24000)
