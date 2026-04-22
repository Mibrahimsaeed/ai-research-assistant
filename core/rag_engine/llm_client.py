# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# client = OpenAI()


# def ask_llm(prompt, temperature=0.3, model="gpt-4o-mini"):
#     """
#     Central LLM interface for entire project
#     """

#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a helpful AI research assistant."
#             },
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         temperature=temperature
#     )

#     return response.choices[0].message.content



import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

model = genai.GenerativeModel("gemma-3-1b-it")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(query, context):

    prompt = f"""
You are a research assistant.

Answer the question using ONLY the provided context.

RULES:
- Cite sources using [paper_id]
- Combine multiple sources
- Highlight agreements or disagreements
- If unsure, say "Not enough information"
- Provide a confidence level (Low/Medium/High)
- Do NOT hallucinate
QUESTION:
{query}

CONTEXT:
{context}

ANSWER:
"""

    response = model.generate_content(prompt)

    return response.text