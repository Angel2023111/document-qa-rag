import os

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-120b"


def generate_answer(query, retrieved_chunks):

    # Prepare context from retrieved chunks
    context = ""

    for result in retrieved_chunks:

        metadata = result["metadata"]

        source = metadata["source"]
        page_number = metadata["page_number"]

        context += f"""
    [Source: {source}, Page: {page_number}]

    {result['chunk']}

    """

    system_prompt = """
        You are a document question-answering assistant.

        Your task is to answer questions using ONLY
        the information provided in the context.

        Follow these rules:

        1. Do not use outside knowledge.
        2. Do not invent facts.
        3. If the answer is not present in the context,
        clearly say that you cannot find the answer
        in the provided document.
        4. Provide concise and accurate answers.
        5. Cite the source document name and PDF page number
        used to support your answer.
        6. If multiple pages support your answer,
        cite all relevant pages.
        7. Use this citation format:
        [Source: filename.pdf, Page: 5]
        8. Do not invent or guess source names or page numbers.
        """

    user_prompt = f"""
        Context:
        {context}

        Question:
        {query}

        Provide an answer with source citations.
        """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=700
    )

    return response.choices[0].message.content