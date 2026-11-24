import re


def chunk_document(pages, chunk_size=500, overlap=50):

    chunks = []
    metadata = []

    for page in pages:

        text = page["text"]
        page_number = page["page_number"]

        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Handle unusually long sentences
            if len(sentence) > chunk_size:

                if current_chunk:
                    chunks.append(current_chunk)
                    metadata.append({
                        "page_number": page_number
                    })
                    current_chunk = ""

                for start in range(0, len(sentence), chunk_size - overlap):
                    piece = sentence[start:start + chunk_size]

                    if piece.strip():
                        chunks.append(piece)
                        metadata.append({
                            "page_number": page_number
                        })

                continue

            # Add sentence if it fits
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:

                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

            else:

                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    metadata.append({
                        "page_number": page_number
                    })

                # Retain the previous sentence as overlap
                words = current_chunk.split()

                overlap_text = ""

                for word in reversed(words):
                    candidate = (word + " " + overlap_text).strip()

                    if len(candidate) > overlap:
                        break

                    overlap_text = candidate

                current_chunk = (overlap_text + " " + sentence).strip()

        # Save remaining text
        if current_chunk:
            chunks.append(current_chunk)
            metadata.append({
                "page_number": page_number
            })

    return chunks, metadata