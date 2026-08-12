def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    word = text.split()
    chunks = []
    # calculate stride which is basically chunk size - overlap
    stride = chunk_size - overlap
    for i  in range(0, len(word), stride):
        chunk = word[i:i + chunk_size]
        # by using .join, it combines all the word in chunk, with space between them, so its like a sentence rather than a list of words
        # and append just adds that sentence to the chunks list
        chunks.append(" ".join(chunk))
    return chunks

if __name__ == "__main__":
    sample_text = "This is a sample text that we will use to demonstrate how the chunking function works. It will split this text into smaller chunks based on the specified chunk size and overlap."
    chunked_text = chunk_text(sample_text, chunk_size=10, overlap=2)
    print(chunked_text)