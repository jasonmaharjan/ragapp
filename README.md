# ragapp
RAG application with the intention to analyze and parse local documents.


### Chunking Strategy
- Split text into overlapping chunks that adhere to paragraph boundaries, where paragraphs are never split mid-sentence. 
- When the processed text exceeds the max_chars, the current buffer is saved as a chunk and the next chunk starts with the last overlap_chars characters of the previous one, potentially preserving cross-chunk context.