from src.data_collection.stock_scraper import StockScraper
from src.data_collection.earnings_scraper import EarningsScraper
from src.data_processing.text_processor import TextProcessor
from src.data_processing.embedding_generator import EmbeddingGenerator
from src.models.language_model import LanguageModel
from src.utils.helpers import get_gpu_memory, recommend_model_config

def main():
    # Stock data collection
    stock_scraper = StockScraper()
    filtered_stocks = stock_scraper.get_filtered_stocks(["Enphase"])

    # Earnings call data collection
    earnings_scraper = EarningsScraper()
    urls = earnings_scraper.fetch_earnings_call_urls("ENPH", 2019, 2024)
    transcripts = earnings_scraper.fetch_transcripts(urls)
    earnings_scraper.close()

    # Text processing
    text_processor = TextProcessor()
    chunked_transcripts = []
    for (year, quarter), transcript in transcripts.items():
        sentences = text_processor.split_into_sentences(transcript)
        chunks = text_processor.chunk_sentences(sentences)
        for i, chunk in enumerate(chunks):
            chunked_transcripts.append({
                "year": year,
                "quarter": quarter,
                "chunk_num": i,
                "chunk_text": " ".join(chunk)
            })

    # Embedding generation
    embedding_generator = EmbeddingGenerator(device="cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embedding_generator.generate_embeddings([item["chunk_text"] for item in chunked_transcripts])

    # Language model setup
    gpu_memory = get_gpu_memory()
    model_id, use_quantization = recommend_model_config(gpu_memory)
    if model_id:
        language_model = LanguageModel(model_id, use_quantization)
    else:
        print("Not enough GPU memory to run a language model.")
        return

    # Example query
    query = "What were Enphase's gross margins for Q4 2023 and Q4 2024?"
    context_items = chunked_transcripts[:5]  # Just using the first 5 chunks as an example
    prompt = language_model.format_prompt(query, context_items)
    response = language_model.generate_response(prompt)

    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()