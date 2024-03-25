from groq import Groq
import os
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["GROQ_API_KEY"] = "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"

# initialize
client = Groq()
embeddings = JinaEmbeddings(
    jina_api_key="jina_2f109bd96dcd4988a9fcc2962302bb2dqi8vqyi2pWxb7qUSI4EDSU0hAFnE",
    model_name="jina-embeddings-v2-base-en"
)

async def rephrase_input(input):
    """Rephrase the input using Groq."""
    client = Groq()
    response = await client.chat.completion.acreate(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a rephraser and always respond with a rephrased version of the input that is given to a search engine API. Always be succint and use the same words as the input. ONLY RETURN THE REPHRASED VERSION OF THE INPUT."},
            {"role": "user", "content": f"{input}"}
        ]
    )
    return response.choices[0].message.content


async def search_engine_from_source(message):
    """Search engine from source."""
    num_of_pages_to_scan = 1
    loader = ""
    rephrased_message = await rephrase_input(message)
    # recall documents
    docs = await loader.call(rephrased_message, count=num_of_pages_to_scan)
    # normalize data
    nomalized_data = nomalize_data(docs)
    # process and vectorize the content
    return await map(fetch_and_process, nomalized_data)

async def nomalize_data(docs):
    """Nomalize data."""
    
    return docs

async def fetch_and_process(item):
    """process and vectorize the content."""
    raw_content = item.name
    vector_store = await FAISS.from_texts(
        item.name,
        embeddings
    )
    return vector_store.similarity_search(
        message, numberOfSimilarityResults
    )

# fetch and process sources
sources = await search_engine_from_source(message)
source_parsed = map(lambda doc: { doc.name, doc.url }, sources)

# prepare the response content
chat_completion = await client.chat.completion.acreate(
    model="mixtral-8x7b-32768",
    messages=[
        {"role": "system",
         "content"=f"""
            - Here is my query '{message}', respond back with an answer that is as long as possible. If you can't find any results, respond with '没有找到相关结果，您可以尝试其他关键词。'
            - {embedSourcesInLLMResponse ? 'Return the sources used in the response with iterable numbered markdown style annotations.' : ''}
         """},
        {"role": "user", "content": f""" - Here are the top results from a similarity search: {JSON.stringify(sources)}. """},
    ], stream=True
)

def main(input):
    """Main function."""
    
    rephrase_input(input)

if __name__ == "__main__":
    main()