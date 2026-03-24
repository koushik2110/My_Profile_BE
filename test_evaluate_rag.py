import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from app.rag import _get_qa_chain
from langchain_groq import ChatGroq

# HuggingFaceEmbeddings exposes model_name as a plain string, which ragas 0.4.x
# requires for EmbeddingUsageEvent. FastEmbedEmbeddings exposes the raw model
# object instead, causing a Pydantic ValidationError inside AnswerRelevancy.
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import GROQ_API_KEY

def test_run_evaluation():
    print("1. Initializing RAG pipeline...")
    chain = _get_qa_chain()
    
    # --- YOUR TEST DATASET ---
    # Replace these questions and ground_truths with real data from your documents
    questions = [
        "Which two projects are discussed in the documents and what is the primary goal of the GenAI-Based OCI Log Analysis system?",
    ]
    ground_truths = [
        "The two projects discussed are the GenAI-Based OCI (Oracle Cloud Infrastructure) Log Analysis and Recommendation System and Process Mate. The primary goal of the OCI Log Analysis system is to use LLMs for automated log analysis and to provide actionable recommendations.", 
    ]
    
    answers = []
    contexts = []
    
    print("2. Fetching answers and chunks from PGVector...")
    for q in questions:
        # Run question through the RAG pipeline
        response = chain({"query": q})
        answers.append(response["result"])
        print(f"\n--- ANSWER ---\n{response['result']}")
        
        # Extract the chunks retrieved from pgvector
        docs = response["source_documents"]
        page_chunks = [doc.page_content for doc in docs]
        contexts.append(page_chunks)
        print("\n--- RETRIEVED CHUNKS ---")
        for i, chunk in enumerate(page_chunks):
            print(f"[{i}] {chunk[:200]}...")
        
    # Format the data exactly as Ragas expects it (using HuggingFace datasets library)
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("3. Setting up Groq as the Ragas Evaluator...")
    evaluator_llm = LangchainLLMWrapper(ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY))
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5"))

    # In ragas 0.4.x, metrics must be instantiated as objects with LLM/embeddings
    # bound explicitly. The module-level singletons don't carry embeddings, causing
    # answer_relevancy (which needs embeddings for cosine similarity) to return NaN.
    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1),
    ]

    print("4. Scoring Context Precision, Recall, Faithfulness & Answer Relevancy...")
    result = evaluate(
        dataset,
        metrics=metrics,
    )
    
    print("\n==============================")
    print("   RAGAS EVALUATION RESULTS   ")
    print("==============================")
    print(result)

    # assert result["context_precision"] > 0.5
    # assert result["context_recall"] > 0.5

if __name__ == "__main__":
    test_run_evaluation()
