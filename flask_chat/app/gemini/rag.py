import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Carrega a variável de ambiente do arquivo .env (se você estiver usando .env)
load_dotenv()
chave_api = os.getenv("GEMINI_API_KEY") # Configura a API key
if not chave_api:
    raise ValueError("GEMINI_API_KEY não configurada.")
api_key = chave_api

# Caminho para arquivos de texto
PASTA_DOCS = "app/texts"

# Carrega documentos
def carregar_documentos(pasta):
    documentos = []
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.endswith(".txt"):
            caminho = os.path.join(pasta, nome_arquivo)
            loader = TextLoader(caminho, encoding="utf-8")
            documentos.extend(loader.load())
    return documentos

# Carregamento e split
documentos = carregar_documentos(PASTA_DOCS)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documentos_divididos = splitter.split_documents(documentos)

# Embeddings + Vetor
embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=api_key,
    model="models/embedding-001"
)
db = FAISS.from_documents(documentos_divididos, embedding_model)

# Modelo generativo + RAG
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=api_key
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

def responder_com_rag(pergunta: str):
    try:
        resultado = rag_chain(pergunta)
        resposta = resultado['result']
        fontes = [doc.metadata['source'] for doc in resultado['source_documents']]
        return resposta.strip(), fontes
    except Exception as e:
        return f"Erro ao gerar resposta com RAG: {str(e)}", []
