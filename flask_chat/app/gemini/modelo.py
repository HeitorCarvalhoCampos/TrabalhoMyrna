import google.generativeai as genai
from dotenv import load_dotenv
import os

# Carrega a variável de ambiente do arquivo .env (se você estiver usando .env)
load_dotenv()

# Configura a API key
chave_api = os.getenv("GEMINI_API_KEY")

genai.configure(api_key = chave_api)

# Inicia o modelo
model = genai.GenerativeModel("gemini-2.0-flash")

# Cria uma sessão de chat 
chat = model.start_chat() 

# Prompt de sistema
system_prompt = "Você é um assistente técnico em IA que responde de forma objetiva, clara e sem rodeios. Sempre utilize exemplos em Python."

# rotina para enviar pergunta ao modelo
def responder_pergunta(pergunta: str) -> str:
    try:
        # Envia o prompt de sistema como a primeira mensagem para definir o comportamento 
        chat.send_message(system_prompt) 

        # Envia a pergunta para o modelo e recebe a resposta
        resposta = model.generate_content(pergunta)
        return resposta.text.strip()
    except Exception as e:
        return f"Erro ao consumir API: {str(e)}"
