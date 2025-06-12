from app.gemini.rag import responder_com_rag

# Prompt de sistema
system_prompt = "Você é um assistente técnico em IA que responde de forma objetiva, clara e sem rodeios. Sempre utilize exemplos em Python."

# rotina para enviar pergunta ao modelo RAG
def responder_pergunta(pergunta: str) -> str:
    try:
        resposta, fontes = responder_com_rag(pergunta)
        if fontes:
            fontes_formatadas = "\n\nFontes:\n" + "\n".join(f"- {fonte}" for fonte in fontes)
            return resposta + fontes_formatadas
        return resposta
    except Exception as e:
        return f"Erro ao consumir API: {str(e)}"
