import os
from typing import Optional
import ollama

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", None)


def obter_cliente_ollama():
    """Retorna a instância configurada do cliente Ollama."""
    return ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama


def listar_modelos_ollama() -> list:
    """Consulta os modelos instalados no serviço Ollama."""
    try:
        client = obter_cliente_ollama()
        resp = client.list()
        if hasattr(resp, "models"):
            return [m.model for m in resp.models]
        if isinstance(resp, dict) and "models" in resp:
            return [m.get("name", m.get("model")) for m in resp["models"]]
        return []
    except Exception as e:
        print(f"[ERRO OLLAMA] Falha ao listar modelos: {e}")
        return []


def gerar_relatorio_agronomico(
    planta: str,
    status: str,
    folhas_doentes: int,
    semana: Optional[int] = None,
    fase_esperada: Optional[str] = None,
    orientacao_adicional: Optional[str] = None,
    doencas: Optional[list] = None,
) -> str:
    """
    Envia o diagnóstico estruturado para o Ollama local e retorna o relatório explicativo.
    Em estrita conformidade com as Seções 5 e 7 do Plano de Projeto (ABNT).
    """
    semana_txt = f"Semana atual do ciclo: {semana}." if semana is not None else ""
    fase_txt = f"Fase fenológica esperada: '{fase_esperada}'." if fase_esperada else ""
    adicional_txt = f"Observação do classificador: {orientacao_adicional}" if orientacao_adicional else ""
    doencas_txt = f"Doenças identificadas pelo modelo: {', '.join(doencas)}." if doencas else ""

    prompt = f"""
Você é o assistente virtual agronômico do sistema AgroSaúde, desenvolvido para apoiar pequenos produtores no manejo sustentável.
O usuário enviou 3 fotografias de folhas de uma cultura de {planta}.
Diagnóstico da Visão Computacional:
- Status geral: '{status}' (saudável, doença ou incerto).
- Folhas com indício de patógeno: {folhas_doentes} de 3.
{semana_txt}
{fase_txt}
{adicional_txt}
{doencas_txt}

Instruções para o relatório:
1. Redija um texto conciso, empático e em linguagem simples e acessível.
2. Explique o que o resultado significa para o produtor e a importância de monitorar semanalmente.
3. Se o status for 'incerto', recomende o reenvio de fotografias com boa iluminação, fundo neutro e uma folha por foto.
4. Sugira exclusivamente práticas culturais e preventivas de manejo (poda sanitária, rotação, espaçamento, irrigação na base evitando molhar as folhas, drenagem e isolamento).
5. REGRA ÉTICA MANDATÓRIA: Você NUNCA deve prescrever ou citar produtos químicos, agrotóxicos, fungicidas, inseticidas, pesticidas industriais ou dosagens. Recomende sempre a avaliação presencial de um engenheiro agrônomo ou técnico de extensão rural.
6. Escreva em português do Brasil, em até 3 parágrafos curtos. Não use placeholders como [nome do produtor] nem assinatura; comece direto pelo conteúdo.
""".strip()

    try:
        client = obter_cliente_ollama()
        resposta = client.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return resposta['message']['content']

    except Exception as e:
        # Texto de contingência exigido no plano de projeto (Seção 7(d) ABNT) caso a IA falhe
        print(f"[ERRO LLM] Falha ao comunicar com Ollama ({OLLAMA_MODEL}): {e}")
        if status == "saudavel":
            return (
                f"Sua cultura de {planta} apresentou aspecto saudável nas 3 fotografias analisadas. "
                "Continue mantendo a irrigação adequada na base da planta sem encharcar o solo e continue o monitoramento na próxima semana. "
                "Nota: Esta é uma ferramenta de apoio orientativo e não substitui uma consultoria agronômica presencial."
            )
        elif status == "incerto":
            return (
                "A confiança na classificação de uma ou mais fotos foi inconclusiva. "
                "Para garantir sua segurança, solicitamos o envio de novas fotografias: "
                "fotografe uma folha inteira por vez, em ambiente bem iluminado e com fundo liso/neutro."
            )
        else:
            return (
                f"O sistema detectou indícios de patógenos em {folhas_doentes} de 3 folhas da sua cultura de {planta}. "
                "Recomendamos isolar ou descartar adequadamente as folhas com manchas visíveis, evitar molhar a folhagem durante a rega, "
                "higienizar as ferramentas de poda e consultar um agrônomo ou técnico de extensão rural da sua região para uma avaliação presencial."
            )


def responder_chat_agronomico(
    planta_nome: str,
    mensagem_usuario: str,
    historico_mensagens: Optional[list] = None,
    modelo: Optional[str] = None,
) -> str:
    """
    Conversa interativa com o assistente agronômico usando o Ollama com contexto do plantio.
    """
    system_prompt = f"""
Você é o assistente virtual agronômico inteligente do AgroSaúde.
O usuário está cultivando {planta_nome}.
Seu objetivo é orientar o produtor sobre boas práticas de manejo, irrigação, espaçamento, podas sanitárias e prevenção fitossanitária.
REGRAS ÉTICAS:
1. NUNCA prescreva produtos químicos sintéticos, agrotóxicos ou dosagens de venenos industriais.
2. Sempre recomende a consulta presencial a um engenheiro agrônomo ou técnico de extensão rural para diagnósticos definitivos e receituário agronômico.
3. Seja claro, prestativo e acolhedor em português brasileiro.
""".strip()

    messages = [{"role": "system", "content": system_prompt}]

    if historico_mensagens:
        # Pega as últimas 8 mensagens para manter o contexto sem estourar limite
        for msg in historico_mensagens[-8:]:
            papel = "user" if msg.get("papel") == "usuario" else "assistant"
            messages.append({"role": papel, "content": msg.get("texto", "")})

    messages.append({"role": "user", "content": mensagem_usuario})

    try:
        client = obter_cliente_ollama()
        resposta = client.chat(
            model=modelo or OLLAMA_MODEL,
            messages=messages,
        )
        return resposta["message"]["content"]
    except Exception as e:
        print(f"[ERRO CHAT OLLAMA] {e}")
        return (
            f"Recebi sua dúvida sobre o plantio de {planta_nome}. "
            "No momento o serviço do assistente virtual está temporariamente indisponível para resposta dinâmica. "
            "Recomendamos manter o monitoramento semanal e procurar orientação presencial de um agrônomo."
        )


def chat_generico_ollama(
    mensagens: list,
    modelo: Optional[str] = None,
) -> str:
    """Endpoint direto para chat livre com qualquer modelo do Ollama."""
    client = obter_cliente_ollama()
    resposta = client.chat(
        model=modelo or OLLAMA_MODEL,
        messages=mensagens,
    )
    return resposta["message"]["content"]
