import os
from typing import Optional
import ollama

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", None)


def gerar_relatorio_agronomico(
    planta: str,
    status: str,
    folhas_doentes: int,
    semana: Optional[int] = None,
    fase_esperada: Optional[str] = None,
    orientacao_adicional: Optional[str] = None,
) -> str:
    """
    Envia o diagnóstico estruturado para o Ollama local e retorna o relatório explicativo.
    Em estrita conformidade com as Seções 5 e 7 do Plano de Projeto (ABNT).
    """
    semana_txt = f"Semana atual do ciclo: {semana}." if semana is not None else ""
    fase_txt = f"Fase fenológica esperada: '{fase_esperada}'." if fase_esperada else ""
    adicional_txt = f"Observação do classificador: {orientacao_adicional}" if orientacao_adicional else ""

    prompt = f"""
Você é o assistente virtual agronômico do sistema AgroSaúde, desenvolvido para apoiar pequenos produtores no manejo sustentável.
O usuário enviou 3 fotografias de folhas de uma cultura de {planta}.
Diagnóstico da Visão Computacional:
- Status geral: '{status}' (saudável, doença ou incerto).
- Folhas com indício de patógeno: {folhas_doentes} de 3.
{semana_txt}
{fase_txt}
{adicional_txt}

Instruções para o relatório:
1. Redija um texto conciso, empático e em linguagem simples e acessível.
2. Explique o que o resultado significa para o produtor e a importância de monitorar semanalmente.
3. Se o status for 'incerto', recomende o reenvio de fotografias com boa iluminação, fundo neutro e uma folha por foto.
4. Sugira exclusivamente práticas culturais e preventivas de manejo (poda sanitária, rotação, espaçamento, irrigação na base evitando molhar as folhas, drenagem e isolamento).
5. REGRA ÉTICA MANDATÓRIA (Seção 7 ABNT): Você NUNCA deve prescrever produtos químicos, agrotóxicos, pesticidas industriais ou dosagens. Recomende sempre a avaliação presencial de um engenheiro agrônomo ou técnico de extensão rural.
""".strip()

    try:
        client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama
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
