import os
import json
import time
import logging
import re
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import tenacity
from cachetools import TTLCache
from google import genai

# --- Configurações Gerais ---

# Logging estruturado
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Cache em memória: até 100 itens, expira em 10 minutos
cache = TTLCache(maxsize=100, ttl=600)

# Cliente Gemini
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBYZExfZcBiaEtQsmCK0cS_pqcRwfWV2Mw")
client = genai.Client(api_key=API_KEY)

# Executor para chamadas bloqueantes
executor = ThreadPoolExecutor(max_workers=4)

def run_in_executor(func, *args):
    """Executa função bloqueante num executor e retorna o resultado."""
    return executor.submit(func, *args).result()

# Decorator de retry para falhas transitórias
def retry_on_failure(func):
    @wraps(func)
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True
    )
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} executado em {elapsed:.2f}s, payload size={len(str(args)+str(kwargs))}")
        return result
    return wrapper

# ----- "Agentes" inline -----

@retry_on_failure
def fetch_subtopics(theme: str, count: int) -> list[str]:
    """Agent1: busca 'count' subtemas para 'theme'."""
    cache_key = f"subtopics:{theme}:{count}"
    if cache_key in cache:
        return cache[cache_key]

    prompt = f"Liste {count} subtemas relevantes sobre '{theme}', voltados para jovens de 18-30 anos."
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    text = resp.text or ""
    items = re.findall(r'^\s*(?:\d+[\.)]|[-•])\s*(.+)$', text, flags=re.MULTILINE)
    if not items:
        items = [line.strip() for line in text.splitlines() if len(line.strip()) > 5]
    cache[cache_key] = items[:count]
    return items[:count]

@retry_on_failure
def choose_subtopic(theme: str, subtopics: list[str]) -> dict:
    """Agent2: escolhe o melhor subtema e justifica."""
    cache_key = f"choice:{theme}:{tuple(subtopics)}"
    if cache_key in cache:
        return cache[cache_key]

    prompt = (
        f"Dentre estes subtemas: {subtopics}, escolha o mais relevante para um post no Instagram "
        "e explique por que."
    )
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = resp.text or ""
    try:
        doc = json.loads(raw)
        result = {"choice": doc["choice"], "reason": doc["reason"]}
    except Exception:
        m = re.search(r'Choice\s*:\s*(.+)', raw)
        choice = m.group(1).strip() if m else subtopics[0]
        result = {"choice": choice, "reason": raw}
    cache[cache_key] = result
    return result

@retry_on_failure
def generate_caption(subtopic: str, length: str, formality: str) -> dict:
    """Agent3: gera legenda com CTA, hashtags e emojis."""
    prompt = (
        f"Você é um Redator Criativo. Crie uma legenda para Instagram sobre '{subtopic}', "
        f"tamanho {length}, formalidade {formality}. Inclua um CTA claro, 2-4 hashtags e emojis."
    )
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = resp.text or ""
    try:
        return json.loads(raw)
    except Exception:
        return {"caption": raw}

@retry_on_failure
def generate_image_prompt(subtopic: str) -> str:
    """Agent5: sugere prompt de imagem."""
    prompt = (
        f"Você é Leonardo da Vinci moderno. Baseado no subtema '{subtopic}', "
        "retorne apenas um prompt detalhado para gerar uma imagem de post no Instagram."
    )
    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return resp.text or ""

# ----- Streamlit UI -----

st.set_page_config(
    page_title="Gerador de Legendas AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carrega CSS custom (se existir)
if os.path.exists("styles.css"):
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Estado inicial
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'caption_res' not in st.session_state:
    st.session_state['caption_res'] = {}

# Sidebar de configurações
if os.path.exists("logo.svg"):
    st.sidebar.image("logo.svg", use_column_width=True)
st.sidebar.title("Configurações")
palette = st.sidebar.selectbox("Paleta de cores", ["Azul", "Roxo", "Verde"])
num_subs = st.sidebar.slider("Número de subtemas", 3, 10, 5)
length = st.sidebar.radio("Tamanho da legenda", ["Curta", "Média", "Longa"], index=1)
formality = st.sidebar.radio("Formalidade", ["Baixa", "Média", "Alta"], index=1)

# Wizard em abas
tabs = st.tabs(["1. Tema", "2. Subtemas", "3. Escolha", "4. Legenda", "5. Imagem"])

# Etapa 1: Tema
with tabs[0]:
    theme_input = st.text_input("Digite o tema", value=st.session_state.get('theme', ''))
    if st.button("Próximo"):
        if not theme_input.strip():
            st.error("Informe um tema.")
        else:
            st.session_state['theme'] = theme_input.strip()

if 'theme' not in st.session_state:
    st.stop()

theme = st.session_state['theme']
progress = st.progress(0)

# Etapa 2: Subtemas
with tabs[1]:
    if 'subtopics' not in st.session_state:
        with st.spinner("Buscando subtemas..."):
            st.session_state['subtopics'] = run_in_executor(fetch_subtopics, theme, num_subs)
    subs = st.session_state['subtopics']
    for s in subs:
        st.checkbox(s, key=f"sub_{s}", value=True)
    progress.progress(20)

# Etapa 3: Escolha
with tabs[2]:
    if 'chosen' not in st.session_state:
        with st.spinner("Escolhendo subtema..."):
            st.session_state['chosen'] = run_in_executor(choose_subtopic, theme, subs)
    chosen = st.session_state['chosen']
    st.markdown(f"**Escolhido:** {chosen['choice']}\n\n**Por quê:** {chosen['reason']}")
    progress.progress(40)

# Etapa 4: Legenda
with tabs[3]:
    if st.button("Gerar legenda"):
        with st.spinner("Gerando legenda..."):
            st.session_state['caption_res'] = run_in_executor(
                generate_caption,
                chosen['choice'],
                length,
                formality
            )
    caption = st.session_state['caption_res'].get('caption', '')
    st.text_area("Legenda gerada", caption, height=150)
    if caption:
        st.download_button(
            "Baixar JSON da legenda",
            data=json.dumps(st.session_state['caption_res'], ensure_ascii=False, indent=2),
            file_name="caption.json"
        )
    progress.progress(60)

# Etapa 5: Imagem
with tabs[4]:
    if 'img_prompt' not in st.session_state and st.button("Sugerir imagem"):
        with st.spinner("Gerando prompt de imagem..."):
            st.session_state['img_prompt'] = run_in_executor(generate_image_prompt, chosen['choice'])
    if 'img_prompt' in st.session_state:
        prompt = st.session_state['img_prompt']
        st.code(prompt, language='text')
        st.download_button(
            "Baixar JSON do prompt",
            data=json.dumps({"prompt": prompt}, ensure_ascii=False, indent=2),
            file_name="image_prompt.json"
        )
    progress.progress(100)

# Histórico
if caption:
    entry = {"tema": theme, "subtema": chosen['choice'], "legenda": caption}
    if not any(h['tema']==entry['tema'] and h['subtema']==entry['subtema'] for h in st.session_state['history']):
        st.session_state['history'].append(entry)

st.sidebar.header("Histórico")
for i, h in enumerate(reversed(st.session_state['history'][-3:])):
    st.sidebar.write(f"{i+1}. {h['tema']} → {h['subtema']}")

# Feedback
st.sidebar.header("Feedback")
fdb = st.sidebar.text_area("Comentários")
if st.sidebar.button("Enviar feedback"):
    with open("feedback.csv", "a", encoding="utf-8") as f:
        f.write(f"{time.asctime()};{theme};{fdb}\n")
    st.sidebar.success("Obrigado pelo feedback!")

# Instruções de Uso
with st.expander("Como usar"):
    st.markdown(
        "1. Informe o tema  \n"
        "2. Revise os subtemas  \n"
        "3. Veja o subtema escolhido  \n"
        "4. Gere e copie a legenda  \n"
        "5. Sugira prompt de imagem"
    )

