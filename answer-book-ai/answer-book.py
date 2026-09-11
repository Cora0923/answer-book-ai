import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import random
from textblob import TextBlob

# ---------------------- 答案库 中英双语 ----------------------
answer_bank = {
    "cn": {
        "anxious": [
            "时机尚未成熟",
            "保持耐心，静待变化",
            "不必过度思虑",
            "不妨暂且搁置",
            "顺其自然就好"
        ],
        "hesitate": [
            "相信你的直觉",
            "不要急于下定论",
            "答案藏在行动里",
            "此事大可一试",
            "前路自有转机"
        ],
        "hope": [
            "坚持便会如愿",
            "结局会超出预期",
            "大胆去尝试吧",
            "听从内心的声音",
            "好事正在靠近"
        ],
        "neutral": [
            "这件事值得等待",
            "最好另寻方向",
            "这件事可以放手",
            "此刻不宜行动"
        ]
    },
    "en": {
        "anxious": [
            "The time is not yet ripe",
            "Be patient and wait for change",
            "Do not overthink",
            "Leave it for now",
            "Let nature take its course"
        ],
        "hesitate": [
            "Trust your instinct",
            "Do not rush to decide",
            "The answer lies in action",
            "Go ahead without hesitation",
            "A new chance will come"
        ],
        "hope": [
            "Persist and your wish will come true",
            "The result will surprise you",
            "Take the bold step",
            "Follow your inner voice",
            "Good things are coming"
        ],
        "neutral": [
            "It is worth waiting",
            "Look for another direction",
            "You may let it go",
            "Now is not the time to act"
        ]
    }
}

# 页面配置
st.set_page_config(page_title="Answer Book AI", page_icon="📖", layout="centered")

# 初始化session state
if "history" not in st.session_state:
    st.session_state.history = []
if "lang" not in st.session_state:
    st.session_state.lang = "cn"

# ---------------------- 加载语义模型 ----------------------
@st.cache_resource
def load_model():
    model_name = "uer/sbert-base-chinese-nli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

# 预加载全部答案
def get_all_answers(lang):
    all_ans = []
    for cat in answer_bank[lang].values():
        all_ans.extend(cat)
    return all_ans

# ---------------------- 情绪检测 ----------------------
def detect_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score < -0.1:
        return "anxious"
    elif score > 0.1:
        return "hope"
    else:
        hesitate_words = ["要不要", "该不该", "犹豫", "选哪个", "纠结"]
        if any(w in text for w in hesitate_words):
            return "hesitate"
        return "neutral"

# ---------------------- 获取答案函数 ----------------------
def get_book_answer(question, randomness):
    sensitive_words = ["看病", "诊断", "投资", "股票", "法律", "判刑", "disease", "invest", "law", "sue"]
    if any(w in question.lower() for w in sensitive_words):
        return "⚠️ This is for entertainment only. Please consult professionals for medical, financial or legal matters."

    lang = st.session_state.lang
    all_answers = get_all_answers(lang)
    answer_embeds = np.vstack([get_embedding(sent) for sent in all_answers])
    q_embed = get_embedding(question)
    sims = np.dot(answer_embeds, q_embed.T).flatten()

    top_k = int(randomness * 4) + 1
    top_k = min(top_k, len(all_answers))
    top_idx = np.argsort(sims)[-top_k:]
    candidates = [all_answers[i] for i in top_idx]
    return random.choice(candidates)

# ---------------------- UI界面 ----------------------
st.title("📖 Answer‑Book‑AI")
st.markdown("*A semantic‑powered Answer Book agent, for entertainment only*")

# 侧边栏
with st.sidebar:
    st.subheader("Settings")
    st.session_state.lang = st.radio("Language", options=["cn", "en"], format_func=lambda x: "中文" if x=="cn" else "English")
    randomness = st.slider("Randomness", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                           help="0 = most relevant, 1 = fully random")
    st.divider()
    if st.button("Clear History"):
        st.session_state.history = []

# 输入框
user_q = st.text_area("Write your question here:", height=120)
if st.button("Get Answer"):
    if len(user_q.strip()) > 2:
        ans = get_book_answer(user_q, randomness)
        st.session_state.history.append({"question": user_q, "answer": ans})
        st.success(f"✨ {ans}")
    else:
        st.warning("Please enter a real question.")

# 历史记录区域
st.divider()
st.subheader("📜 Q&A History")
if st.session_state.history:
    history_text = ""
    for idx, item in enumerate(st.session_state.history, 1):
        history_text += f"Q: {item['question']}\nA: {item['answer']}\n---\n"
        st.markdown(f"**{idx}.** Q: {item['question']}\n>A: {item['answer']}")
    st.download_button(label="Download History (.txt)", data=history_text, file_name="answer_book_history.txt")
else:
    st.info("No history yet. Ask your first question!")