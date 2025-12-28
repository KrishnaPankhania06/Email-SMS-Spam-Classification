import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


# ---------------- Page Config ----------------
st.markdown("""
<style>
header, footer {visibility: hidden;}

/* IMPORTANT: Use .stApp instead of body */
.stApp {
    background:
        radial-gradient(900px circle at 15% 10%, rgba(99,102,241,0.20), transparent 40%),
        radial-gradient(700px circle at 85% 20%, rgba(34,211,238,0.18), transparent 40%),
        radial-gradient(700px circle at 50% 85%, rgba(168,85,247,0.18), transparent 40%),
        linear-gradient(180deg, #020617, #020617);
}

/* Main card */
.card {
    background: rgba(2, 6, 23, 0.90);
    backdrop-filter: blur(18px);
    padding: 2.8rem;
    border-radius: 18px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.7);
    border: 1px solid rgba(255,255,255,0.08);
}

/* Title */
.title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2.2rem;
    font-size: 1.05rem;
}

/* Button */
button[kind="primary"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 14px;
    height: 3.2rem;
    font-size: 1.05rem;
    font-weight: 700;
}
/* RESULT CONTAINER */
.result-box {
    margin-top: 1.8rem;
    padding: 1.4rem;
    border-radius: 14px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.5px;
}

/* SPAM RESULT */
.result-spam {
    background: linear-gradient(90deg, #7f1d1d, #991b1b, #7f1d1d);
    color: #fecaca;
    box-shadow: 0 0 20px rgba(239,68,68,0.6);
    animation: spamPulse 1.2s infinite alternate;
}

/* SAFE RESULT */
.result-ham {
    background: linear-gradient(90deg, #064e3b, #065f46, #064e3b);
    color: #bbf7d0;
    box-shadow: 0 0 20px rgba(34,197,94,0.5);
}

@keyframes spamPulse {
    from {
        box-shadow: 0 0 12px rgba(239,68,68,0.6);
        transform: scale(1);
    }
    to {
        box-shadow: 0 0 26px rgba(239,68,68,0.9);
        transform: scale(1.02);
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------- NLP ----------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# ---------------- Load Model ----------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ---------------- UI ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='title'>🤖 Spam Detection AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Real-time Email & SMS Spam Classification using Machine Learning</div>",
    unsafe_allow_html=True
)

message = st.text_area(
    "📩 Enter Message",
    height=130,
    placeholder="Paste an email or SMS message here..."
)

if st.button("🚀 Analyze Message", use_container_width=True):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed = transform_text(message)
        vector = tfidf.transform([processed])
        result = model.predict(vector)[0]

        st.markdown("---")

        if result == 1:
            st.markdown("""
                <div class="result-box result-spam">
                    🚨 SPAM MESSAGE DETECTED
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="result-box result-ham">
                    ✅ THIS MESSAGE IS SAFE
                </div>
            """, unsafe_allow_html=True)

