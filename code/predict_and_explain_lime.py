import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from lime.lime_text import LimeTextExplainer

model = joblib.load("hinglish_fake_news_v2_model.joblib")

HINGLISH_FILLERS = set(ENGLISH_STOP_WORDS).union({
    "hai", "hain", "ho", "tha", "thi", "the", "ke", "ka", "ki", "ko",
    "mein", "me", "se", "par", "aur", "bhi", "hi", "ne", "ye", "yeh",
    "wo", "woh", "jo", "jab", "tab", "ab", "ek", "liye", "raha", "rahi",
    "rahe", "gaya", "gayi", "gaye", "kar", "karne", "kiya", "kya", "hua",
    "urltoken"
})

def clean_hinglish_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_proba_for_lime(texts):
    return model.predict_proba([clean_hinglish_text(t) for t in texts])

explainer = LimeTextExplainer(
    class_names=["Real News", "Fake News"],
    split_expression=r"\s+",
    bow=True,
    random_state=42,
)

def explain_news(text: str, output_html: str = "lime_explanation.html"):
    probs = predict_proba_for_lime([text])[0]
    pred_idx = int(np.argmax(probs))
    label = "Fake News" if pred_idx == 1 else "Real News"

    print("=" * 60)
    print(f"Input      : {text[:180]}{'...' if len(text) > 180 else ''}")
    print(f"Prediction : {label} ({probs[pred_idx] * 100:.2f}% confidence)")
    print(f"Scores     : Real = {probs[0]:.4f} | Fake = {probs[1]:.4f}")

    exp = explainer.explain_instance(
        clean_hinglish_text(text),
        predict_proba_for_lime,
        num_features=16,
        num_samples=1000,
    )

    meaningful = [
        (w, wt) for w, wt in exp.as_list()
        if re.sub(r"[^a-z]", "", w.lower()) not in HINGLISH_FILLERS
    ][:10]

    print("\nTop Meaningful Word Contributions (+ Fake | - Real):")
    for w, wt in meaningful:
        tag = "FAKE" if wt > 0 else "REAL"
        print(f"  {w:<22} : {wt:+.4f} [{tag}]")

    exp.save_to_file(output_html)
    print(f"Saved interactive LIME explanation -> {output_html}")

if __name__ == "__main__":
    sample = "breaking news rbi ne aaj raat se sabhi 500 ke note band karne ka aadesh jari kiya hai whatsapp forward"
    explain_news(sample, "lime_v2_explanation.html")
