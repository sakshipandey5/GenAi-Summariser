from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
   
    limited_text = text[:3000]
    summary = summarizer(limited_text, max_length=150, min_length=50, do_sample=False)

    return summary[0]["summary_text"]


