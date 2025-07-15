import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# 🎌 Sample anime sentences (simulated social media data)
anime_sentences = [
    "Gojo Satoru is the most badass character ever!",
    "I hate how they ended Attack on Titan. Terrible writing.",
    "One Piece is a legendary anime. Oda is a genius!",
    "Naruto had a lot of fillers, but the story was emotional.",
    "Demon Slayer's animation is top-tier!",
    "Boruto is boring compared to Naruto.",
    "Spy x Family is a fun and wholesome anime.",
    "Tokyo Ghoul's ending ruined everything.",
    "Bleach comeback is hype!",
    "The pacing in Dragon Ball Super is awful."
]

# 🔍 Sentiment classification
sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

for sentence in anime_sentences:
    score = sia.polarity_scores(sentence)
    if score['compound'] >= 0.05:
        sentiment_counts["positive"] += 1
    elif score['compound'] <= -0.05:
        sentiment_counts["negative"] += 1
    else:
        sentiment_counts["neutral"] += 1

# 📊 Visualization
labels = list(sentiment_counts.keys())
values = list(sentiment_counts.values())

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=['green', 'grey', 'red'])
plt.title("Sentiment Analysis of Anime Opinions")
plt.xlabel("Sentiment")
plt.ylabel("Number of Sentences")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()