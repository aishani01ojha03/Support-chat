import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FAQItem:
    tag: str
    question: str
    answer: str


class SupportChatbot:
    """
    Semantic-search FAQ chatbot:
    - Encodes FAQ questions using a sentence-transformer model
    - Encodes user query
    - Picks best match by cosine similarity
    - Uses threshold to decide fallback
    """

    def __init__(
        self,
        faq_path: str = "data/faq.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.55,
    ):
        self.faq_path = faq_path
        self.model_name = model_name
        self.threshold = float(threshold)

        self.model = SentenceTransformer(self.model_name)

        self.faq_items: List[FAQItem] = self._load_faq(self.faq_path)
        if not self.faq_items:
            raise ValueError("FAQ file is empty. Add at least 1 Q/A pair in data/faq.json")

        self.questions = [x.question for x in self.faq_items]
        self.answers = [x.answer for x in self.faq_items]
        self.tags = [x.tag for x in self.faq_items]

        # Precompute embeddings for FAQ questions (fast at runtime)
        self.faq_embeddings = self.model.encode(self.questions, convert_to_numpy=True, normalize_embeddings=True)

    def _load_faq(self, path: str) -> List[FAQItem]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items: List[FAQItem] = []
        for r in raw:
            if "question" in r and "answer" in r:
                items.append(
                    FAQItem(
                        tag=r.get("tag", "unknown"),
                        question=r["question"].strip(),
                        answer=r["answer"].strip(),
                    )
                )
        return items

    def reply(self, user_message: str) -> Dict:
        user_message = (user_message or "").strip()
        if not user_message:
            return {
                "answer": "Please type your question so I can help.",
                "confidence": 0.0,
                "matched_question": None,
                "tag": None,
            }

        query_emb = self.model.encode([user_message], convert_to_numpy=True, normalize_embeddings=True)
        sims = cosine_similarity(query_emb, self.faq_embeddings)[0]  # shape: (n_faq,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            return {
                "answer": (
                    "I’m not fully sure I understood. "
                    "Can you rephrase, or share your order ID / product name so I can help better?"
                ),
                "confidence": best_score,
                "matched_question": self.questions[best_idx],
                "tag": self.tags[best_idx],
            }

        return {
            "answer": self.answers[best_idx],
            "confidence": best_score,
            "matched_question": self.questions[best_idx],
            "tag": self.tags[best_idx],
        }