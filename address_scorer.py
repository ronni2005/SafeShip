import re


def score_address_completeness(address_text: str) -> int:
    """
    Genuine rule-based address parser. Reads the actual string and scores
    it out of 100 based on which real-world delivery-critical components
    are present. Weights are set so ONE missing minor field (e.g. PIN)
    costs a modest amount, not enough alone to tank the score — matches
    how couriers actually cope with partial addresses in practice.

    Scoring breakdown (out of 100):
      40  - base, for having any street/area text at all
      +27 - house/flat/door/plot number present
      +23 - 6-digit PIN code present
      +10 - landmark present (near/opposite/behind X)
    """
    if not address_text or not address_text.strip():
        return 20  # empty address — floor score, not zero, since an order can still sometimes be delivered on name + phone number alone

    score = 40

    if re.search(r"(h\.?\s?no\.?|flat|plot\s?no\.?|door\s?no\.?)\s*\.?\s*\d+", address_text, re.IGNORECASE) \
       or re.search(r"^\d+[,\s]", address_text):
        score += 27

    if re.search(r"\b\d{6}\b", address_text):
        score += 23

    if re.search(r"(near|opposite|behind)\s+\w+", address_text, re.IGNORECASE):
        score += 10

    return min(score, 100)