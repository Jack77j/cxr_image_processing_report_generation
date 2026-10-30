"""
Report-text cleaning for study-level report generation.

Raw MIMIC-CXR reports in the CSV look like
    "Findings: <findings text> Impression: <impression text>"
and the paraphrased `text_augment` versions look like
    "Results: <paraphrased findings> Impression: <paraphrased impression>"
(the first header is paraphrased too: Results / Finds / Observations / ...).

Steps applied to every report (original and paraphrase alike):
  1. remove de-identification placeholders ("___") and list numbering ("1. ", "2. ")
  2. split into Findings / Impression sections
  3. pick the target section(s)            -> TARGET_SECTION in config.py
  4. split into sentences and
       - drop sentences that only make sense with an earlier scan  (COMPARISON_MODE="drop")
       - drop sentences about phoning / notifying clinicians
       - strip temporal modifiers such as "new", "again", "persistent"
  5. normalise whitespace / punctuation
"""
import re

# --------------------------------------------------------------------------- #
# Regular expressions
# --------------------------------------------------------------------------- #
_PLACEHOLDER = re.compile(r"_{2,}")
_LIST_NUMBER = re.compile(r"(^|\s)\d{1,2}[.)]\s+(?=[A-Za-z])")
_IMPRESSION_SPLIT = re.compile(r"\bImpressions?\s*:", re.IGNORECASE)
_LEADING_HEADER = re.compile(r"^\s*(?:the\s+)?[A-Za-z]+\s*:\s*", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

# Sentences containing any of these refer to an earlier examination and cannot be
# grounded in a single study's images -> dropped when COMPARISON_MODE == "drop".
_COMPARISON = re.compile(
    r"\b("
    r"compar(?:ed|ison|ing)|prior|previous(?:ly)?|preceding|earlier|"
    r"interval(?:ly)?|since|as before|"
    r"unchanged|no (?:significant |substantial |appreciable |definite )?(?:interval )?change|"
    r"(?:essentially |grossly |overall )?stable|"
    r"redemonstrat\w*|re-demonstrat\w*|"
    r"improv(?:ed|ing|ement)|worsen(?:ed|ing)|progress(?:ed|ion)|"
    r"(?:has|have) (?:increased|decreased|resolved|developed|enlarged)|"
    r"(?:increase|decrease) in (?:size|extent|the)|resolution|resolved|"
    r"no longer|again|recent|last (?:study|exam|radiograph|film)"
    r")\b",
    re.IGNORECASE,
)

# Communication / administrative sentences (never visible in the image).
_COMMUNICATION = re.compile(
    r"\b(notif\w*|telephon\w*|phone|paged|page|conveyed|communicated|discussed|"
    r"called|relayed|informed|dr\.|nurse|provider|team|a\.m\.|p\.m\.|"
    r"recommend\w*|suggest(?:ed)? (?:clinical|follow|correlation)|follow-?up|"
    r"clinical correlation|findings were)\b|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)

# Temporal modifiers that can be removed while keeping the finding itself.
_TEMPORAL_WORDS = re.compile(
    r"\b(newly|new|persistent(?:ly)?|continued|residual|re-?demonstrated|again)\s+",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalise(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([.,;:]){2,}", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"^[\s.,;:]+", "", text)
    return text.strip()


def _capitalise(sentence: str) -> str:
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def basic_clean(text) -> str:
    """Remove de-id placeholders, list numbering and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = _PLACEHOLDER.sub(" ", text)
    text = _LIST_NUMBER.sub(" ", text)
    return _normalise(text)


def split_sections(text: str):
    """Return (findings, impression) from a cleaned report string."""
    parts = _IMPRESSION_SPLIT.split(text, maxsplit=1)
    findings = parts[0]
    impression = parts[1] if len(parts) > 1 else ""
    findings = _LEADING_HEADER.sub("", findings, count=1)
    return _normalise(findings), _normalise(impression)


def split_sentences(text: str):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def filter_sentences(text: str, comparison_mode: str = "drop"):
    """Drop ungroundable sentences. Returns (clean_text, n_dropped)."""
    kept, dropped = [], 0
    for sent in split_sentences(text):
        if _COMMUNICATION.search(sent):
            dropped += 1
            continue
        if comparison_mode == "drop":
            if _COMPARISON.search(sent):
                dropped += 1
                continue
            sent = _capitalise(_TEMPORAL_WORDS.sub("", sent))
        if not sent.endswith((".", "!", "?")):
            sent += "."
        if len(sent.split()) >= 2:
            kept.append(sent)
    return _normalise(" ".join(kept)), dropped


def build_target(report, section="findings_fallback_impression", comparison_mode="drop"):
    """
    Full cleaning for one report string.

    Returns a dict:
        target          - cleaned training target ("" if nothing usable is left)
        source_section  - "findings" | "impression" | "findings+impression" | ""
        findings_raw    - findings section before sentence filtering
        impression_raw  - impression section before sentence filtering
        n_dropped       - number of sentences removed by the filters
    """
    text = basic_clean(report)
    findings, impression = split_sections(text)
    f_clean, f_drop = filter_sentences(findings, comparison_mode) if findings else ("", 0)
    i_clean, i_drop = filter_sentences(impression, comparison_mode) if impression else ("", 0)

    if section == "findings_only":
        target, src, dropped = f_clean, "findings", f_drop
    elif section == "findings_impression":
        target = _normalise(f"{f_clean} {i_clean}")
        src, dropped = "findings+impression", f_drop + i_drop
    elif section == "findings_fallback_impression":
        if f_clean:
            target, src, dropped = f_clean, "findings", f_drop
        else:
            target, src, dropped = i_clean, "impression", f_drop + i_drop
    else:
        raise ValueError(f"Unknown TARGET_SECTION: {section}")

    if not target:
        src = ""
    return {
        "target": target,
        "source_section": src,
        "findings_raw": findings,
        "impression_raw": impression,
        "n_dropped": dropped,
    }


if __name__ == "__main__":
    demo = ("Findings: Compared to chest radiographs since ___, most recently ___. "
            "Large right and moderate left pleural effusions are unchanged. "
            "New right upper lobe opacity. No pneumothorax. Heart size is normal. "
            "Impression: 1. Interval increase in right effusion. 2. Findings were "
            "conveyed to Dr. ___ at 3:40 p.m.")
    for k, v in build_target(demo).items():
        print(f"{k:15s}: {v}")
