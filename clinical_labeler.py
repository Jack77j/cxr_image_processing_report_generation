"""
Lightweight rule-based CheXpert-style labeler (no model download needed).

For each of the 14 CheXpert observations a report is labelled 1 (mentioned as present or
uncertain) or 0 (absent / negated / not mentioned). Negation is detected with cue words
("no", "without", "negative for", "free of", ...) that occur *before* the finding inside
the same sentence scope (a scope ends at "but", "however", ";" etc.).

This is an approximation of the Stanford CheXpert labeler / CheXbert. It is good enough to
compare models against each other; for publication-grade numbers use CheXbert
(evaluate.py --chexbert) if you can install it.
"""
import re

OBSERVATIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
]
# the 5 observations most papers report ("CheXbert-5") + pneumothorax from the project goals
KEY_OBSERVATIONS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion", "Pneumothorax"]

_PATTERNS = {
    "Enlarged Cardiomediastinum": r"widen\w* (?:of the )?(?:superior )?mediastin\w*|mediastin\w* (?:is |appears )?widen\w*|"
                                  r"enlarged cardiomediastin\w*|cardiomediastinal (?:silhouette|contours?) (?:is |are |appears? )?(?:\w+ )?enlarged|"
                                  r"mediastinal (?:widening|enlargement)",
    "Cardiomegaly": r"cardiomegaly|(?:heart|cardiac silhouette|cardiac size|heart size) (?:is |appears |remains )?"
                    r"(?:\w+ )?(?:enlarged|large)|enlarge\w* (?:of the )?(?:heart|cardiac)|cardiac enlargement|enlarged heart",
    "Lung Lesion": r"nodul\w*|\bmass(?:es)?\b|lesion|tumou?r|neoplas\w*|carcinoma|malignan\w*|cavitary|metasta\w*",
    "Lung Opacity": r"opacit\w*|opacific\w*|densit(?:y|ies)|infiltrat\w*|hazy|haziness|airspace (?:disease|process)|"
                    r"reticular|interstitial (?:marking|abnormalit|prominence|opacit)\w*|ground[- ]glass",
    "Edema": r"edema|oedema|vascular congestion|pulmonary congestion|\bchf\b|heart failure|fluid overload|"
             r"vascular (?:engorgement|redistribution)|cephalization",
    "Consolidation": r"consolidat\w*",
    "Pneumonia": r"pneumonia|infection|infectious process|pneumonitis",
    "Atelectasis": r"atelecta\w*|collapse",
    "Pneumothorax": r"pneumothora\w*|pneumothoraces",
    "Pleural Effusion": r"(?<!pericardial )effusion|pleural fluid|hydrothorax|blunting of the (?:\w+ )?costophrenic|"
                        r"costophrenic (?:angle )?blunting",
    "Pleural Other": r"pleural (?:thickening|scar\w*|plaque\w*|calcification)|fibrothorax|pleural-parenchymal scar\w*",
    "Fracture": r"fracture\w*",
    "Support Devices": r"\b(?:et|endotracheal|ng|og|nasogastric|orogastric|enteric|feeding|chest|dobhoff|tracheostomy) tubes?\b|"
                       r"\bpicc\b|central (?:venous )?(?:line|catheter)|\bcatheter|pacemaker|\bpacer\b|\bicd\b|"
                       r"\bleads?\b|sternotomy wires?|\bport\b|port-a-cath|\bstent|\bclips?\b|prosthe\w*|"
                       r"valve replacement|swan-?ganz|\bdrain\b|\bdevice",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in _PATTERNS.items()}

_NEGATION = re.compile(
    r"\b(?:no|not|without|negative for|free of|clear of|absence of|resolved|resolution of|"
    r"removal of|removed|rule out|ruled out|neither|nor|no evidence of|no signs? of|"
    r"there is no|there are no|no new|no definite|no focal|no acute)\b",
    re.IGNORECASE,
)
_SCOPE_BREAK = re.compile(r"\b(?:but|however|although|though|which|except|aside from|apart from)\b|[;:]",
                          re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NORMAL_HEART = re.compile(r"(?:heart|cardiac) (?:size|silhouette) (?:is |appears )?(?:within )?(?:normal|upper limits? of normal|"
                           r"top[- ]normal)", re.IGNORECASE)


def _is_negated(sentence, start):
    """True if a negation cue appears before position `start` within the same scope."""
    prefix = sentence[:start]
    breaks = list(_SCOPE_BREAK.finditer(prefix))
    if breaks:
        prefix = prefix[breaks[-1].end():]
    return bool(_NEGATION.search(prefix))


def label_report(text):
    """Return {observation: 0/1} for one report."""
    labels = {o: 0 for o in OBSERVATIONS}
    if not isinstance(text, str) or not text.strip():
        labels["No Finding"] = 1
        return labels
    for sentence in _SENT_SPLIT.split(text):
        for obs, pattern in _COMPILED.items():
            if labels[obs]:
                continue
            for m in pattern.finditer(sentence):
                if obs == "Cardiomegaly" and _NORMAL_HEART.search(sentence):
                    break
                if not _is_negated(sentence, m.start()):
                    labels[obs] = 1
                    break
    abnormal = [o for o in OBSERVATIONS if o not in ("No Finding", "Support Devices")]
    labels["No Finding"] = int(not any(labels[o] for o in abnormal))
    return labels


def label_reports(texts):
    return [label_report(t) for t in texts]


if __name__ == "__main__":
    for t in ["No pleural effusion or pneumothorax. Heart size is normal.",
              "Moderate cardiomegaly with mild pulmonary edema. Small left pleural effusion.",
              "ET tube terminates 3 cm above the carina. No focal consolidation, but there is left basilar atelectasis."]:
        print(t, "->", [k for k, v in label_report(t).items() if v])
