from __future__ import annotations
import re
from typing import Dict, Any, Optional

# add near the top with other hints
_RESULTS_QA_HINTS = (
    r"\bask\b.*\b(my|our)\b.*\bresult[s]?\b",
    r"\bquestion\b.*\bresult[s]?\b",
    r"\bhow many\b.*\b(day|blast|embryo)\b",
    r"\bwhat\b.*\bgrade[s]?\b",
)

_UPLOAD_HINTS = (
    r"\bupload\b", r"\battach\b", r"\bsend (?:a )?file\b",
    r"\bprocess (?:my )?(?:report|pdf|image|file)\b",
    r"\bextract\b", r"\bparse (?:this|that|my) (?:report|pdf|image)\b",
)
_RESULT_HINTS = (
    r"\b(result|results|report|summary)\b",
    r"\bembryology\b", r"\bday\s*(\d+)\b", r"\bblast(?:ocyst)?\b", r"\bgrade[s]?\b",
    r"\bshow (?:my|the) (?:result|summary)\b",
)
_POLICY_HINTS = (
    r"\bpolicy\b", r"\binsurance\b", r"\bbilling\b", r"\bconsent\b", r"\bprivacy\b",
    r"\bcancellation\b", r"\brefund\b", r"\bclinic\b.*\bhours\b", r"\bschedule\b|\bappointment\b",
)
# NEW appointment/treatment patterns
_APPT_HINTS = (
    r"\bappointment\b", r"\bbook\b|\bschedule\b", r"\breschedule\b", r"\bcancel\b.*\bappointment\b",
    r"\bwhen\b.*\bnext\b.*\bappointment\b", r"\bmy next appointment\b"
)
_TREAT_HINTS = (
    r"\btreatment\b", r"\bongoing\b.*\btreatment\b", r"\bprotocol\b", r"\bregimen\b",
    r"\biui\b|\bfet\b|\bstims?\b"
)

_GREETING_HINTS = (r"\bhi\b|\bhello\b|\bhey\b|\bgood (morning|afternoon|evening)\b",)
_SMALLTALK_HINTS = (r"\bthanks?\b|\bthank you\b|\bok\b|\bgreat\b|\bnice\b|\bbye\b",)

_PATTERNS = {
    
    # add to _PATTERNS dict
    "results_qa": re.compile("|".join(_RESULTS_QA_HINTS), re.I),

    "upload_parse": re.compile("|".join(_UPLOAD_HINTS), re.I),
    "personal_result": re.compile("|".join(_RESULT_HINTS), re.I),
    "policy": re.compile("|".join(_POLICY_HINTS), re.I),
    "appointments": re.compile("|".join(_APPT_HINTS), re.I),
    "treatments": re.compile("|".join(_TREAT_HINTS), re.I),
    "greeting": re.compile("|".join(_GREETING_HINTS), re.I),
    "smalltalk": re.compile("|".join(_SMALLTALK_HINTS), re.I),
}

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def classify_intent(user_msg: str) -> str:
    t = _normalize(user_msg)
    if not t: return "unknown"
     
    for key in ("upload_parse","personal_result","results_qa","appointments","treatments","policy","greeting","smalltalk"):
        if _PATTERNS[key].search(t): return key
    # clinical words → faq
    if any(k in t for k in ["amh","fsh","estradiol","progesterone","hsg","ivf","iui","icsi",
                            "embryo","transfer","trigger","stimulation","antral","follicle","beta hcg","luteal"]):
        return "faq"
    return "faq"

def route(message: str, **kwargs) -> Dict[str, Any]:
    patient_id: Optional[str] = kwargs.get("patient_id")
    has_pending_upload: bool = bool(kwargs.get("has_pending_upload", False))
    clinic_namespace: str = kwargs.get("clinic_namespace", "patient_education")
    intent = classify_intent(message)

    if intent == "upload_parse":
        if has_pending_upload:
            return {"intent": intent, "action": "extract",
                    "params": {"patient_id": patient_id, "need_upload": True, "use_patient_ns": True,
                               "namespace": clinic_namespace}}
        return {"intent": intent, "action": "clarify",
                "params": {"patient_id": patient_id, "need_upload": True, "use_patient_ns": True,
                           "namespace": clinic_namespace, "message": "Please attach a PDF/Image and click Process."}}

    if intent == "personal_result":
        return {"intent": intent, "action": "show_result",
                "params": {"patient_id": patient_id, "need_upload": False, "use_patient_ns": True,
                           "namespace": clinic_namespace}}

    # NEW: appointments & treatments
    if intent == "appointments":
        return {"intent": intent, "action": "appointments",
                "params": {"patient_id": patient_id}}

    if intent == "treatments":
        return {"intent": intent, "action": "treatments",
                "params": {"patient_id": patient_id}}
    
    if intent == "results_qa":
        return {"intent": intent, "action": "results_qa",
                "params": {"patient_id": patient_id, "namespace": clinic_namespace}}

    # policy/faq/greeting/smalltalk → answer via hybrid RAG
    if intent in ("policy","faq","greeting","smalltalk","unknown"):
        return {"intent": intent, "action": "answer",
                "params": {"patient_id": patient_id, "need_upload": False, "use_patient_ns": True,
                           "namespace": clinic_namespace}}

    return {"intent": "faq", "action": "answer",
            "params": {"patient_id": patient_id, "namespace": clinic_namespace}}
