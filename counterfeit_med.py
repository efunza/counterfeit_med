import os
import re
import json
import base64
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="Counterfeit Medicine Detection", page_icon="💊", layout="wide")


SYSTEM_PROMPT = """
You are a pharmaceutical packaging inspection assistant.
Your job is to analyze uploaded medicine packaging images and estimate whether they might be counterfeit.

Important constraints:
- You are NOT a lab test.
- You cannot confirm authenticity from images alone.
- You must clearly say this is a visual risk screening only.
- Focus on visible signs such as spelling mistakes, inconsistent fonts, missing regulatory markings,
  tampered seals, odd pill blister patterns, bad print quality, missing batch/expiry info,
  suspicious branding, and packaging damage.
- If OCR text is available, use it to assess whether batch number, expiry date, or registration number are visible.
- If reference images are available, compare the uploaded package against them and note similarities and differences.
- Be conservative and avoid overclaiming.

Return strict JSON with this schema:
{
  "verdict": "Likely genuine" | "Suspicious" | "High risk of counterfeit" | "Insufficient evidence",
  "confidence": 0,
  "summary": "short plain-English summary",
  "signals_for_authenticity": ["..."],
  "signals_of_concern": ["..."],
  "ocr_findings": {
    "batch_number": "... or Not found",
    "expiry_date": "... or Not found",
    "registration_number": "... or Not found"
  },
  "reference_comparison": {
    "matches_reference": true,
    "similarities": ["..."],
    "differences": ["..."]
  },
  "regulator_check": {
    "country": "...",
    "status": "Pass" | "Warning" | "Not checked",
    "notes": ["..."]
  },
  "recommended_next_steps": ["..."],
  "medical_safety_notice": "..."
}
""".strip()


COUNTRY_RULES: Dict[str, Dict[str, Any]] = {
    "Kenya": {
        "regulator": "Pharmacy and Poisons Board (PPB)",
        "registration_keywords": ["PPB", "Pharmacy and Poisons Board"],
        "registration_patterns": [r"\bPPB[-\s]?[A-Z0-9]{3,}\b"],
        "notes": [
            "Check whether PPB-related registration details are visible on the packaging.",
            "Verify suspicious products with the Pharmacy and Poisons Board or a licensed pharmacist.",
        ],
    },
    "Nigeria": {
        "regulator": "NAFDAC",
        "registration_keywords": ["NAFDAC"],
        "registration_patterns": [r"\bNAFDAC\b", r"\bA\d{2,}-\d{2,}\b"],
        "notes": [
            "Look for NAFDAC markings and product registration details on the pack.",
            "If the markings are absent or look altered, escalate for manual verification.",
        ],
    },
    "India": {
        "regulator": "CDSCO / State drug regulators",
        "registration_keywords": ["Rx", "Schedule", "Mfg Lic", "Manufacturing Licence"],
        "registration_patterns": [r"\bMfg\.?\s*Lic\.?\s*No\.?\b", r"\bLic\.?\s*No\.?\b"],
        "notes": [
            "Check whether manufacturing licence information appears clearly and matches the label layout.",
            "Escalate if licence, batch, or expiry details are missing or visibly tampered with.",
        ],
    },
    "United States": {
        "regulator": "FDA",
        "registration_keywords": ["NDC", "LOT", "EXP"],
        "registration_patterns": [r"\bNDC\b", r"\b\d{4,5}-\d{3,4}-\d{1,2}\b"],
        "notes": [
            "Check whether NDC or other expected labeling markers are present and readable.",
            "Use packaging inconsistencies as a warning sign, but do not claim formal FDA verification.",
        ],
    },
    "United Kingdom": {
        "regulator": "MHRA",
        "registration_keywords": ["PL", "LOT", "EXP"],
        "registration_patterns": [r"\bPL\s*\d{2,}/\d{3,4}\b"],
        "notes": [
            "Look for a visible PL number and readable batch/expiry details.",
            "Escalate any suspected counterfeit to a pharmacist or MHRA reporting channel.",
        ],
    },
    "Other": {
        "regulator": "Local medicines regulator",
        "registration_keywords": [],
        "registration_patterns": [],
        "notes": [
            "Check for the country-specific registration or product license details expected in your market.",
            "If the source is suspicious, ask a licensed pharmacist or the national medicines regulator to verify it.",
        ],
    },
}


LOG_DIR = Path("review_logs")
LOG_DIR.mkdir(exist_ok=True)
JSONL_LOG = LOG_DIR / "suspicious_cases.jsonl"
CSV_LOG = LOG_DIR / "suspicious_cases.csv"


def get_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to Streamlit secrets or environment variables.")
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False)
def image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON.")

    return json.loads(text[start : end + 1])


def safe_get_text_output(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    return str(response)


def ocr_extract_fields(ocr_text: str) -> Dict[str, str]:
    cleaned = re.sub(r"\s+", " ", ocr_text or "").strip()

    patterns = {
        "batch_number": [
            r"(?:batch|lot|bn|b\.no|batch no\.?|lot no\.?)[:\s-]*([A-Z0-9\-/]{3,})",
        ],
        "expiry_date": [
            r"(?:exp|expiry|expires|use before|best before)[:\s-]*([A-Z0-9\-/]{3,20})",
            r"\b((?:0?[1-9]|1[0-2])[\-/](?:20)?\d{2,4})\b",
        ],
        "registration_number": [
            r"(?:reg(?:istration)?\s*(?:no\.?|number)?|license no\.?|lic no\.?)[:\s-]*([A-Z0-9\-/]{3,})",
            r"\b(?:PPB[-\s]?[A-Z0-9]{3,}|PL\s*\d{2,}/\d{3,4}|\d{4,5}-\d{3,4}-\d{1,2})\b",
        ],
    }

    results: Dict[str, str] = {}
    for field, field_patterns in patterns.items():
        found = "Not found"
        for pattern in field_patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                found = match.group(1).strip() if match.groups() else match.group(0).strip()
                break
        results[field] = found
    return results


def run_ocr(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    client = get_client()
    data_url = image_to_data_url(image_bytes, mime_type)
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Read all visible text from this medicine package image. "
                            "Return strict JSON: {\"full_text\": \"...\"}. "
                            "Do not invent text that is not visible."
                        ),
                    },
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            }
        ],
    )
    ocr_json = extract_json(safe_get_text_output(response))
    full_text = str(ocr_json.get("full_text", "")).strip()
    fields = ocr_extract_fields(full_text)
    return {"full_text": full_text, "fields": fields}


def compare_with_reference_images(
    suspect_bytes: bytes,
    suspect_mime: str,
    reference_images: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not reference_images:
        return {
            "matches_reference": False,
            "similarities": [],
            "differences": ["No manufacturer reference images were provided for comparison."],
        }

    client = get_client()
    content: List[Dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Compare the first image, which is the suspected package, against the following manufacturer reference images. "
                "Assess only visible packaging similarities and differences. "
                "Return strict JSON with this schema: "
                "{\"matches_reference\": true, \"similarities\": [\"...\"], \"differences\": [\"...\"]}."
            ),
        },
        {"type": "input_image", "image_url": image_to_data_url(suspect_bytes, suspect_mime), "detail": "high"},
    ]

    for item in reference_images:
        content.append({"type": "input_text", "text": f"Reference image: {item['name']}"})
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(item["bytes"], item["mime_type"]),
                "detail": "high",
            }
        )

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4.1")),
        input=[{"role": "user", "content": content}],
    )
    return extract_json(safe_get_text_output(response))


def run_regulator_check(country: str, ocr_fields: Dict[str, str], ocr_text: str) -> Dict[str, Any]:
    rules = COUNTRY_RULES.get(country, COUNTRY_RULES["Other"])
    notes: List[str] = list(rules.get("notes", []))
    status = "Not checked"

    if country:
        status = "Pass"
        reg_value = ocr_fields.get("registration_number", "Not found")
        if reg_value == "Not found":
            status = "Warning"
            notes.append(f"No clear registration number was extracted for {country}.")

        for keyword in rules.get("registration_keywords", []):
            if keyword.lower() in (ocr_text or "").lower():
                notes.append(f"Found keyword associated with {rules['regulator']}: {keyword}")
                break
        else:
            if rules.get("registration_keywords"):
                status = "Warning"
                notes.append(f"No obvious {rules['regulator']} keyword was detected in OCR text.")

        matched_pattern = False
        for pattern in rules.get("registration_patterns", []):
            if re.search(pattern, ocr_text or "", flags=re.IGNORECASE):
                matched_pattern = True
                break
        if rules.get("registration_patterns") and not matched_pattern:
            status = "Warning"
            notes.append(f"Expected registration-style format for {country} was not clearly detected.")

    return {"country": country, "status": status, "notes": notes}


def analyze_medicine_image(
    image_bytes: bytes,
    mime_type: str,
    user_notes: str,
    country: str,
    ocr_result: Dict[str, Any],
    reference_result: Dict[str, Any],
) -> Dict[str, Any]:
    client = get_client()
    data_url = image_to_data_url(image_bytes, mime_type)

    prompt = f"""
Analyze this medicine image for signs of possible counterfeit packaging.
User notes: {user_notes or 'None provided'}
Country for regulator context: {country}
OCR extracted fields: {json.dumps(ocr_result.get('fields', {}), ensure_ascii=False)}
OCR full text: {ocr_result.get('full_text', '')[:4000]}
Reference comparison: {json.dumps(reference_result, ensure_ascii=False)}

Remember:
- This is only a visual screening, not proof.
- Output JSON only.
- Keep recommendations practical for an ordinary consumer.
- Include advice to verify with a pharmacist, manufacturer, or regulator when suspicious.
""".strip()

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")),
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": "high"},
                ],
            },
        ],
    )

    result = extract_json(safe_get_text_output(response))
    result["ocr_findings"] = ocr_result.get("fields", {})
    result["reference_comparison"] = reference_result
    result["regulator_check"] = run_regulator_check(country, ocr_result.get("fields", {}), ocr_result.get("full_text", ""))
    return result


def hash_file(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def log_case_for_review(
    result: Dict[str, Any],
    country: str,
    source_filename: str,
    user_notes: str,
    image_bytes: bytes,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {
        "timestamp_utc": timestamp,
        "case_id": hash_file(image_bytes),
        "filename": source_filename,
        "country": country,
        "verdict": result.get("verdict", ""),
        "confidence": result.get("confidence", 0),
        "summary": result.get("summary", ""),
        "signals_of_concern": result.get("signals_of_concern", []),
        "ocr_findings": result.get("ocr_findings", {}),
        "regulator_check": result.get("regulator_check", {}),
        "user_notes": user_notes,
        "requires_human_review": True,
    }

    with JSONL_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_exists = CSV_LOG.exists()
    csv_row = {
        "timestamp_utc": timestamp,
        "case_id": record["case_id"],
        "filename": source_filename,
        "country": country,
        "verdict": record["verdict"],
        "confidence": record["confidence"],
        "summary": record["summary"],
        "signals_of_concern": " | ".join(record["signals_of_concern"]),
        "batch_number": result.get("ocr_findings", {}).get("batch_number", ""),
        "expiry_date": result.get("ocr_findings", {}).get("expiry_date", ""),
        "registration_number": result.get("ocr_findings", {}).get("registration_number", ""),
        "review_flag": "Yes",
    }

    import csv

    with CSV_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if not csv_exists:
            writer.writeheader()
        writer.writerow(csv_row)


def should_flag_for_review(result: Dict[str, Any]) -> bool:
    verdict = str(result.get("verdict", "")).lower()
    reg_status = str(result.get("regulator_check", {}).get("status", "")).lower()
    has_concerns = bool(result.get("signals_of_concern"))
    return verdict in {"suspicious", "high risk of counterfeit"} or reg_status == "warning" or has_concerns


st.title("💊 Counterfeit Medicine Detection By Galana Boys")
st.caption("AI-powered visual screening for suspicious medicine packaging using OpenAI")

with st.sidebar:
    st.header("How it works")
    st.write(
        "Upload a photo of medicine packaging, blister pack, bottle label, or carton. "
        "The app checks visible warning signs, extracts text like batch and expiry information, "
        "compares the pack against manufacturer reference images, and applies country-specific regulator checks."
    )
    st.warning("This is not a medical or forensic device. Images alone cannot confirm authenticity.")
    st.info(
        "For suspected counterfeits, do not consume the product until it has been verified by a pharmacist, manufacturer, or medicines regulator."
    )

left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader(
        "Upload suspected medicine image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear, well-lit image showing the label, batch number, expiry date, and seal if possible.",
    )

    reference_files = st.file_uploader(
        "Upload manufacturer reference images (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload official product images from the manufacturer or authorized distributor to compare packaging layout and print quality.",
    )

    country = st.selectbox("Country / market for regulator checks", options=list(COUNTRY_RULES.keys()), index=0)

    user_notes = st.text_area(
        "Optional notes",
        placeholder="Example: Bought from an online seller. Seal looked odd. Batch number is blurred.",
        height=120,
    )

    analyze = st.button("Analyze image", type="primary", use_container_width=True)

with right:
    if uploaded_file:
        st.image(uploaded_file, caption="Suspected package", use_container_width=True)
    if reference_files:
        st.write("Reference images")
        preview_cols = st.columns(min(3, len(reference_files)))
        for idx, ref in enumerate(reference_files[:3]):
            with preview_cols[idx % len(preview_cols)]:
                st.image(ref, caption=ref.name, use_container_width=True)

if analyze:
    if not uploaded_file:
        st.error("Please upload an image first.")
    else:
        try:
            image_bytes = uploaded_file.read()
            mime_type = uploaded_file.type or "image/jpeg"

            ref_payloads: List[Dict[str, Any]] = []
            for ref in reference_files or []:
                ref_payloads.append(
                    {
                        "name": ref.name,
                        "bytes": ref.read(),
                        "mime_type": ref.type or "image/jpeg",
                    }
                )

            with st.spinner("Running OCR, comparing packaging, and screening for counterfeit signals..."):
                ocr_result = run_ocr(image_bytes, mime_type)
                reference_result = compare_with_reference_images(image_bytes, mime_type, ref_payloads)
                result = analyze_medicine_image(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    user_notes=user_notes,
                    country=country,
                    ocr_result=ocr_result,
                    reference_result=reference_result,
                )

            verdict = result.get("verdict", "Insufficient evidence")
            confidence = result.get("confidence", 0)
            summary = result.get("summary", "No summary returned.")
            good_signals = result.get("signals_for_authenticity", [])
            concern_signals = result.get("signals_of_concern", [])
            ocr_findings = result.get("ocr_findings", {})
            reference_comparison = result.get("reference_comparison", {})
            regulator_check = result.get("regulator_check", {})
            next_steps = result.get("recommended_next_steps", [])
            safety_notice = result.get("medical_safety_notice", "")

            if verdict == "Likely genuine":
                st.success(f"Verdict: {verdict}")
            elif verdict == "Suspicious":
                st.warning(f"Verdict: {verdict}")
            elif verdict == "High risk of counterfeit":
                st.error(f"Verdict: {verdict}")
            else:
                st.info(f"Verdict: {verdict}")

            st.metric("Confidence", f"{confidence}%")

            st.subheader("Summary")
            st.write(summary)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Signals for authenticity")
                if good_signals:
                    for item in good_signals:
                        st.markdown(f"- {item}")
                else:
                    st.write("No strong authenticity signals identified.")

            with c2:
                st.subheader("Signals of concern")
                if concern_signals:
                    for item in concern_signals:
                        st.markdown(f"- {item}")
                else:
                    st.write("No obvious red flags identified from the visible image.")

            st.subheader("OCR findings")
            o1, o2, o3 = st.columns(3)
            o1.metric("Batch number", ocr_findings.get("batch_number", "Not found"))
            o2.metric("Expiry date", ocr_findings.get("expiry_date", "Not found"))
            o3.metric("Registration number", ocr_findings.get("registration_number", "Not found"))

            with st.expander("Extracted packaging text"):
                st.text(ocr_result.get("full_text", "No text extracted."))

            st.subheader("Reference image comparison")
            st.write(f"Matches reference: {reference_comparison.get('matches_reference', False)}")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Similarities**")
                for item in reference_comparison.get("similarities", []) or ["None noted"]:
                    st.markdown(f"- {item}")
            with rc2:
                st.markdown("**Differences**")
                for item in reference_comparison.get("differences", []) or ["None noted"]:
                    st.markdown(f"- {item}")

            st.subheader("Country-specific regulator check")
            st.write(f"Country: {regulator_check.get('country', country)}")
            st.write(f"Status: {regulator_check.get('status', 'Not checked')}")
            for note in regulator_check.get("notes", []):
                st.markdown(f"- {note}")

            st.subheader("Recommended next steps")
            for item in next_steps:
                st.markdown(f"- {item}")

            if safety_notice:
                st.warning(safety_notice)

            flagged = should_flag_for_review(result)
            if flagged:
                log_case_for_review(
                    result=result,
                    country=country,
                    source_filename=uploaded_file.name,
                    user_notes=user_notes,
                    image_bytes=image_bytes,
                )
                st.error("This case has been flagged and logged for human review.")
            else:
                st.success("This case was not auto-flagged for human review.")

            with st.expander("Raw JSON output"):
                st.json(result)

        except Exception as e:
            st.exception(e)

st.divider()
st.markdown(
    """
### Setup
1. Install dependencies: `pip install streamlit openai`
2. Set your API key:
   - Environment variable: `OPENAI_API_KEY=your_key`
   - Or Streamlit secrets: `.streamlit/secrets.toml`
3. Optional model override:
   - `OPENAI_MODEL=gpt-4.1`
4. Run the app: `streamlit run counterfeit_medicine_detector_app.py`

### What was added
- OCR extraction for batch number, expiry date, and registration number.
- Packaging comparison against uploaded manufacturer reference images.
- Country-specific regulator checks for a few example markets.
- Suspicious case logging to `review_logs/suspicious_cases.jsonl` and `review_logs/suspicious_cases.csv`.

### Notes
- OCR and packaging comparison are vision-based and may miss tiny or blurry text.
- The regulator checks are heuristic checks, not live regulatory database verification.
- For production deployment, replace file logging with a database or ticketing workflow for reviewers.
"""
)
