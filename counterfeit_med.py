import os
import json
import base64
from typing import Any, Dict

import streamlit as st
from openai import OpenAI


st.set_page_config(page_title="Counterfeit Medicine Detection", page_icon="💊", layout="wide")


SYSTEM_PROMPT = """
You are a pharmaceutical packaging inspection assistant.
Your job is to analyze an uploaded medicine package image and estimate whether it might be counterfeit.

Important constraints:
- You are NOT a lab test.
- You cannot confirm authenticity from an image alone.
- You must clearly say this is a visual risk screening only.
- Focus on visible signs such as spelling mistakes, inconsistent fonts, missing regulatory markings,
  tampered seals, odd pill blister patterns, bad print quality, missing batch/expiry info,
  suspicious branding, and packaging damage.
- Be conservative and avoid overclaiming.

Return strict JSON with this schema:
{
  "verdict": "Likely genuine" | "Suspicious" | "High risk of counterfeit" | "Insufficient evidence",
  "confidence": 0,
  "summary": "short plain-English summary",
  "signals_for_authenticity": ["..."],
  "signals_of_concern": ["..."],
  "recommended_next_steps": ["..."],
  "medical_safety_notice": "..."
}
""".strip()


def get_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Add it to Streamlit secrets or environment variables."
        )
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False)
def image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model response did not contain JSON.")

    return json.loads(text[start:end + 1])


def analyze_medicine_image(image_bytes: bytes, mime_type: str, user_notes: str) -> Dict[str, Any]:
    client = get_client()
    data_url = image_to_data_url(image_bytes, mime_type)

    prompt = f"""
Analyze this medicine image for signs of possible counterfeit packaging.
User notes: {user_notes or 'None provided'}

Remember:
- This is only a visual screening, not proof.
- Output JSON only.
- Keep recommendations practical for an ordinary consumer.
- Include advice to verify with a pharmacist, manufacturer, or regulator when suspicious.
""".strip()

    response = client.responses.create(
        model="gpt-5.4",
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT}
                ],
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

    return extract_json(response.output_text)


st.title("💊 Counterfeit Medicine Detection")
st.caption("AI-powered visual screening for suspicious medicine packaging using OpenAI")

with st.sidebar:
    st.header("How it works")
    st.write(
        "Upload a photo of medicine packaging, blister pack, bottle label, or carton. "
        "The app checks for visible warning signs such as poor print quality, labeling issues, "
        "tampered seals, or suspicious branding inconsistencies."
    )
    st.warning(
        "This is not a medical or forensic device. A photo cannot confirm whether a medicine is genuine."
    )
    st.info(
        "For suspected counterfeits, do not consume the product until it has been verified by a pharmacist, "
        "manufacturer, or medicines regulator."
    )

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload medicine image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear, well-lit image showing the label, batch number, expiry date, and seal if possible.",
    )
    user_notes = st.text_area(
        "Optional notes",
        placeholder="Example: Bought from an online seller. Seal looked odd. Batch number is blurred.",
        height=120,
    )
    analyze = st.button("Analyze image", type="primary", use_container_width=True)

with col2:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    else:
        st.empty()

if analyze:
    if not uploaded_file:
        st.error("Please upload an image first.")
    else:
        try:
            image_bytes = uploaded_file.read()
            mime_type = uploaded_file.type or "image/jpeg"

            with st.spinner("Inspecting packaging for suspicious visual signals..."):
                result = analyze_medicine_image(image_bytes, mime_type, user_notes)

            verdict = result.get("verdict", "Insufficient evidence")
            confidence = result.get("confidence", 0)
            summary = result.get("summary", "No summary returned.")
            good_signals = result.get("signals_for_authenticity", [])
            concern_signals = result.get("signals_of_concern", [])
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

            st.subheader("Recommended next steps")
            for item in next_steps:
                st.markdown(f"- {item}")

            if safety_notice:
                st.warning(safety_notice)

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
3. Run the app: `streamlit run counterfeit_medicine_detector_app.py`

### Suggested improvements
- Add OCR to read batch number, expiry date, and registration number.
- Compare packaging against manufacturer reference images.
- Add country-specific medicine regulator checks.
- Log suspicious cases for human review.
"""
)
