import asyncio
import json
import os
import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip()
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(EVAL_DIR, os.pardir))
TEST_CASE_LIMIT = int(os.environ.get("TEST_CASE_LIMIT", "10"))
JSONL_PATH = os.environ.get(
    "JSONL_PATH",
    os.path.join(WORKSPACE_ROOT, "Result", "llava16", "eval_run", "llava16_dpo_eval_max_slice_6_no_zero3.jsonl"),
)
JSONL_BASENAME = os.path.splitext(os.path.basename(JSONL_PATH))[0]
RESULT_PARENT_DIR = os.path.basename(os.path.dirname(os.path.dirname(JSONL_PATH)))
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    RESULT_PARENT_DIR,
).strip() or RESULT_PARENT_DIR
MODEL_DIR_NAME = re.sub(r"[\\/:*?\"<>|]+", "_", MODEL_NAME)
RUN_NAME = os.path.basename(os.path.dirname(JSONL_PATH)).removesuffix("_run")
RUN_DIR_NAME = re.sub(r"[\\/:*?\"<>|]+", "_", os.environ.get("RUN_DIR_NAME", "eval_train_dataset") or "eval_train_dataset")
OUTPUT_XLSX = os.environ.get(
    "OUTPUT_XLSX",
    os.path.join(EVAL_DIR, MODEL_DIR_NAME, RUN_DIR_NAME, JSONL_BASENAME + ".xlsx"),
)

MARKERS = ["TTF-1", "Napsin A", "CD56", "P63", "P40", "Syn", "CgA", "CK5/6", "Ki-67"]

PROMPT = (
    "You are an information extraction assistant for pathology reports. "
    "Extract only immunohistochemistry findings from the given text. "
    "Return a compact English result with semicolons between items. "
    "Rules: "
    "1) Keep only markers explicitly mentioned in the text. "
    "2) Treat any fuzzy or partial positivity expression as positive, including forms like partial+, partial positive, positive (focal), focal positive, weak positive, or similar wording; "
    "negative forms -> negative. "
    "3) If Ki-67 is present, keep the percentage; treat Ki-67 > 30% as Ki-67 positive. "
    "4) Do not include background text, diagnosis text, or unrelated findings. "
    "5) If no IHC is found, return none. "
)

STRUCTURED_OUTPUT_RULES = (
    "Output ONLY the 8 structured fields below, in this exact order. "
    "ALWAYS output all 8 fields; if a field is not mentioned in the source, write 'none'. "
    "Use one line per field in the format 'Field: value'. "
    "Fields:\n"
    "Cancer type:\n"
    "Differentiation:\n"
    "Growth patterns:\n"
    "Invasion:\n"
    "Margin:\n"
    "Immunohistochemistry:\n"
)

STRUCTURE_GENERATION_PROMPT = (
    "You are a clinical information structuring assistant for pathology reports. "
    "Rewrite the given generation result into the required structured 8-field format. "
    "Return ONLY the final structured result with no explanation, notes, markdown, or extra text. "
    "Do not use any ground truth or external information. "
    "Use the generation result as the only source. "
    "Standardize wording and merge duplicates while preserving clinically meaningful content. "
    "Field definitions: "
    "1) 'Cancer type' is the pure histologic entity only, e.g. 'Adenocarcinoma', 'Squamous cell carcinoma', "
    "   'Small cell carcinoma'. Do NOT include any modifiers such as 'invasive', 'minimally invasive', "
    "   'in situ', differentiation, growth patterns, invasion, margins, or lymph node information. "
    "2) 'Differentiation' is the differentiation degree only, "
    "   e.g. 'well differentiated', 'moderately differentiated', 'poorly differentiated'. "
    "   Do NOT include growth patterns. "
    "3) 'Growth patterns' are the architectural growth patterns of the tumor only, "
    "   e.g. 'acinar', 'lepidic', 'papillary', 'micropapillary', 'solid'. "
    "   Do NOT include differentiation degree. "
    "4) 'Invasion' describes tumor invasion and spread findings, "
    "   e.g. 'visceral pleural invasion present', 'vascular invasion absent', 'STAS present'. "
    "   Do NOT include surgical margin or lymph node findings. "
    "5) 'Margin' describes surgical margin status only, e.g. 'bronchial margin negative; staple line margin negative'. "
    "   Do NOT include invasion or lymph node findings. "
    "6) Exclude lymph node metastasis information entirely (e.g. 'lymph nodes negative', 'station 7: 1/5'). "
    "If a field is not mentioned in the source, write 'none'. "
    + STRUCTURED_OUTPUT_RULES
)

REPLACE_PROMPT = (
    "You are editing an already-structured pathology result. "
    "Return only the corrected structured result with no explanation, notes, markdown, or extra text. "
    "Rules: "
    "1) Only edit the 'Location' field and the 'Tumor size' field. "
    "2) Copy Location exactly from Ground Truth into the structured result when Ground Truth contains it. "
    "3) Copy Tumor size exactly from Ground Truth into the structured result when Ground Truth contains it. "
    "4) Do NOT modify Cancer type, Differentiation, Growth patterns, Invasion, Margin, or Immunohistochemistry. "
    "   Keep those fields exactly as they appear in the structured generation result. "
    "5) Do not add any ground-truth findings into other fields. "
    "6) Keep the same structured field order and omit any field not present. "
    + STRUCTURED_OUTPUT_RULES
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_case_id(entry):
    case_id = "Unknown"
    images = entry.get("images") or []
    if images:
        img_path = images[0].get("path", "") or ""
        m = re.search(r"Case_(\d+)", img_path)
        if m:
            case_id = f"Case_{m.group(1)}"
        else:
            m2 = re.search(r"(?i)Lung(\d+)", img_path)
            if m2:
                case_id = f"Lung{m2.group(1)}"
    return case_id


def extract_marker_status(text):
    if not text or not isinstance(text, str):
        return {}
    text = text.replace("＋", "+").replace("－", "-")
    status = {}

    patterns = {
        "TTF-1": r"TTF\s*-?\s*1[^;\n,。]*",
        "Napsin A": r"Napsin\s*A[^;\n,。]*",
        "CD56": r"CD56[^;\n,。]*",
        "P63": r"P63[^;\n,。]*",
        "P40": r"P40[^;\n,。]*",
        "Syn": r"(?:Synaptophysin|Syn)[^;\n,。]*",
        "CgA": r"(?:Chromogranin\s*-?\s*A|CgA|CGA)[^;\n,。]*",
        "CK5/6": r"CK5/?6[^;\n,。]*",
        "Ki-67": r"KI\s*-?\s*67[^;\n,。]*",
    }

    for marker, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        status[marker] = m.group(0).strip()
    return status


MARKER_PATTERNS = {
    "TTF-1": r"\bTTF\s*-?\s*1\b",
    "Napsin A": r"\bNapsin\s*A\b",
    "CD56": r"\bCD\s*56\b",
    "P63": r"\bP\s*63\b",
    "P40": r"\bP\s*40\b",
    "Syn": r"\b(?:Synaptophysin|Syn)\b",
    "CgA": r"\b(?:Chromogranin\s*-?\s*A|CgA|CGA)\b",
    "CK5/6": r"\bCK\s*5\s*/?\s*6\b",
    "Ki-67": r"\bKI\s*-?\s*67\b",
}

POSITIVE_PATTERN = r"positive|positiv(?:e|ity)?|\bpos\b|partial|focal|weak|diffuse|strong|\+|阳性"
NEGATIVE_PATTERN = r"negative|negativ(?:e|ity)?|\bneg\b|not\s+expressed|not\s+detected|无表达|阴性"


def marker_binary_from_text(text, marker):
    """Map each marker to positive=1 / negative=0 / missing=None.

    Unmentioned markers return None and must be skipped by the caller.
    Explicit negatives stay 0; only clear positive wording (or Ki-67 > 30%) is 1.
    """
    if not text or not isinstance(text, str):
        return None

    marker_pattern = MARKER_PATTERNS.get(marker)
    if marker_pattern is None:
        return None

    normalized = text.replace("＋", "+").replace("－", "-")
    match = re.search(marker_pattern, normalized, re.IGNORECASE)
    if not match:
        return None

    start = match.start()
    end_match = re.search(r"[;\n。，]", normalized[start:])
    end = start + end_match.start() if end_match else len(normalized)
    phrase = normalized[start:end]

    if re.search(NEGATIVE_PATTERN, phrase, re.IGNORECASE):
        return 0

    if marker == "Ki-67":
        percentage = re.search(r"(?:>|≥|>=)?\s*(\d+(?:\.\d+)?)\s*%", phrase)
        if percentage and float(percentage.group(1)) > 30:
            return 1

    if re.search(POSITIVE_PATTERN, phrase, re.IGNORECASE):
        return 1

    return 0


def calc_bleu_scores(references, candidates):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    bleu1s, bleu2s, bleu3s, bleu4s = [], [], [], []
    smoothie = SmoothingFunction().method1
    for ref, cand in zip(references, candidates):
        ref_tokens = [ref.split()]
        cand_tokens = cand.split()
        bleu1s.append(sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        bleu2s.append(sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        bleu3s.append(sentence_bleu(ref_tokens, cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        bleu4s.append(sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
    return np.mean(bleu1s), np.mean(bleu2s), np.mean(bleu3s), np.mean(bleu4s)


def calc_rouge_l(references, candidates):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, cand in zip(references, candidates):
        score = scorer.score(ref, cand)
        scores.append(score["rougeL"].fmeasure)
    return np.mean(scores)


async def ask_llm(client, message_content, model=DEEPSEEK_MODEL, max_retries=3):
    for attempt in range(max_retries):
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message_content}],
                timeout=60,
            )
            return (chat.choices[0].message.content or "").strip() or "none"
        except Exception as e:
            wait_seconds = 5 * (attempt + 1)
            print(f"[Warning] DeepSeek request failed: {e}. Retry {attempt + 1}/{max_retries}, waiting {wait_seconds}s...")
            await asyncio.sleep(wait_seconds)
    return "none"


async def extract_ihc_with_llm(client, text, prompt, model=DEEPSEEK_MODEL, max_retries=3):
    text = (text or "").strip()
    if not text:
        return "none"
    if len(text) > 12000:
        text = text[:12000]

    message_content = prompt + "\nHere is the report:\n" + text

    for attempt in range(max_retries):
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message_content}],
                timeout=60,
            )
            return (chat.choices[0].message.content or "").strip() or "none"
        except Exception as e:
            wait_seconds = 5 * (attempt + 1)
            print(f"[Warning] DeepSeek request failed: {e}. Retry {attempt + 1}/{max_retries}, waiting {wait_seconds}s...")
            await asyncio.sleep(wait_seconds)

    return "none"


async def main():
    api_key = DEEPSEEK_API_KEY
    if not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is missing.")

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL or "https://api.deepseek.com")

    jsonl_path = JSONL_PATH
    output_xlsx = OUTPUT_XLSX

    cache = OrderedDict()
    entries = list(read_jsonl(jsonl_path))
    if TEST_CASE_LIMIT is not None and TEST_CASE_LIMIT > 0:
        entries = entries[:TEST_CASE_LIMIT]

    rows = []
    marker_eval_rows = []

    for entry in tqdm(entries, desc=f"Processing cases ({'first ' + str(TEST_CASE_LIMIT) if TEST_CASE_LIMIT else 'all'})"):
        case_id = extract_case_id(entry)
        gt = entry.get("labels", "") or ""
        gen = entry.get("response", "") or ""

        gt_key = f"GT::{gt}"
        gen_key = f"GEN::{gen}"

        if gt_key not in cache:
            cache[gt_key] = await extract_ihc_with_llm(client, gt, PROMPT)
        if gen_key not in cache:
            cache[gen_key] = await extract_ihc_with_llm(client, gen, PROMPT)

        gt_ihc = cache[gt_key]
        gen_ihc = cache[gen_key]

        structure_request = STRUCTURE_GENERATION_PROMPT + "\nGeneration Result:\n" + gen
        structured_generation_raw = await ask_llm(client, structure_request)
        if structured_generation_raw == "none":
            structured_generation_raw = gen

        correction_request = (
            REPLACE_PROMPT
            + "\nGround Truth:\n"
            + gt
            + "\nStructured Generation Result:\n"
            + structured_generation_raw
        )
        structured_generation = await ask_llm(client, correction_request)
        if structured_generation == "none":
            structured_generation = structured_generation_raw

        structured_gen_key = f"GEN_STRUCTURED::{structured_generation}"
        if structured_gen_key not in cache:
            cache[structured_gen_key] = await extract_ihc_with_llm(client, structured_generation, PROMPT)
        structured_gen_ihc = cache[structured_gen_key]

        rows.append(
            {
                "ID": case_id,
                "Ground Truth": gt,
                "Ground Truth IHC": gt_ihc,
                "IHC result": gt_ihc,
                "Generation Result": structured_generation,
                "Generation Result IHC": gen_ihc,
                "Structured Generated IHC result": structured_gen_ihc,
            }
        )

        for marker in MARKERS:
            gt_bin = marker_binary_from_text(gt_ihc, marker)
            if gt_bin is None:
                continue
            gen_bin = marker_binary_from_text(structured_gen_ihc, marker)
            if gen_bin is None:
                gen_bin = 0
            marker_eval_rows.append(
                {
                    "ID": case_id,
                    "Marker": marker,
                    "Ground Truth": gt_bin,
                    "Prediction": gen_bin,
                }
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Ground Truth",
            "Ground Truth IHC",
            "IHC result",
            "Generation Result",
            "Generation Result IHC",
            "Structured Generated IHC result",
        ],
    )

    bleu1, bleu2, bleu3, bleu4 = calc_bleu_scores(
        [row["Ground Truth"] for row in rows],
        [row["Generation Result"] for row in rows],
    )
    rouge_l = calc_rouge_l(
        [row["Ground Truth"] for row in rows],
        [row["Generation Result"] for row in rows],
    )
    metrics_df = pd.DataFrame(
        [
            {
                "BLEU_1": bleu1,
                "BLEU_2": bleu2,
                "BLEU_3": bleu3,
                "BLEU_4": bleu4,
                "ROUGE_L": rouge_l,
            }
        ]
    )

    marker_eval_df = pd.DataFrame(marker_eval_rows)
    if marker_eval_df.empty:
        marker_eval_df = pd.DataFrame(columns=["ID", "Marker", "Ground Truth", "Prediction"])
    marker_eval_df["Ground Truth"] = marker_eval_df["Ground Truth"].fillna(0).astype(int)
    marker_eval_df["Prediction"] = marker_eval_df["Prediction"].fillna(0).astype(int)
    y_true = marker_eval_df["Ground Truth"].tolist()
    y_pred = marker_eval_df["Prediction"].tolist()

    if y_true:
        marker_precision = precision_score(y_true, y_pred, zero_division=0)
        marker_recall = recall_score(y_true, y_pred, zero_division=0)
        marker_accuracy = accuracy_score(y_true, y_pred)
    else:
        marker_precision = marker_recall = marker_accuracy = np.nan

    total_marker_df = pd.DataFrame(
        [
            {
                "Precision": marker_precision,
                "Recall": marker_recall,
                "Accuracy": marker_accuracy,
                "N (GT positive)": int(sum(y_true)) if y_true else 0,
                "N (evaluated marker-case pairs)": len(y_true),
            }
        ]
    )

    marker_rows = []
    for marker in MARKERS:
        marker_df = marker_eval_df[marker_eval_df["Marker"] == marker]
        gt_labels = marker_df["Ground Truth"].astype(int).tolist()
        gen_labels = marker_df["Prediction"].astype(int).tolist()
        gt_positive_count = int(sum(gt_labels)) if gt_labels else 0
        tn = sum(gt == 0 and pred == 0 for gt, pred in zip(gt_labels, gen_labels))
        fp = sum(gt == 0 and pred == 1 for gt, pred in zip(gt_labels, gen_labels))
        fn = sum(gt == 1 and pred == 0 for gt, pred in zip(gt_labels, gen_labels))
        tp = sum(gt == 1 and pred == 1 for gt, pred in zip(gt_labels, gen_labels))
        marker_rows.append(
            {
                "Marker": marker,
                "Precision": precision_score(gt_labels, gen_labels, zero_division=0) if gt_labels else np.nan,
                "Recall": recall_score(gt_labels, gen_labels, zero_division=0) if gt_labels else np.nan,
                "Accuracy": accuracy_score(gt_labels, gen_labels) if gt_labels else np.nan,
                "N (GT positive)": gt_positive_count,
                "N (evaluated cases)": len(gt_labels),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
            }
        )

    marker_stats_df = pd.DataFrame(marker_rows)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Case Detail")
        metrics_df.to_excel(writer, index=False, sheet_name="Text Metrics")
        total_marker_df.to_excel(writer, index=False, sheet_name="Marker Summary")
        marker_stats_df.to_excel(writer, index=False, sheet_name="Marker Stats")
        marker_eval_df.to_excel(writer, index=False, sheet_name="Marker Case Detail")

    print(f"Saved to {output_xlsx}")


if __name__ == "__main__":
    asyncio.run(main())
