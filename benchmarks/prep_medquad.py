#!/usr/bin/env python3
import os, sys, json, glob
from lxml import etree
from tqdm import tqdm

"""
Parse MedQuAD XML into a clean JSONL:
- fields: id, question, answer, url, type, focus
- skip the three MedlinePlus subsets with removed answers
"""

EXCLUDE_DIRS = {
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}

def text_or_none(x):
    return (x or "").strip() if isinstance(x, str) else None

def find_first(root, cand_tags):
    # Case- and namespace-insensitive
    cand = {t.lower() for t in cand_tags}
    for el in root.iter():
        tag = el.tag
        if isinstance(tag, str):
            name = tag.split("}")[-1].lower()
            if name in cand:
                if el.text and el.text.strip():
                    return el.text.strip()
    return None

def parse_xml(xml_path):
    root = etree.parse(xml_path).getroot()
    q = find_first(root, ["Question", "question", "q_text", "q"])
    a = find_first(root, ["Answer", "answer", "a_text", "a", 
"AnswerText"])
    u = find_first(root, ["URL", "url", "link"])
    t = find_first(root, ["Type", "type", "QuestionType"])
    f = find_first(root, ["Focus", "focus"])
    return q, a, u, t, f

def main():
    if len(sys.argv) != 3:
        print("Usage: prep_medquad.py <input_medquad_dir> <out_jsonl>")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_path = sys.argv[2]
    n_total = n_kept = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for xml_path in tqdm(sorted(glob.glob(os.path.join(in_dir, "*", 
"*.xml")))):
            parts = set(os.path.normpath(xml_path).split(os.sep))
            if parts & EXCLUDE_DIRS:
                continue
            n_total += 1
            try:
                q, a, u, t, f = parse_xml(xml_path)
                if q and a:
                    _id = os.path.splitext(os.path.basename(xml_path))[0]
                    rec = {
                        "id": _id, "question": q, "answer": a,
                        "url": text_or_none(u), "type": text_or_none(t),
                        "focus": text_or_none(f)
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_kept += 1
            except Exception as e:
                # silently skip bad files
                pass

    print(f"Done. Parsed={n_total}, kept(with answers)={n_kept}, 
out={out_path}")

if __name__ == "__main__":
    main()

