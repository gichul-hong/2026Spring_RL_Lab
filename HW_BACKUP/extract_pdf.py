import fitz  # pymupdf

pdf_path = "C:/hong/2026Spring_RL_Lab/HW/[2026_DS_RL]_HW2.pdf"
out_path = "C:/hong/2026Spring_RL_Lab/HW/HW2_raw.md"

doc = fitz.open(pdf_path)
pages = []
for i, page in enumerate(doc):
    text = page.get_text("text")
    pages.append(f"## Page {i+1}\n\n{text}")

raw = "\n\n---\n\n".join(pages)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("# [2026_DS_RL] HW2 - Raw Extracted Text\n\n")
    f.write(raw)

print(f"총 {len(doc)} 페이지 추출 완료")
print(f"총 글자 수: {len(raw):,}")
doc.close()
