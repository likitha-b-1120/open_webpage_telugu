# 📝 Telugu OCR Post-Processing Tool

A web-based tool for post-processing the OCR output of Telugu text using frequency-based dictionary correction.

🔗 **Live Demo**: [Click here to try the tool](https://open-webpage-telugu.onrender.com)

---

## 📁 Inputs Required

1. **OCR Output File** (`.txt`)
   - Each line contains a single OCR-predicted Telugu word.

2. **Dictionary File** (`.tsv`)
   - Format: `word<TAB>frequency`
   - Example:
     ```
     ఉదాహరణ	1089
     తెలుగు	906
     ```

---

## 🚀 How to Use

1. Go to the [Live Demo](https://open-webpage-telugu.onrender.com).
2. Upload:
   - A `.txt` file containing OCR-predicted words (one per line).
   - A `.tsv` dictionary file with word-frequency pairs.
3. The tool:
   - Validates each OCR word against the dictionary.
   - Suggests corrections using edit distance.
   - Automatically picks the **most frequent valid correction**.
4. Results are displayed page-by-page with:
   - OCR word
   - Suggested correction
   - Frequency used for decision
5. Final corrected list
