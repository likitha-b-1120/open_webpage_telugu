from flask import Flask, render_template_string, request, send_file
import os
import editdistance
import re
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

telugu_pattern = re.compile(r'^[\u0C00-\u0C7F]+$')

def is_telugu_word(word):
    return bool(telugu_pattern.fullmatch(word))

def read_data(file_path):
    gt_list, pred_list, prob_list = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            gt, pred, prob = parts
            gt_list.append(gt)
            pred_list.append(pred)
            prob_list.append(float(prob))
    return gt_list, pred_list, prob_list

def read_dictionary(dict_path):
    df = pd.read_csv(dict_path, sep="\t", header=None, names=["word", "frequency"])
    df = df[df["word"].apply(is_telugu_word)]
    return dict(zip(df["word"], df["frequency"]))

def word_accuracy(gt_list, pred_list):
    return sum(g == p for g, p in zip(gt_list, pred_list)) / len(gt_list)

def char_accuracy(gt_list, pred_list):
    total_chars = sum(len(g) for g in gt_list)
    total_correct = sum(len(g) - editdistance.eval(g, p) for g, p in zip(gt_list, pred_list))
    return total_correct / total_chars

def post_process(pred_list, prob_list, dictionary, edit_dist_threshold=2, prob_threshold=0.90):
    corrected = []
    dict_words = list(dictionary.keys())

    for word, prob in zip(pred_list, prob_list):
        if not is_telugu_word(word) or word in dictionary or prob > prob_threshold:
            corrected.append(word)
        else:
            first_char = word[0]
            candidates = [w for w in dict_words if w.startswith(first_char)]
            if not candidates:
                corrected.append(word)
                continue
            distances = [(w, editdistance.eval(word, w)) for w in candidates]
            min_dist = min(d[1] for d in distances)
            tied = [w for w, d in distances if d == min_dist]
            if min_dist > edit_dist_threshold:
                corrected.append(word)
                continue
            best_candidate = max(tied, key=lambda w: dictionary.get(w, 0))
            corrected.append(best_candidate)
    return corrected

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Telugu OCR Auto-Correction Tool</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body class="bg-light">
            <div class="container py-5">
                <h2 class="mb-4 text-center text-primary">🤖 Telugu OCR Auto-Correction Tool</h2>
                <form method="POST" action="/process" enctype="multipart/form-data" class="bg-white p-4 rounded shadow">
                    <div class="mb-3">
                        <label class="form-label">📂 Input File (.txt)</label>
                        <input type="file" name="input_file" class="form-control" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">📖 Dictionary File (.tsv)</label>
                        <input type="file" name="dict_file" class="form-control" required>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">🎯 Probability Threshold</label>
                        <input type="text" name="prob_threshold" class="form-control" value="0.90">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">✂️ Edit Distance Threshold</label>
                        <input type="text" name="edit_dist_threshold" class="form-control" value="2">
                    </div>
                    <button type="submit" class="btn btn-primary w-100">Start Correction 🔧</button>
                </form>
            </div>
        </body>
        </html>
    ''')

@app.route('/process', methods=['POST'])
def process():
    input_file = request.files['input_file']
    dict_file = request.files['dict_file']
    prob_threshold = float(request.form.get('prob_threshold', 0.9))
    dist_threshold = int(request.form.get('edit_dist_threshold', 2))

    input_path = os.path.join(UPLOAD_FOLDER, secure_filename(input_file.filename))
    dict_path = os.path.join(UPLOAD_FOLDER, secure_filename(dict_file.filename))
    input_file.save(input_path)
    dict_file.save(dict_path)

    gt_list, pred_list, prob_list = read_data(input_path)
    dictionary = read_dictionary(dict_path)

    # Before
    wrr_before = word_accuracy(gt_list, pred_list)
    crr_before = char_accuracy(gt_list, pred_list)

    # Post-processing
    corrected_preds = post_process(pred_list, prob_list, dictionary, dist_threshold, prob_threshold)

    # After
    wrr_after = word_accuracy(gt_list, corrected_preds)
    crr_after = char_accuracy(gt_list, corrected_preds)

    # Write output
    output_path = os.path.join(OUTPUT_FOLDER, "auto_corrected_output.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("GT\tPred\tCorrected\tProb\tWRR_Correct\tPost_WRR_Correct\n")
        for gt, pred, corr, prob in zip(gt_list, pred_list, corrected_preds, prob_list):
            f.write(f"{gt}\t{pred}\t{corr}\t{prob:.4f}\t{int(gt==pred)}\t{int(gt==corr)}\n")
        f.write(f"\nWRR_Before\t{wrr_before:.4f}\n")
        f.write(f"CRR_Before\t{crr_before:.4f}\n")
        f.write(f"WRR_After\t{wrr_after:.4f}\n")
        f.write(f"CRR_After\t{crr_after:.4f}\n")

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    import os
    print("✅ Starting the Flask server...")  # Debug line
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
