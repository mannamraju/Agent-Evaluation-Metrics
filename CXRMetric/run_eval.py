import json
import numpy as np
import os
import re
import pandas as pd
import pickle
import torch

# Safe imports with fallbacks
try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    print("Warning: bert_score not available. BERTScore will be skipped.")

try:
    from fast_bleu import BLEU
    FAST_BLEU_AVAILABLE = True
except ImportError:
    FAST_BLEU_AVAILABLE = False
    try:
        from nltk.translate.bleu_score import sentence_bleu
        import nltk
        NLTK_BLEU_AVAILABLE = True
    except ImportError:
        NLTK_BLEU_AVAILABLE = False
        print("Warning: Neither fast_bleu nor nltk available. BLEU scores will use fallback implementation.")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_recall_fscore_support

import config
import importlib
RUN_RADGRAPH_AVAILABLE = True
try:
    # Delay importing radgraph_evaluate_model until it's needed; if the
    # radgraph_inference code/folder is missing this avoids immediate ImportError.
    importlib.import_module('CXRMetric.radgraph_evaluate_model')
except Exception:
    RUN_RADGRAPH_AVAILABLE = False
    # run_radgraph will be imported lazily where needed

"""Computes 4 individual metrics and a composite metric on radiology reports."""


CHEXBERT_PATH = config.CHEXBERT_PATH
RADGRAPH_PATH = config.RADGRAPH_PATH

NORMALIZER_PATH = "CXRMetric/normalizer.pkl"
COMPOSITE_METRIC_V0_PATH = "CXRMetric/composite_metric_model.pkl"
COMPOSITE_METRIC_V1_PATH = "CXRMetric/radcliq-v1.pkl"

REPORT_COL_NAME = "report"
STUDY_ID_COL_NAME = "study_id"
COLS = ["radgraph_combined", "bertscore", "semb_score", "bleu_score"]

cache_path = "cache/"
pred_embed_path = os.path.join(cache_path, "pred_embeddings.pt")
gt_embed_path = os.path.join(cache_path, "gt_embeddings.pt")
weights = {"bigram": (1/2., 1/2.)}
composite_metric_col_v0 = "RadCliQ-v0"
composite_metric_col_v1 = "RadCliQ-v1"


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """
    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


def prep_reports(reports):
    """Preprocesses reports"""
    return [list(filter(
        lambda val: val !=  "", str(elem)\
            .lower().replace(".", " .").split(" "))) for elem in reports]

def compute_bleu_score_safe(reference_tokens, candidate_tokens, n_grams=2):
    """Safely compute BLEU score with fallbacks for missing packages."""
    if FAST_BLEU_AVAILABLE:
        # Use fast-bleu implementation
        if n_grams == 2:
            weights_dict = {"bigram": (1/2., 1/2.)}
            bleu = BLEU([reference_tokens], weights_dict)
            score = bleu.get_score([candidate_tokens])["bigram"]
        else:  # n_grams == 4
            weights_dict = {"bleu4": (1/4., 1/4., 1/4., 1/4.)}
            bleu = BLEU([reference_tokens], weights_dict)
            score = bleu.get_score([candidate_tokens])["bleu4"]
        return score[0] if isinstance(score, list) else score
    
    elif NLTK_BLEU_AVAILABLE:
        # Use NLTK implementation
        if n_grams == 2:
            weights = (0.5, 0.5, 0.0, 0.0)
        else:  # n_grams == 4
            weights = (0.25, 0.25, 0.25, 0.25)
        
        try:
            score = sentence_bleu([reference_tokens], candidate_tokens, weights=weights)
            return score
        except ZeroDivisionError:
            return 0.0
    
    else:
        # Fallback: simple n-gram precision
        if not candidate_tokens or not reference_tokens:
            return 0.0
        
        # Simple n-gram overlap for fallback
        ref_ngrams = {}
        cand_ngrams = {}
        
        for n in range(1, n_grams + 1):
            # Reference n-grams
            for i in range(len(reference_tokens) - n + 1):
                ngram = tuple(reference_tokens[i:i + n])
                ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
            
            # Candidate n-grams  
            for i in range(len(candidate_tokens) - n + 1):
                ngram = tuple(candidate_tokens[i:i + n])
                cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
        
        # Count matches
        matches = 0
        total = 0
        for ngram, count in cand_ngrams.items():
            total += count
            if ngram in ref_ngrams:
                matches += min(count, ref_ngrams[ngram])
        
        return matches / total if total > 0 else 0.0

def add_bleu_col(gt_df, pred_df):
    """Computes BLEU-2 and adds scores as a column to prediction df."""
    pred_df["bleu_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row[REPORT_COL_NAME]])[0]
        pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
        predicted_report = \
            prep_reports([pred_row[REPORT_COL_NAME].values[0]])[0]
        if len(pred_row) == 1:
            try:
                score = compute_bleu_score_safe(gt_report, predicted_report, n_grams=2)
                _index = pred_df.index[
                    pred_df[STUDY_ID_COL_NAME]==row[STUDY_ID_COL_NAME]].tolist()[0]
                pred_df.at[_index, "bleu_score"] = score
            except Exception as e:
                print(f"Warning: BLEU-2 computation failed for sample {i}: {e}")
                _index = pred_df.index[
                    pred_df[STUDY_ID_COL_NAME]==row[STUDY_ID_COL_NAME]].tolist()[0]
                pred_df.at[_index, "bleu_score"] = 0.0
    return pred_df

def add_bertscore_col(gt_df, pred_df, use_idf):
    """Computes BERTScore and adds scores as a column to prediction df."""
    if not BERTSCORE_AVAILABLE:
        print("Warning: BERTScore not available. Adding zero scores.")
        pred_df["bertscore"] = [0.0] * len(pred_df)
        return pred_df
        
    try:
        test_reports = gt_df[REPORT_COL_NAME].tolist()
        test_reports = [re.sub(r' +', ' ', test) for test in test_reports]
        method_reports = pred_df[REPORT_COL_NAME].tolist()
        method_reports = [re.sub(r' +', ' ', report) for report in method_reports]

        scorer = BERTScorer(
            model_type="distilroberta-base",
            batch_size=256,
            lang="en",
            rescale_with_baseline=True,
            idf=use_idf,
            idf_sents=test_reports)
        _, _, f1 = scorer.score(method_reports, test_reports)
        pred_df["bertscore"] = f1
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}")
        pred_df["bertscore"] = [0.0] * len(pred_df)
    return pred_df

def add_semb_col(pred_df, semb_path, gt_path):
    """Computes s_emb and adds scores as a column to prediction df."""
    label_embeds = torch.load(gt_path)
    pred_embeds = torch.load(semb_path)
    list_label_embeds = []
    list_pred_embeds = []
    for data_idx in sorted(label_embeds.keys()):
        list_label_embeds.append(label_embeds[data_idx])
        list_pred_embeds.append(pred_embeds[data_idx])
    np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
    np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
    scores = []
    for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
        sim_scores = (label * pred).sum() / (
            np.linalg.norm(label) * np.linalg.norm(pred))
        scores.append(sim_scores)
    pred_df["semb_score"] = scores
    return pred_df

def add_radgraph_col(pred_df, entities_path, relations_path):
    """Computes RadGraph F1 and adds scores as a column to prediction df."""
    from CXRMetric.radgraph_evaluate_model import run_radgraph
    study_id_to_radgraph = {}
    with open(entities_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            try:
                study_id_to_radgraph[int(study_id)] = float(f1)
            except:
                continue
    with open(relations_path, "r") as f:
        scores = json.load(f)
        for study_id, (f1, _, _) in scores.items():
            try:
                study_id_to_radgraph[int(study_id)] += float(f1)
                study_id_to_radgraph[int(study_id)] /= float(2)
            except:
                continue
    radgraph_scores = []
    count = 0
    for i, row in pred_df.iterrows():
        radgraph_scores.append(study_id_to_radgraph[int(row[STUDY_ID_COL_NAME])])
    pred_df["radgraph_combined"] = radgraph_scores
    return pred_df

# --- New evaluation helpers: ROUGE-L, BLEU-4, CheXpert micro-F1, and box IoU metrics ---

def _lcs_length(a, b):
    """Compute length of longest common subsequence between token lists a and b."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def rouge_l_score(ref, cand, beta=1.2):
    """Compute ROUGE-L F1 between two strings (reference and candidate).
    Uses token-based LCS and beta=1.2 as in common ROUGE implementations.
    Returns a float in [0, 1]."""
    ref_tokens = str(ref).split()
    cand_tokens = str(cand).split()
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0
    lcs_len = _lcs_length(ref_tokens, cand_tokens)
    if lcs_len == 0:
        return 0.0
    prec = lcs_len / len(cand_tokens)
    rec = lcs_len / len(ref_tokens)
    beta_sq = beta * beta
    denom = rec + beta_sq * prec
    if denom == 0:
        return 0.0
    score = ((1 + beta_sq) * prec * rec) / denom
    return float(score)


def add_rouge_col(gt_df, pred_df):
    """Computes ROUGE-L per-report and adds column `rouge_l` to pred_df."""
    pred_df["rouge_l"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = re.sub(r' +', ' ', str(row[REPORT_COL_NAME]).strip())
        pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
        if len(pred_row) == 1:
            predicted_report = re.sub(r' +', ' ', str(pred_row[REPORT_COL_NAME].values[0]).strip())
            score = rouge_l_score(gt_report, predicted_report)
            _index = pred_df.index[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]].tolist()[0]
            pred_df.at[_index, "rouge_l"] = score
    return pred_df


def add_bleu4_col(gt_df, pred_df):
    """Computes BLEU-4 (uniform 1/4 weights) and adds column `bleu4_score` to pred_df."""
    pred_df["bleu4_score"] = [0.0] * len(pred_df)
    for i, row in gt_df.iterrows():
        gt_report = prep_reports([row[REPORT_COL_NAME]])[0]
        pred_row = pred_df[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]]
        if len(pred_row) == 1:
            predicted_report = prep_reports([pred_row[REPORT_COL_NAME].values[0]])[0]
            try:
                score = compute_bleu_score_safe(gt_report, predicted_report, n_grams=4)
                _index = pred_df.index[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]].tolist()[0]
                pred_df.at[_index, "bleu4_score"] = score
            except Exception as e:
                print(f"Warning: BLEU-4 computation failed for sample {i}: {e}")
                _index = pred_df.index[pred_df[STUDY_ID_COL_NAME] == row[STUDY_ID_COL_NAME]].tolist()[0]
                pred_df.at[_index, "bleu4_score"] = 0.0
    return pred_df


def _run_chexbert_labeler(checkpoint_path, csv_path, out_path):
    """Run the CheXbert labeling script if available; otherwise warn and return None.

    The CheXbert code and model files were removed from this repository. When
    `checkpoint_path` is not provided or the CheXbert code is missing, this
    function will print a warning and return None so downstream code handles
    CheXpert evaluation gracefully.
    """
    os.makedirs(out_path, exist_ok=True)
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Warning: CheXbert model not available at {checkpoint_path}. Skipping CheXbert labeling.")
        return None
    chexbert_script = os.path.join("CXRMetric", "CheXbert", "src", "label.py")
    if not os.path.exists(chexbert_script):
        print("Warning: CheXbert labeling code not present in the repository. Skipping CheXbert labeling.")
        return None
    # If CheXbert is restored, fallback to the original behavior
    cmd = f"python {chexbert_script} -c {checkpoint_path} -d {csv_path} -o {out_path}"
    print("Running CheXbert labeler:", cmd)
    os.system(cmd)
    labeled_csv = os.path.join(out_path, "labeled_reports.csv")
    if os.path.exists(labeled_csv):
        return labeled_csv
    else:
        return None


def _parse_chex_labels(labeled_csv):
    """Parse a labeled_reports.csv produced by the CheXbert labeler and return a binary array (n_samples, n_conditions).
    The labeling script maps: positive -> 1, negative -> 0, uncertain -> -1, blank -> NaN. We treat positive and uncertain as positive (1) by default.
    """
    df = pd.read_csv(labeled_csv)
    cols = [c for c in df.columns if c != REPORT_COL_NAME]
    arr = np.zeros((len(df), len(cols)), dtype=int)
    for i in range(len(df)):
        for j, c in enumerate(cols):
            val = df[c].iloc[i]
            try:
                if pd.isna(val):
                    arr[i, j] = 0
                else:
                    num = float(val)
                    # positive (1) or uncertain (-1) treated as presence
                    if num == 1 or num == -1:
                        arr[i, j] = 1
                    else:
                        arr[i, j] = 0
            except Exception:
                # fallback: if the cell contains strings like 'Positive'
                s = str(val).lower()
                if 'pos' in s or 'uncer' in s:
                    arr[i, j] = 1
                else:
                    arr[i, j] = 0
    return arr, cols


def compute_chexpert_micro_f1(gt_labeled_csv, pred_labeled_csv):
    """Compute CheXpert micro-F1 and per-condition precision/recall/f1 from two labeled CSVs."""
    y_true, cols = _parse_chex_labels(gt_labeled_csv)
    y_pred, _ = _parse_chex_labels(pred_labeled_csv)
    # ensure same shape
    assert y_true.shape == y_pred.shape
    # micro F1 across all conditions and samples
    micro_f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='micro')
    per_cond = {}
    for j, c in enumerate(cols):
        p, r, f, _ = precision_recall_fscore_support(y_true[:, j], y_pred[:, j], average='binary', zero_division=0)
        per_cond[c] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}
    return {'micro_f1': float(micro_f1), 'per_condition': per_cond}


def _normalize_box(box):
    """Normalize box to [x1, y1, x2, y2]. Accepts dicts or lists.
    Supports dict formats with keys (x,y,w,h) or (x1,y1,x2,y2) or lists.
    """
    if isinstance(box, dict):
        if all(k in box for k in ('x', 'y', 'w', 'h')):
            x1 = float(box['x'])
            y1 = float(box['y'])
            x2 = x1 + float(box['w'])
            y2 = y1 + float(box['h'])
            return [x1, y1, x2, y2]
        if all(k in box for k in ('x1', 'y1', 'x2', 'y2')):
            return [float(box['x1']), float(box['y1']), float(box['x2']), float(box['y2'])]
        # fallback try common keys
        keys = list(box.keys())
        vals = [float(box[k]) for k in keys[:4]]
        return vals
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        # ambiguous: assume [x1,y1,x2,y2] if x2 > x1
        x1, y1, x2, y2 = map(float, box)
        if x2 >= x1 and y2 >= y1:
            return [x1, y1, x2, y2]
        # otherwise treat as [x,y,w,h]
        return [x1, y1, x1 + x2, y1 + y2]
    else:
        raise ValueError(f"Unsupported box format: {box}")


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def compute_box_precision_recall(gt_boxes_col, pred_boxes_col, iou_thresh=0.5):
    """Compute dataset-level box precision and recall.
    gt_boxes_col and pred_boxes_col are pandas Series where each entry is a JSON string or Python list of boxes.
    Returns (precision, recall).
    """
    tp = 0
    total_pred = 0
    total_gt = 0
    for gt_entry, pred_entry in zip(gt_boxes_col.tolist(), pred_boxes_col.tolist()):
        try:
            if isinstance(gt_entry, str):
                gboxes = json.loads(gt_entry)
            else:
                gboxes = gt_entry
        except Exception:
            gboxes = gt_entry
        try:
            if isinstance(pred_entry, str):
                pboxes = json.loads(pred_entry)
            else:
                pboxes = pred_entry
        except Exception:
            pboxes = pred_entry

        if gboxes is None:
            gboxes = []
        if pboxes is None:
            pboxes = []
        # normalize
        try:
            gboxes_norm = [_normalize_box(b) for b in gboxes]
            pboxes_norm = [_normalize_box(b) for b in pboxes]
        except Exception:
            # skip badly formatted entries
            continue
        total_gt += len(gboxes_norm)
        total_pred += len(pboxes_norm)
        matched_gt = set()
        for pb in pboxes_norm:
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(gboxes_norm):
                if gi in matched_gt:
                    continue
                iouv = _iou(pb, gb)
                if iouv > best_iou:
                    best_iou = iouv
                    best_idx = gi
            if best_iou >= iou_thresh and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    return precision, recall


def calc_metric(gt_csv, pred_csv, out_csv, use_idf): # TODO: support single metrics at a time
    """Computes metrics and composite metric scores; returns a summary dict."""
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    cache_gt_csv = os.path.join(
        os.path.dirname(gt_csv), f"cache_{os.path.basename(gt_csv)}")
    cache_pred_csv = os.path.join(
        os.path.dirname(pred_csv), f"cache_{os.path.basename(pred_csv)}")
    gt = pd.read_csv(gt_csv)\
        .sort_values(by=[STUDY_ID_COL_NAME]).reset_index(drop=True)
    pred = pd.read_csv(pred_csv)\
        .sort_values(by=[STUDY_ID_COL_NAME]).reset_index(drop=True)

    # Keep intersection of study IDs
    gt_study_ids = set(gt[STUDY_ID_COL_NAME])
    pred_study_ids = set(pred[STUDY_ID_COL_NAME])
    shared_study_ids = gt_study_ids.intersection(pred_study_ids)
    print(f"Number of shared study IDs: {len(shared_study_ids)}")
    gt = gt.loc[gt[STUDY_ID_COL_NAME].isin(shared_study_ids)].reset_index()
    pred = pred.loc[pred[STUDY_ID_COL_NAME].isin(shared_study_ids)].reset_index()

    gt.to_csv(cache_gt_csv, index=False)
    pred.to_csv(cache_pred_csv, index=False)

    # check that length and study IDs are the same
    assert len(gt) == len(pred)
    assert (REPORT_COL_NAME in gt.columns) and (REPORT_COL_NAME in pred.columns)
    assert (gt[STUDY_ID_COL_NAME].equals(pred[STUDY_ID_COL_NAME]))

    # BLEU-2 (existing)
    pred = add_bleu_col(gt, pred)

    # BLEU-4 (new)
    pred = add_bleu4_col(gt, pred)

    # ROUGE-L (new)
    pred = add_rouge_col(gt, pred)

    # BERTScore
    pred = add_bertscore_col(gt, pred, use_idf)

    # Semantic embedding (CheXbert) — run only if model and code are present
    os.makedirs(cache_path, exist_ok=True)
    pred['semb_score'] = [0.0] * len(pred)
    chexbert_script = os.path.join('CXRMetric', 'CheXbert', 'src', 'encode.py')
    if CHEXBERT_PATH and os.path.exists(CHEXBERT_PATH) and os.path.exists(chexbert_script):
        try:
            os.system(f"python {chexbert_script} -c {CHEXBERT_PATH} -d {cache_pred_csv} -o {pred_embed_path}")
            os.system(f"python {chexbert_script} -c {CHEXBERT_PATH} -d {cache_gt_csv} -o {gt_embed_path}")
            pred = add_semb_col(pred, pred_embed_path, gt_embed_path)
        except Exception as e:
            print(f"Warning: semantic embedding computation failed: {e}")
            pred['semb_score'] = [0.0] * len(pred)
    else:
        print("Warning: CheXbert model or code not available — semantic embedding scores set to 0.0")

    # RadGraph inference — run only if RadGraph model/code are present
    entities_path = os.path.join(cache_path, "entities_cache.json")
    relations_path = os.path.join(cache_path, "relations_cache.json")
    if RADGRAPH_PATH and os.path.exists(RADGRAPH_PATH):
        if RUN_RADGRAPH_AVAILABLE:
            try:
                from CXRMetric.radgraph_evaluate_model import run_radgraph
                run_radgraph(cache_gt_csv, cache_pred_csv, cache_path, RADGRAPH_PATH,
                             entities_path, relations_path)
                pred = add_radgraph_col(pred, entities_path, relations_path)
            except Exception as e:
                print(f"Warning: RadGraph inference failed: {e}")
                pred['radgraph_combined'] = [0.0] * len(pred)
        else:
            print("Warning: RadGraph code not available in repository. Skipping RadGraph inference.")
            pred['radgraph_combined'] = [0.0] * len(pred)
    else:
        print("Warning: RadGraph model not available — radgraph_combined set to 0.0")
        pred['radgraph_combined'] = [0.0] * len(pred)

    # compute composite metric: RadCliQ-v0 and v1 — only if required columns are present
    if all(c in pred.columns for c in COLS):
        try:
            with open(COMPOSITE_METRIC_V0_PATH, "rb") as f:
                composite_metric_v0_model = pickle.load(f)
            with open(NORMALIZER_PATH, "rb") as f:
                normalizer = pickle.load(f)
            # normalize
            input_data = np.array(pred[COLS])
            norm_input_data = normalizer.transform(input_data)
            # generate new col
            radcliq_v0_scores = composite_metric_v0_model.predict(norm_input_data)
            pred[composite_metric_col_v0] = radcliq_v0_scores

            # compute composite metric: RadCliQ-v1
            with open(COMPOSITE_METRIC_V1_PATH, "rb") as f:
                composite_metric_v1_model = pickle.load(f)
            input_data = np.array(pred[COLS])
            radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
            pred[composite_metric_col_v1] = radcliq_v1_scores
        except Exception as e:
            print(f"Warning: composite metric computation failed: {e}")
            pred[composite_metric_col_v0] = [0.0] * len(pred)
            pred[composite_metric_col_v1] = [0.0] * len(pred)
    else:
        print("Warning: missing metric columns for composite computation; skipping RadCliQ metrics")
        pred[composite_metric_col_v0] = [0.0] * len(pred)
        pred[composite_metric_col_v1] = [0.0] * len(pred)

    # --- Compute CheXpert micro-F1 (uses CheXbert labeler) and box metrics if available ---
    summary = {}

    # run CheXbert labeler on cached CSVs
    pred_labeled_path = os.path.join(cache_path, "pred_labels")
    gt_labeled_path = os.path.join(cache_path, "gt_labels")
    pred_label_csv = _run_chexbert_labeler(CHEXBERT_PATH, cache_pred_csv, pred_labeled_path)
    gt_label_csv = _run_chexbert_labeler(CHEXBERT_PATH, cache_gt_csv, gt_labeled_path)
    if pred_label_csv and gt_label_csv:
        chexpert_metrics = compute_chexpert_micro_f1(gt_label_csv, pred_label_csv)
        summary['chexpert'] = chexpert_metrics

    # compute box IoU precision/recall if box columns are available in both files
    if 'boxes' in gt.columns and 'boxes' in pred.columns:
        box_precision, box_recall = compute_box_precision_recall(gt['boxes'], pred['boxes'])
        summary['box_precision'] = float(box_precision)
        summary['box_recall'] = float(box_recall)

    # aggregate dataset-level means for common metrics
    mean_metrics = {}
    metric_cols = ['bleu_score', 'bleu4_score', 'rouge_l', 'bertscore', 'semb_score', 'radgraph_combined', composite_metric_col_v0, composite_metric_col_v1]
    for col in metric_cols:
        if col in pred.columns:
            mean_metrics[col] = float(np.nanmean(pred[col].values))
    summary['mean_metrics'] = mean_metrics

    # save results in the out folder and write a summary JSON
    pred.to_csv(out_csv, index=False)
    summary_path = out_csv + ".summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary
