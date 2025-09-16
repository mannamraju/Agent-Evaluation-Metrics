# Model checkpoints
# CheXbert and RadGraph models were removed from the repository.
# If you need these metrics again, set these to the appropriate checkpoint
# paths (local path or remote storage location). For now they are disabled.
CHEXBERT_PATH = None
RADGRAPH_PATH = None

# Report paths
GT_REPORTS = "reports/gt_reports.csv"
PREDICTED_REPORTS = "reports/predicted_reports.csv"
OUT_FILE = "report_scores.csv"

# Whether to use inverse document frequency (idf) for BERTScore
USE_IDF = False
