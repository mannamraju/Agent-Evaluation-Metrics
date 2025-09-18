# BERTScore run summary (20250918T130653)

## Per-sample BERTScore Results

| study_id | precision | recall | f1 | good_range | why |
|---|---:|---:|---:|---|---|
| 1 | 0.6945 | 0.5333 | 0.6132 | Moderate (0.55-0.65) | Precision > Recall — predictions are relatively precise but miss some reference content, producing a moderate F1. |
| 2 | 0.7433 | 0.5784 | 0.6601 | Good (>=0.65) | Precision > Recall — strong precision indicates accurate wording where present; good semantic overlap overall. |
| 3 | 0.6710 | 0.5102 | 0.5900 | Moderate (0.55-0.65) | Precision > Recall — moderate overlap with some missing details in predictions. |
| 4 | 0.4701 | 0.4741 | 0.4725 | Poor (<0.55) | Both precision and recall are low — predictions lack key content or use substantially different wording, reducing overlap. |
| 5 | 0.4900 | 0.5225 | 0.5066 | Poor (<0.55) | Low F1 with recall slightly higher — predictions include some content but are imprecise or contain extra/irrelevant wording. |

## Summary statistics

```json
{
  "bertscore": {
    "mean": 0.5684863328933716,
    "std": 0.06913888454437256,
    "min": 0.47252577543258667,
    "max": 0.6601134538650513,
    "median": 0.589994490146637
  },
  "bertscore_precision": {
    "mean": 0.6137832403182983,
    "std": 0.11183291673660278,
    "min": 0.4701065123081207,
    "max": 0.7432998418807983,
    "median": 0.6710258722305298
  },
  "bertscore_recall": {
    "mean": 0.5236848592758179,
    "std": 0.03384505584836006,
    "min": 0.47408121824264526,
    "max": 0.5783898830413818,
    "median": 0.5224673748016357
  },
  "bertscore_analysis": {
    "description": "BERTScore uses contextual embeddings for semantic similarity",
    "model_type": "distilroberta-base",
    "use_idf": false,
    "rescale_with_baseline": true,
    "range": "Typically [-1, 1] but varies by model and baseline",
    "interpretation": {
      "advantages": "Captures semantic similarity beyond surface-level matches",
      "f1_score": "Harmonic mean of precision and recall (most commonly reported)",
      "use_case": "Good for evaluating semantic content preservation"
    },
    "score_characteristics": {
      "negative_scores_count": 0,
      "high_scores_pct": "0.0% (> 0.8)",
      "baseline_note": "Scores rescaled with baseline if enabled"
    },
    "precision_vs_recall": {
      "precision_mean": 0.6137832403182983,
      "recall_mean": 0.5236848592758179,
      "difference": 0.09009838104248047,
      "note": "Precision > Recall suggests conservative generation"
    }
  }
}
```
