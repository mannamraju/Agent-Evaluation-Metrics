"""
Bounding box evaluation for CXR report generation.

This module implements bounding box IoU (Intersection over Union) evaluation
for assessing the accuracy of predicted bounding boxes against ground truth.
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union, Tuple

from .base_evaluator import BaseEvaluator


class BoundingBoxEvaluator(BaseEvaluator):
    """Evaluator for bounding box precision and recall using IoU matching.
    
    This evaluator compares predicted bounding boxes with ground truth boxes
    using Intersection over Union (IoU) thresholding to compute precision
    and recall metrics at the dataset level.
    """
    
    def __init__(self, 
                 iou_threshold: float = 0.5,
                 box_column: str = "boxes",
                 **kwargs):
        """Initialize bounding box evaluator.
        
        Args:
            iou_threshold: IoU threshold for considering a match
            box_column: Name of column containing bounding box data
            **kwargs: Additional arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.box_column = box_column
        
        # Storage for computed metrics
        self._last_results = None
    
    def _normalize_box(self, box: Union[Dict, List, Tuple]) -> List[float]:
        """Normalize box to [x1, y1, x2, y2] format.
        
        Supports multiple input formats:
        - Dict with keys (x,y,w,h) or (x1,y1,x2,y2)
        - List/tuple of 4 values
        
        Args:
            box: Box in various formats
            
        Returns:
            Box in [x1, y1, x2, y2] format
        """
        if isinstance(box, dict):
            # Format: {x, y, w, h}
            if all(k in box for k in ('x', 'y', 'w', 'h')):
                x1 = float(box['x'])
                y1 = float(box['y'])
                x2 = x1 + float(box['w'])
                y2 = y1 + float(box['h'])
                return [x1, y1, x2, y2]
            
            # Format: {x1, y1, x2, y2}
            if all(k in box for k in ('x1', 'y1', 'x2', 'y2')):
                return [float(box['x1']), float(box['y1']), float(box['x2']), float(box['y2'])]
            
            # Fallback: try first 4 values
            keys = list(box.keys())
            if len(keys) >= 4:
                values = [float(box[k]) for k in keys[:4]]
                return values
            else:
                raise ValueError(f"Insufficient box coordinates in dict: {box}")
        
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = map(float, box)
            
            # Check if format is [x1,y1,x2,y2] (x2 > x1)
            if x2 >= x1 and y2 >= y1:
                return [x1, y1, x2, y2]
            
            # Otherwise treat as [x,y,w,h]
            return [x1, y1, x1 + x2, y1 + y2]
        
        else:
            raise ValueError(f"Unsupported box format: {box}")
    
    def _compute_iou(self, box_a: List[float], box_b: List[float]) -> float:
        """Compute Intersection over Union (IoU) between two boxes.
        
        Args:
            box_a: First box [x1, y1, x2, y2]
            box_b: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Compute intersection
        x_left = max(box_a[0], box_b[0])
        y_top = max(box_a[1], box_b[1])
        x_right = min(box_a[2], box_b[2])
        y_bottom = min(box_a[3], box_b[3])
        
        # Check if there's no intersection
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        # Compute intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Compute union area
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union_area = area_a + area_b - intersection_area
        
        # Avoid division by zero
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _parse_boxes(self, box_entry: Any) -> List[List[float]]:
        """Parse box entry into list of normalized boxes.
        
        Args:
            box_entry: Box data (JSON string, list, or other format)
            
        Returns:
            List of boxes in [x1, y1, x2, y2] format
        """
        try:
            # Handle JSON strings
            if isinstance(box_entry, str):
                boxes_data = json.loads(box_entry)
            else:
                boxes_data = box_entry
            
            # Handle None or empty
            if boxes_data is None:
                return []
            
            # Ensure it's a list
            if not isinstance(boxes_data, list):
                boxes_data = [boxes_data]
            
            # Normalize each box
            normalized_boxes = []
            for box in boxes_data:
                try:
                    normalized_box = self._normalize_box(box)
                    normalized_boxes.append(normalized_box)
                except Exception as e:
                    print(f"Warning: Failed to normalize box {box}: {e}")
                    continue
            
            return normalized_boxes
        
        except Exception as e:
            print(f"Warning: Failed to parse box entry {box_entry}: {e}")
            return []
    
    def _compute_dataset_metrics(self, 
                                gt_boxes_series: pd.Series, 
                                pred_boxes_series: pd.Series) -> Dict[str, Any]:
        """Compute precision and recall at the dataset level.
        
        Args:
            gt_boxes_series: Series of ground truth box data
            pred_boxes_series: Series of predicted box data
            
        Returns:
            Dictionary with precision, recall, and detailed statistics
        """
        total_tp = 0
        total_pred_boxes = 0
        total_gt_boxes = 0
        processed_samples = 0
        failed_samples = 0
        
        # Process each sample
        for gt_entry, pred_entry in zip(gt_boxes_series, pred_boxes_series):
            try:
                # Parse boxes
                gt_boxes = self._parse_boxes(gt_entry)
                pred_boxes = self._parse_boxes(pred_entry)
                
                total_gt_boxes += len(gt_boxes)
                total_pred_boxes += len(pred_boxes)
                
                # Find matches using IoU threshold
                matched_gt = set()
                
                for pred_box in pred_boxes:
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    # Find best matching ground truth box
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue  # Already matched
                        
                        iou = self._compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    # Check if match meets threshold
                    if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                        total_tp += 1
                        matched_gt.add(best_gt_idx)
                
                processed_samples += 1
            
            except Exception as e:
                print(f"Warning: Failed to process sample: {e}")
                failed_samples += 1
                continue
        
        # Compute metrics
        precision = total_tp / total_pred_boxes if total_pred_boxes > 0 else 0.0
        recall = total_tp / total_gt_boxes if total_gt_boxes > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'true_positives': total_tp,
            'total_predictions': total_pred_boxes,
            'total_ground_truth': total_gt_boxes,
            'processed_samples': processed_samples,
            'failed_samples': failed_samples,
            'iou_threshold': self.iou_threshold
        }
    
    def compute_metric(self, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Compute bounding box metrics.
        
        Note: Bounding box evaluation is dataset-level, so no columns are added to pred_df.
        
        Args:
            gt_df: Ground truth dataframe with bounding box column
            pred_df: Predictions dataframe with bounding box column
            
        Returns:
            pred_df unchanged (results stored for summary)
        """
        # Check if box columns exist
        if self.box_column not in gt_df.columns:
            print(f"Warning: Box column '{self.box_column}' not found in ground truth data")
            self._last_results = None
            return pred_df
        
        if self.box_column not in pred_df.columns:
            print(f"Warning: Box column '{self.box_column}' not found in prediction data")
            self._last_results = None
            return pred_df
        
        # Align dataframes
        gt_aligned, pred_aligned = self.align_dataframes(gt_df, pred_df)
        
        # Compute metrics
        print(f"Computing bounding box metrics with IoU threshold {self.iou_threshold}...")
        
        try:
            results = self._compute_dataset_metrics(
                gt_aligned[self.box_column], 
                pred_aligned[self.box_column]
            )
            
            self._last_results = results
            
            print(f"Box precision: {results['precision']:.4f}")
            print(f"Box recall: {results['recall']:.4f}")
            print(f"Box F1: {results['f1_score']:.4f}")
            print(f"Processed {results['processed_samples']} samples")
        
        except Exception as e:
            print(f"Warning: Bounding box evaluation failed: {e}")
            self._last_results = None
        
        return pred_aligned
    
    def get_metric_columns(self) -> List[str]:
        """Get the list of column names this metric adds.
        
        Bounding box evaluation is dataset-level, so no columns are added.
        
        Returns:
            Empty list (no per-report columns)
        """
        return []
    
    def get_bbox_results(self) -> Optional[Dict[str, Any]]:
        """Get the most recent bounding box evaluation results.
        
        Returns:
            Dictionary with bounding box results or None if not available
        """
        return self._last_results
    
    def get_summary_stats(self, pred_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for bounding box metrics.
        
        Args:
            pred_df: Dataframe (not used for dataset-level bbox metrics)
            
        Returns:
            Dictionary with bounding box summary statistics
        """
        summary = {}
        
        # Get bounding box results
        bbox_results = self.get_bbox_results()
        
        if bbox_results:
            summary['bounding_box'] = bbox_results
            
            # Add detailed analysis
            bbox_info = {
                'description': f'Bounding box evaluation using IoU threshold {self.iou_threshold}',
                'evaluation_level': 'Dataset-level (not per-report)',
                'matching_strategy': 'Best IoU match above threshold',
                'box_column': self.box_column,
                'interpretation': {
                    'precision': 'Fraction of predicted boxes that match ground truth',
                    'recall': 'Fraction of ground truth boxes that are detected', 
                    'f1_score': 'Harmonic mean of precision and recall',
                    'use_case': 'Evaluates localization accuracy for anatomical regions'
                }
            }
            
            # Analyze performance
            precision = bbox_results['precision']
            recall = bbox_results['recall']
            f1 = bbox_results['f1_score']
            
            if f1 > 0.7:
                performance_level = "Excellent localization accuracy"
            elif f1 > 0.5:
                performance_level = "Good localization accuracy" 
            elif f1 > 0.3:
                performance_level = "Moderate localization accuracy"
            else:
                performance_level = "Poor localization accuracy"
            
            bbox_info['performance_assessment'] = {
                'level': performance_level,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'balance_note': 'High precision, low recall = conservative; Low precision, high recall = liberal'
            }
            
            # Statistics summary
            bbox_info['dataset_statistics'] = {
                'total_gt_boxes': bbox_results['total_ground_truth'],
                'total_pred_boxes': bbox_results['total_predictions'],
                'avg_gt_per_sample': bbox_results['total_ground_truth'] / max(bbox_results['processed_samples'], 1),
                'avg_pred_per_sample': bbox_results['total_predictions'] / max(bbox_results['processed_samples'], 1)
            }
            
            # Check for potential issues
            if bbox_results['failed_samples'] > 0:
                bbox_info['warnings'] = [
                    f"Failed to process {bbox_results['failed_samples']} samples",
                    "Check box format consistency"
                ]
            
            if bbox_results['total_predictions'] == 0:
                bbox_info['warnings'] = bbox_info.get('warnings', [])
                bbox_info['warnings'].append("No predicted boxes found")
            
            summary['bounding_box_analysis'] = bbox_info
        
        else:
            summary['bounding_box_analysis'] = {
                'status': 'Bounding box evaluation not available',
                'note': f"Box column '{self.box_column}' may be missing from input data"
            }
        
        return summary
    
    @property
    def name(self) -> str:
        """Get descriptive name for this evaluator."""
        return f"BoundingBoxEvaluator(IoU≥{self.iou_threshold})"
