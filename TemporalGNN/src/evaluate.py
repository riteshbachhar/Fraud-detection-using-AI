#!/usr/bin/env python3
import logging
from pathlib import Path
from pyexpat import model
import torch
from model import create_model
from utils import load_config
from sklearn.metrics import fbeta_score, recall_score
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path):
        model = create_model(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def find_optimal_threshold(self, probs, labels):
        """Find optimal threshold for F2 score"""
        best_f2 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs >= threshold).astype(int)
            f2 = fbeta_score(labels, preds, beta=2, average='binary', zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
        
        return best_threshold, best_f2
    
    def evaluate_model(self, model, snapshots, global_num_nodes, split_name='test', threshold=None):
        """Comprehensive model evaluation
        split_name: 'val', or 'test'
        """
        logger.info(f"Evaluating on {split_name} set...")
        
        probs_list, labels_list = [], []
        
        with torch.no_grad():
            h = torch.zeros(global_num_nodes, self.config['model']['hidden_dim']).to(self.device)
            for snap in snapshots:
                snap = snap.to(self.device)
                out, h = model(snap, h)
                preds = torch.sigmoid(out).squeeze().cpu().numpy()
                probs_list.extend(preds)
                labels_list.extend(snap.y.cpu().numpy())
        
        probs = np.array(probs_list)
        labels = np.array(labels_list)
        
        # Find optimal threshold
        if threshold is None:
            optimal_threshold, _ = self.find_optimal_threshold(probs, labels)
        else:
            optimal_threshold = threshold
        # Generate binary predictions
        binary_preds = (probs >= optimal_threshold).astype(int)
        
        # Calculate all metrics
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision_score(labels, binary_preds, zero_division=0),
            'recall': recall_score(labels, binary_preds, zero_division=0),
            'f1': fbeta_score(labels, binary_preds, beta=1, zero_division=0),
            'f2': fbeta_score(labels, binary_preds, beta=2, zero_division=0),
            'roc_auc': roc_auc_score(labels, probs),
            'pr_auc': average_precision_score(labels, probs)
        }
        
        # Print results
        logger.info(f"{split_name.upper()} RESULTS:")
        logger.info(f"Optimal Threshold: {metrics['threshold']:.3f}")
        logger.info(f"F2 Score: {metrics['f2']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")
        
        # Generate plots
        self.plot_results(probs, labels, binary_preds, split_name)
        
        return metrics, probs, labels

    def plot_results(self, probs, labels, preds, split_name):
        """Generate evaluation plots"""
        from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        axes[0].plot(recall, precision, linewidth=2)
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title(f'Precision-Recall Curve (AP={ap:.3f})')
        axes[0].grid(True, alpha=0.3)
        
        # Confusion Matrix with seaborn heatmap
        cm = confusion_matrix(labels, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    square=True, ax=axes[1],
                    xticklabels=['Non-suspicious', 'Suspicious'],
                    yticklabels=['Non-suspicious', 'Suspicious'])
        axes[1].set_title('Confusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'results/{split_name}_evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main evaluation pipeline"""
    # Load config
    config = load_config('config.yaml')
    snapshots = torch.load('data/temporal_graph_snapshots.pth')
    graph_info = torch.load('data/graph_info.pth')

    # Split data (Same )
    train_size = int(len(snapshots) * (1 - config['preprocessing']['validation_split'] - config['preprocessing']['test_split']))
    val_size = int(len(snapshots) * config['preprocessing']['validation_split'])

    # Evaluate model
    evaluator = ModelEvaluator(config)
    model = evaluator.load_model('weights/model_weights.pth')

    # Validation set evaluation
    val_snapshots = snapshots[train_size: train_size + val_size]
    val_metrics, val_probs, val_labels = evaluator.evaluate_model(
        model, val_snapshots, graph_info['num_nodes'], 'val'
    )
    # Save results
    with open(f'results/val_metrics.json', 'w') as f:
        json.dump(val_metrics, f, indent=2)
    np.save(f'results/val_predictions.npy',
            {'probs': val_probs, 'labels': val_labels})

    # Test set evaluation
    threshold = val_metrics['threshold']
    test_snapshots = snapshots[train_size + val_size:]
    test_metrics, test_probs, test_labels = evaluator.evaluate_model(
        model, test_snapshots, graph_info['num_nodes'], 'test', threshold=threshold
    )

    # Save results
    with open(f'results/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    np.save(f'results/test_predictions.npy',
            {'probs': test_probs, 'labels': test_labels})

if __name__ == "__main__":
    main()