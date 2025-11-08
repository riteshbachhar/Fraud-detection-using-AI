#!/usr/bin/env python3
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import fbeta_score
try:
    from model import create_model
    from utils import FocalLoss, load_config
except ImportError:
    from src.model import create_model
    from src.utils import FocalLoss, load_config
from sklearn.metrics import recall_score
import json
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
    
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def find_optimal_threshold(self, probs, labels, 
                               threshold_search_range=np.arange(0.05, 0.95, 0.05)):
        """Find optimal threshold for F2 score"""
        best_f2 = 0
        best_threshold = 0.5

        for threshold in threshold_search_range:
            preds = (probs >= threshold).astype(int)
            f2 = fbeta_score(labels, preds, beta=self.config['training']['beta'], average='binary', zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
        return best_threshold, best_f2

    def train_model(self, train_snapshots, val_snapshots, global_num_nodes):
        """Enhanced training with F2 optimization"""

        # Initialize model
        model = create_model(self.config).to(self.device)
        
        # Focal Loss
        criterion = FocalLoss(alpha=self.config['loss']['alpha'], gamma=self.config['loss']['gamma'])
        
        # Optimizer with different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': model.rnn.parameters(), 'lr': self.config['training']['learning_rate'] * 0.5},
            {'params': model.gnn1.parameters(), 'lr': self.config['training']['learning_rate']},
            {'params': model.gnn2.parameters(), 'lr': self.config['training']['learning_rate']},
            {'params': model.gnn3.parameters(), 'lr': self.config['training']['learning_rate']},
            {'params': model.classifier.parameters(), 'lr': self.config['training']['learning_rate'] * 1.5}
        ], weight_decay=self.config['training']['weight_decay'])

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=self.config['scheduler']['factor'], 
            patience=self.config['scheduler']['patience']
        )
        
        # Training loop
        best_f2_score = 0
        patience_counter = 0
        train_loss_history = []
        val_loss_history = []
        f2_history = []

        for epoch in range(self.config['training']['epochs']):
            # Sequence length for BPTT
            k_steps = self.config['training']['sequence_length']
            # Training mode
            model.train()
            train_loss = 0.0

            # Initialize hidden state at the start of each epoch
            h = torch.zeros(global_num_nodes, self.config['model']['hidden_dim']).to(self.device)

            optimizer.zero_grad()
            chunk_loss = 0.0

            for i, snap in enumerate(train_snapshots):
                snap = snap.to(self.device)
                out, h = model(snap, h)      # Forward pass
                loss = criterion(out.squeeze(), snap.y)  # Loss computation
                chunk_loss += loss

                # Backpropagation every k_steps
                if (i + 1) % k_steps == 0 or (i + 1) == len(train_snapshots):
                    chunk_loss /= k_steps
                    # Backward pass
                    chunk_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                    optimizer.step()
                    # Detach h
                    h = h.detach()
                    # Reset for next chunk
                    optimizer.zero_grad()
                    chunk_loss = 0.0

                # Total epoch loss
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_snapshots)
            train_loss_history.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_probs_list, val_labels_list = [], []
            val_loss = 0
            
            with torch.no_grad():
                h = torch.zeros(global_num_nodes, self.config['model']['hidden_dim']).to(self.device)
                for snap in val_snapshots:
                    snap = snap.to(self.device)
                    out, h = model(snap, h)
                    loss = criterion(out.squeeze(), snap.y)
                    val_loss += loss.item()
                    
                    preds = torch.sigmoid(out).squeeze()
                    val_probs_list.append(preds.cpu())
                    val_labels_list.append(snap.y.cpu())

            avg_val_loss = val_loss / len(val_snapshots)
            val_loss_history.append(avg_val_loss)
            
            # Calculate F2 score with optimal threshold
            val_probs = torch.cat(val_probs_list).numpy()
            val_labels = torch.cat(val_labels_list).numpy()
            
            optimal_threshold, f2_score = self.find_optimal_threshold(val_probs, val_labels)
            f2_history.append(f2_score)
            recall = recall_score(val_labels, (val_probs >= optimal_threshold).astype(int), zero_division=0)
            
            # Update scheduler with validation loss
            scheduler.step(avg_val_loss)
            
            # Early stopping based on F2 score
            if f2_score > best_f2_score:
                best_f2_score = f2_score
                patience_counter = 0
            else:
                patience_counter += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch {epoch+1}: Train Loss(x1e3): {1000*avg_train_loss:.4f}, Val Loss(x1e3): {1000*avg_val_loss:.4f}, "
                        f"F2: {f2_score:.4f}, Threshold: {optimal_threshold:.3f}, Recall: {recall:.4f}, "
                        f"LR: {current_lr:.6f}")

            if patience_counter >= self.config['training']['patience']:
                logger.info("Early stopping triggered.")
                break
        
        # Load best model and evaluate
        # model.load_state_dict(torch.load('./outputs/best_model.pth'))
        
        results = {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'model': model,
            'optimal_threshold': optimal_threshold,
            'f2_history': f2_history,
            'best_f2_score': best_f2_score,
        }
        
        return results
    
def main():
    """ Main training pipeline """
    # Set the stage
    config = load_config('config.yaml')
    snapshots = torch.load('data/temporal_graph_snapshots.pth')
    graph_info = torch.load('data/graph_info.pth')

    # Data splitting
    train_size = int(len(snapshots) * (1 - config['preprocessing']['validation_split'] - config['preprocessing']['test_split']))
    val_size = int(len(snapshots) * config['preprocessing']['validation_split'])

    train_snaps = snapshots[:train_size]
    val_snaps = snapshots[train_size:train_size + val_size]
    # test_snaps = snapshots[train_size + val_size:]
    logger.info(f"Data split - Train: {len(train_snaps)}, Val: {len(val_snaps)})")

    # Trainer
    trainer = ModelTrainer(config)
    results = trainer.train_model(train_snaps, val_snaps, graph_info['num_nodes'])

    # Save the trained model
    Path('results').mkdir(parents=True, exist_ok=True)
    Path('weights').mkdir(parents=True, exist_ok=True)
    metrics = {
        'train_loss_history': results['train_loss_history'],
        'val_loss_history': results['val_loss_history'],
        'optimal_threshold': results['optimal_threshold'],
        'f2_history': results['f2_history'],
        'best_f2_score': results['best_f2_score'],
    }
    with open('results/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    # Save model weights
    torch.save(results['model'].state_dict(), 'weights/model_weights.pth')
    logger.info("Training complete. Metrics saved to 'results/training_metrics.json' \
                and model weights to 'weights/model_weights.pth'.")

if __name__ == "__main__":
    main()
