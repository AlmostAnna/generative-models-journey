import os
import sys
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train generative models')
    parser.add_argument('model_type', choices=['diffusion', 'ebm'], 
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Checkpoint interval')
    
    args = parser.parse_args()
    
    if args.model_type == 'diffusion':
        from src.training import train_model
        model, history = train_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            visualize=not args.no_visualize
        )
        print("Finished training diffusion model.")
        print(f"Final loss: {history['final_loss']:.3f}")

    elif args.model_type == 'ebm':
        from src.training import train_ebm_model
        model, history = train_ebm_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            visualize=not args.no_visualize
        )
        print("Finished training energy-based model.")
        print(f"Final loss: {history['final_loss']:.3f}")
        
if __name__ == "__main__":
    main()
