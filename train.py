import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from training import train_model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train diffusion model')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--no-visualize', action='store_true', help='Disable visualization')
    parser.add_argument('--checkpoint-interval', type=int, default=1000, help='Checkpoint interval')
    
    args = parser.parse_args()
    
    model, history = train_model(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        visualize=not args.no_visualize
    )
    
    print("Training completed!")
    print(f"Final loss: {history['final_loss']:.3f}")
