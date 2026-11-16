"""
Feel free to replace this code with your own model training code.
This is just a simple example to get you started.

This training script uses imports relative to the project root directory.
To run this training script with uv, ensure you're in the project root directory
and execute: uv run -m submission.fashion_training
"""
import argparse
from pathlib import Path

import torch, torchvision

from submission import engine
from submission.fashion_model import Net

def train_fashion_model(fashion_mnist,
                        n_epochs,
                        batch_size=32,
                        learning_rate=0.001,
                        USE_GPU=False,
                        seed=None,
                        model_kwargs=None,
                        return_state_dict_only=True,
                        split_indices=None,
                        horizontal_flip=False,
                        ):
    """
    You can modify the contents of this function as needed, but DO NOT CHANGE the arguments,
    the function name, or the return value when return_state_dict_only=True, as this will be called during marking!
    (You can change the default values or add additional keyword arguments if needed.)
    During marking this is also run on a small dummy dataset (30 samples, 1 epoch), so it must work on that too.
    """
    # Optionally fix the random seed, so that results can be repeated
    if seed is not None:
        engine.set_seed(seed)

    # Optionally use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if split_indices is None:
        train_size = int(0.8 * len(fashion_mnist))
        val_size = len(fashion_mnist) - train_size
        train_split, val_split = torch.utils.data.random_split(
            fashion_mnist, [train_size, val_size]
        )
        train_indices = train_split.indices
        val_indices = val_split.indices
    else:
        train_indices, val_indices = split_indices

    train_data = engine.SubsetWithTransform(
        fashion_mnist, train_indices, get_transforms('train', horizontal_flip=horizontal_flip)
    )


    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             )
    val_loader = None
    if len(val_indices):
        val_data = engine.SubsetWithTransform(
            fashion_mnist, val_indices, get_transforms('eval')
        )
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = Net(**(model_kwargs or {}))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    # Training loop
    train_losses, val_losses, val_accuracies = [], [], []
    for epoch in range(n_epochs):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{n_epochs}], Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)
        if val_loader is not None:
            val_loss, accuracy = engine.evaluate(model, val_loader, criterion, device)
            print(f"Epoch [{epoch + 1}/{n_epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
            val_losses.append(val_loss)
            val_accuracies.append(accuracy)
        scheduler.step()

    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    if return_state_dict_only:
        return model.state_dict()
    # Add whatever else you want to return here - return_state_dict_only=True will be enforced at runtime
    return model.state_dict(), train_losses, val_losses, val_accuracies


def get_transforms(mode='train', *, horizontal_flip=False):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions, sorry), please check that
    these pass by running model_calls.py before submission. The 'eval' transforms must be
    deterministic (no random augmentation), so only use transforms such as ToTensor, Normalize,
    Pad, Resize or CenterCrop there.
    """
    if mode not in ('train', 'eval'):
        raise ValueError(f"Unknown mode {mode!r}, expected 'train' or 'eval'.")

    preprocessing = []
    if mode == 'train' and horizontal_flip:
        preprocessing.append(torchvision.transforms.RandomHorizontalFlip(p=0.5))
    preprocessing.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(preprocessing)


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.ipynb!
    return fashion_mnist


def main():
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST model.")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--full-data", action="store_true", help="Train on all images without a validation split.")
    parser.add_argument("--gpu", action="store_true", help="Train on CUDA.")
    parser.add_argument("--horizontal-flip", action="store_true", help="Flip training images with probability 0.5.")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.gpu and not torch.cuda.is_available():
        parser.error("--gpu requires a CUDA runtime")
    if args.gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = load_training_data()
    training_options = dict(USE_GPU=args.gpu, seed=0, horizontal_flip=args.horizontal_flip)

    split_indices = None
    checkpoint_name = "model_weights.pth"
    if args.full_data:
        split_indices = (range(len(dataset)), [])
        checkpoint_name = f"model_weights_full_{args.epochs}_epochs.pth"
        print(f"Training on all {len(dataset):,} images for {args.epochs} epochs.")

    # Comment out this call before enabling cross-validation.
    weights = train_fashion_model(
        dataset,
        n_epochs=19,
        split_indices=(range(len(dataset)), []), #train on all data for final submission
        **training_options,
    )

    # To use cross-validation, uncomment the block below through the final fit.
    # from time import perf_counter
    # from sklearn.model_selection import StratifiedKFold
    #
    # splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # folds = list(splitter.split(dataset.data, dataset.targets))
    # fold_accuracies = []
    # fold_losses = []
    # cv_start = perf_counter()

    # for fold_number, split in enumerate(folds, start=1):
    #     print(f"\nFold {fold_number}/{len(folds)}")
    #     fold_start = perf_counter()
    #     _, _, val_losses, val_accuracies = train_fashion_model(
    #         dataset,
    #         n_epochs=args.epochs,
    #         split_indices=split,
    #         return_state_dict_only=False,
    #         **training_options,
    #     )
    #     fold_losses.append(val_losses)
    #     fold_accuracies.append(val_accuracies)
    #     print(f"Fold {fold_number} wall-clock time: {perf_counter() - fold_start:.1f} s")

    # mean_accuracies = []
    # mean_losses = []
    # for epoch_index, (accuracies, losses) in enumerate(
    #     zip(zip(*fold_accuracies), zip(*fold_losses)), start=1
    # ):
    #     mean_accuracy = sum(accuracies) / len(accuracies)
    #     mean_loss = sum(losses) / len(losses)
    #     mean_accuracies.append(mean_accuracy)
    #     mean_losses.append(mean_loss)
    #     print(f"Epoch {epoch_index}: mean validation accuracy {mean_accuracy:.2%}, mean val loss {mean_loss:.4f}")

    # print(f"Cross-validation wall-clock time: {perf_counter() - cv_start:.1f} s")
    # best_index = max(range(args.epochs), key=lambda i: (mean_accuracies[i], -mean_losses[i]))
    # final_epochs = best_index + 1
    # print(f"\nTraining on all {len(dataset):,} images for {final_epochs} epochs.")
    # weights = train_fashion_model(
    #     dataset,
    #     n_epochs=final_epochs,
    #     split_indices=(range(len(dataset)), []),
    #     **training_options,
    # )

    checkpoint = Path(__file__).with_name(checkpoint_name)
    torch.save({name: tensor.detach().cpu() for name, tensor in weights.items()}, checkpoint)
    print(f"Saved {checkpoint}")


if __name__ == "__main__":
    main()
