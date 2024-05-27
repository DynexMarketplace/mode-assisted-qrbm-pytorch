import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from marbm import MARBM

def plot_visualization_data(rbm_cd, rbm_mode):
    """Visualize metrics and sigmoid values for two trained RBMs."""
    # Extract visualization data
    metrics_name_cd, metrics_values_cd, sigm_values_cd = rbm_cd.get_visualization_data()
    metrics_name_mode, metrics_values_mode, sigm_values_mode = rbm_mode.get_visualization_data()

    # Ensure consistency in metrics between the two RBMs
    assert metrics_name_cd == metrics_name_mode, "The two RBMs are tracking different metrics!"

    # Visualization setup
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(metrics_values_cd, label="CD Trained RBM")
    ax1.plot(metrics_values_mode, label="Mode Assisted RBM")
    ax1.set_xlabel('Steps')
    ax1.set_ylabel(metrics_name_cd)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # Instantiate a second y-axis sharing the same x-axis
    if sigm_values_cd:
        ax2.plot(sigm_values_cd, '--', label="CD Trained RBM Sigm Values")
    if sigm_values_mode:
        ax2.plot(sigm_values_mode, '--', label="Mode Assisted RBM Sigm Values")
    ax2.set_ylabel('Sigmoid Value')
    ax2.legend(loc='upper right')

    plt.title(f'{metrics_name_cd} and Sigmoid Values over Training Steps')
    ax1.grid(True)
    plt.show()

def display_reconstructions(original, reconstructed_cd, reconstructed_mode, image_width):
    """Display original and reconstructed images side-by-side."""
    num_images = 6
    fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(12, 8))

    def plot_image(ax, img, title):
        """Utility function to display an image."""
        ax.imshow(img.detach().numpy().reshape(image_width, image_width), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    for i in range(num_images):
        plot_image(axes[0, i], original[i], "Original" if i == 0 else "")
        plot_image(axes[1, i], reconstructed_cd[i], "CD" if i == 0 else "")
        plot_image(axes[2, i], reconstructed_mode[i], "Mode Assisted" if i == 0 else "")

    plt.tight_layout()
    plt.show()

def main():
    """Main training and evaluation routine."""
    # Configuration and Hyperparameters
    batch_size = 8
    image_width = 20
    visible_units = image_width * image_width
    hidden_units = 20
    epochs = 8
    lr = 0.1
    k = 1
    sigm_a = 20
    sigm_b = -6
    p_max = 0.05
    plotper = 1000
    seed = 55

    # Seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((image_width, image_width)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.round(x.view(-1)))
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Seed for rbm_cd reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Training MARBM with CD and Mode-assisted methods
    rbm_cd = MARBM(visible_units, hidden_units, seed=seed)
    rbm_cd.train(train_loader, val_loader=val_loader, epochs=epochs, lr=lr, k=k, sigm_a=sigm_a, sigm_b=sigm_b, p_max=0.0, plotper=plotper, loss_metric='free_energy')


    # Seed for rbm_mode reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    rbm_mode = MARBM(visible_units, hidden_units, seed=seed)
    rbm_mode.set_sampler_parameters(num_reads=3)
    rbm_mode.train(train_loader, val_loader=val_loader, epochs=epochs, lr=lr, k=k, sigm_a=sigm_a, sigm_b=sigm_b, p_max=p_max, plotper=plotper, loss_metric='free_energy')

    # Sample validation data for evaluation
    sample_val_data = next(iter(val_loader))[0]

    # Lock weights, extract features, and unlock
    rbm_mode.lock_weights()
    features_mode = rbm_mode.extract_features(sample_val_data)
    print("Features from rbm_mode:", features_mode.shape)
    rbm_mode.unlock_weights()

    # Model Saving and Loading Test
    model_save_path = "./rbm_mode_checkpoint.pth"
    rbm_mode.save_model(model_save_path)
    rbm_mode_new = MARBM(visible_units, hidden_units)
    rbm_mode_new.load_model(model_save_path)

    # Ensure loaded model consistency
    features_from_saved_model = rbm_mode_new.extract_features(sample_val_data)
    assert torch.allclose(features_mode, features_from_saved_model), "The loaded model does not match the original model!"
    print("Successfully saved and loaded the model!")

    # Visualization
    reconstructed_data_cd = rbm_cd.reconstruct(sample_val_data)
    reconstructed_data_mode = rbm_mode.reconstruct(sample_val_data)
    display_reconstructions(sample_val_data, reconstructed_data_cd, reconstructed_data_mode, image_width)
    plot_visualization_data(rbm_cd, rbm_mode)

if __name__ == "__main__":
    main()
