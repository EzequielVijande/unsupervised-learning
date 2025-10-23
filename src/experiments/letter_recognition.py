import json
from matplotlib import pyplot as plt
import numpy as np
from src.networks.hopfield import HopfieldNetwork

DATASET_PATH = "./datasets/letters.json"

def plot_letter_pattern(letter_data, letter_char=None, ax=None, title=None):
    """
    Plot a single letter pattern from the JSON data.
    
    Parameters:
    - letter_data: 5x5 list of lists containing -1 and 1 values
    - letter_char: The letter character (for title)
    - ax: matplotlib axis to plot on (if None, creates new figure)
    - title: Custom title for the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    # Convert to numpy array for easier handling
    pattern = np.array(letter_data)
    
    # Create a colormap: -1 -> white, 1 -> black
    cmap = plt.cm.binary
    im = ax.imshow(pattern, cmap=cmap, vmin=-1, vmax=1)
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    if title:
        ax.set_title(title)
    elif letter_char:
        ax.set_title(f"Letter '{letter_char}'")

    return ax

def plot_all_letters(letters_dict, figsize=(15, 10)):
    """
    Plot all 26 letters in a grid layout.
    
    Parameters:
    - letters_dict: Dictionary containing all letter patterns
    - figsize: Figure size for the overall plot
    """
    fig, axes = plt.subplots(4, 7, figsize=figsize)
    axes = axes.flatten()
    
    # Get sorted list of letters
    sorted_letters = sorted(letters_dict.keys())
    
    for i, letter in enumerate(sorted_letters):
        if i < len(axes):
            plot_letter_pattern(letters_dict[letter], letter_char=letter, ax=axes[i])
    
    # Hide any unused subplots
    for i in range(len(sorted_letters), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def genertate_letter_group(letter_dict, letters):
    n_neurons = len(np.array(letter_dict[letters[0]], dtype=np.float32).flatten())
    letter_gr = np.zeros((len(letters),n_neurons))
    for i, letter in enumerate(letters):
        letter_gr[i] = np.array(letter_dict[letter], dtype=np.float32).flatten()
    return letter_gr


def main():
    noise_std = 0.5
    letters2group = ['A', 'B', 'C', 'D']
    with open(DATASET_PATH, 'r') as f:
        dst = json.load(f)
    # plot_all_letters(dst['letters'])
    patterns = genertate_letter_group(dst['letters'], letters2group)
    n_patterns, n_neurons = patterns.shape
    print(f'Number of patterns= {n_patterns}')
    print(f'N neurons= {n_neurons}')
    nn = HopfieldNetwork(patterns.shape[1])
    nn.train(patterns)
    #Add noise
    noise = np.random.normal(0, noise_std, patterns.shape)
    noisy = (patterns + noise).clip(-1, 1)
    results = nn.recall(noisy[0], async_update=False)
    for res in results:
        rshpd = res.reshape((5,5))
        plt.figure()
        plt.imshow(rshpd, cmap='binary')
        plt.show()

if __name__ == '__main__':
    main()
