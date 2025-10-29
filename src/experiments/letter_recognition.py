import json
from matplotlib import pyplot as plt
import numpy as np
from src.networks.hopfield import HopfieldNetwork
import pandas as pd
import seaborn as sns
import string

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

def similarity_arr(patterns):
    matrix = np.stack(patterns) 
    result = matrix @ matrix.T
    return result/patterns[0].size

def plot_similarity_heatmap(similarity_matrix, labels, title="Pattern Similarities", figsize=(10, 8)):
    """
    Plot similarity matrix as a heatmap with labels.
    
    Parameters:
    - similarity_matrix: 2D numpy array of similarities
    - labels: List of labels for rows/columns
    - title: Plot title
    - figsize: Figure size
    """
    # Convert to DataFrame for nice labeling
    df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df, 
                    annot=True,           # Show values in cells
                    fmt=".2f",           # Format to 2 decimal places
                    cmap="seismic",       # Red-Yellow-Blue colormap
                    center=0,            # Center colormap at 0
                    vmin=-1, vmax=1,     # Color scale limits
                    square=True,         # Square cells
                    cbar_kws={'label': 'Similarity'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    return ax

def add_noise(pattern, noise_level):
    """
    Add noise to a pattern by randomly flipping bits.
    
    Parameters:
    - pattern: 1D numpy array with values -1 or 1
    - noise_level: float between 0 and 1 (e.g., 0.25 for 25% noise)
    
    Returns:
    - noisy_pattern: 1D numpy array with noise added
    """
    pattern = np.array(pattern).copy()
    n_neurons = len(pattern)
    n_flips = int(n_neurons * noise_level)
    
    # Choose random indices to flip
    flip_indices = np.random.choice(n_neurons, size=n_flips, replace=False)
    
    # Flip the selected bits (multiply by -1)
    pattern[flip_indices] *= -1
    
    return pattern

def plot_recall_steps(steps_list, original_pattern, noisy_pattern, title_prefix):
    """
    Plot the recall process step by step.
    
    Parameters:
    - steps_list: List of patterns from recall() method
    - original_pattern: Original clean pattern
    - noisy_pattern: Initial noisy pattern
    - title_prefix: Prefix for the overall title
    """
    n_plots = len(steps_list) + 2  # original + noisy + all steps
    fig, axes = plt.subplots(1, n_plots, figsize=(3*n_plots, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Original pattern
    original_2d = original_pattern.reshape((5, 5))
    plot_letter_pattern(original_2d, ax=axes[0], title="Original")
    
    # Plot 2: Noisy pattern
    noisy_2d = noisy_pattern.reshape((5, 5))
    plot_letter_pattern(noisy_2d, ax=axes[1], title="Ruidoso")
    
    # Plots 3+: Each step from recall
    for i, step_pattern in enumerate(steps_list):
        step_2d = step_pattern.reshape((5, 5))
        if i == len(steps_list) - 1:
            title = "Final (Estable)"
        else:
            title = f"Paso {i}"
        plot_letter_pattern(step_2d, ax=axes[i + 2], title=title)
    
    fig.suptitle(f"{title_prefix} - Proceso de Recall", fontsize=14)
    plt.tight_layout()
    plt.show()

def find_spurious_state(network, stored_patterns, n_neurons, max_tries=1000):
    """
    Busca un estado espúreo generando patrones aleatorios muy ruidosos.
    
    Parameters:
    - network: Red de Hopfield ya entrenada
    - stored_patterns: Array de patrones almacenados (p x N)
    - n_neurons: Número de neuronas en la red
    - max_tries: Máximo número de intentos para encontrar un estado espúreo
    """
    print(f"Buscando estado espúreo con hasta {max_tries} intentos...")
    
    for attempt in range(max_tries):
        # Generar un patrón de consulta aleatorio (muy ruidoso)
        random_pattern = np.random.choice([-1, 1], size=n_neurons)
        
        # Obtener el estado final usando recall
        recall_steps = network.recall(random_pattern, max_iter=50, async_update=False)
        final_state = recall_steps[-1]
        
        # Verificar si es espúreo
        is_spurious = True
        
        # Chequear contra todos los patrones almacenados
        for stored_pattern in stored_patterns:
            if np.array_equal(final_state, stored_pattern):
                is_spurious = False
                break
        
        # Si no es igual a ningún patrón almacenado, chequear contra los inversos
        if is_spurious:
            for stored_pattern in stored_patterns:
                if np.array_equal(final_state, -stored_pattern):
                    is_spurious = False
                    break
        
        # Si encontramos un estado espúreo, reportarlo
        if is_spurious:
            print(f"¡Estado espúreo encontrado en el intento {attempt + 1}!")
            print(f"Convergió en {len(recall_steps)-1} pasos")
            
            # Mostrar la visualización del proceso
            plot_recall_steps(recall_steps, random_pattern, random_pattern, 
                            "Estado Espúreo")
            
            # Calcular la energía del estado espúreo
            energy = network.compute_energy(final_state)
            print(f"Energía del estado espúreo: {energy:.4f}")
            
            return final_state
    
    print(f"No se encontró ningún estado espúreo después de {max_tries} intentos.")
    return None

def main():
    # Lista de 4 letras a almacenar según consigna
    # letters_to_store = ['A', 'C', 'J', 'P'] # Conjunto aleatorio sin criterio de ortogonalidad
    # letters_to_store = ['Z', 'C', 'V', 'P'] # Muestra como un conjunto mas ortogonal es mas estable.


    letters_to_store = ['O', 'Q', 'G', 'C'] # De los menos ortogonales
    # letters_to_store = ['A', 'S', 'D', 'Y'] # De los mas ortogonales

    noise_level = 0.0  # 25% de ruido
    
    # Cargar las letras del JSON
    with open(DATASET_PATH, 'r') as f:
        dst = json.load(f)

    # plot_all_letters(dst['letters'])

    # for l in letters_to_store:
    #     plot_letter_pattern(dst['letters'][l], l)
    #     plt.show()
    
    # Generar patrones de entrenamiento
    patterns = genertate_letter_group(dst['letters'], letters_to_store)
    n_patterns, n_neurons = patterns.shape
    print(f'Number of patterns= {n_patterns}')
    print(f'N neurons= {n_neurons}')
    print(f'Letters stored: {letters_to_store}')

    #Calculate and plot similarity/orthogonality between letters
    sim_mat = similarity_arr(patterns)
    n = len(sim_mat)
    
    # Exclude diagonal (self-similarities)
    mask = ~np.eye(n, dtype=bool)
    off_diagonal_similarities = sim_mat[mask]
    print(off_diagonal_similarities)
    avg_simmilarity =np.abs(off_diagonal_similarities).mean()
    print(f"Avg simmilarity = {avg_simmilarity}")
    plot_similarity_heatmap(sim_mat, letters_to_store)
    
    # Instanciar y entrenar la red de Hopfield
    network = HopfieldNetwork(patterns.shape[1])
    network.train(patterns)
    print("\nRed de Hopfield entrenada exitosamente!\n")
    
    # Bucle sobre cada uno de los 4 patrones almacenados
    for i, letter in enumerate(letters_to_store):
        print(f"\n--- Procesando letra '{letter}' ---")
        
        # Patrón original
        original_pattern = patterns[i]
        
        # Generar versión ruidosa usando add_noise
        noisy_pattern = add_noise(original_pattern, noise_level)
        
        # Calcular porcentaje de bits cambiados
        changed_bits = np.sum(original_pattern != noisy_pattern)
        print(f"Bits cambiados: {changed_bits}/{n_neurons} ({changed_bits/n_neurons*100:.1f}%)")
        
        # Llamar a recall con el patrón ruidoso
        recall_steps = network.recall(noisy_pattern, max_iter=50, async_update=False)
        print(f"Converged in {len(recall_steps)-1} steps")
        
        # Mostrar la secuencia completa usando plot_recall_steps
        plot_recall_steps(recall_steps, original_pattern, noisy_pattern, f"Letra '{letter}'")
        
        # Verificar si se recuperó correctamente
        final_pattern = recall_steps[-1]
        if np.array_equal(final_pattern, original_pattern):
            print(f"✓ Letra '{letter}' recuperada correctamente")
        else:
            print(f"✗ Letra '{letter}' NO recuperada correctamente")
            diff_bits = np.sum(final_pattern != original_pattern)
            print(f"  Diferencias: {diff_bits}/{n_neurons} bits")
    
    # --- Punto 2.1.b: Búsqueda de Estado Espúreo ---
    print("\n" + "="*50)
    print("--- Punto 2.1.b: Búsqueda de Estado Espúreo ---")
    print("="*50)
    
    # Encontrar un estado espureo
    noise_lvl = 0.4
    og_pattern = patterns[1]
    noisy_pattern = add_noise(og_pattern, noise_lvl)
    recall_steps = network.recall(noisy_pattern, max_iter=50, async_update=False)
    print(f"Converged in {len(recall_steps)-1} steps")
    plot_recall_steps(recall_steps, og_pattern, noisy_pattern, "Estado espureo")

    # spurious_state = find_spurious_state(network, patterns, n_neurons, max_tries=1000)
    
    # if spurious_state is not None:
    #     print("\n¡Análisis del estado espúreo completado!")
    #     print("Un estado espúreo es un mínimo local de la función de energía")
    #     print("que no corresponde a ninguno de los patrones almacenados.")
    # else:
    #     print("\nNo se pudo encontrar un estado espúreo en los intentos realizados.")
    #     print("Esto puede indicar que la red tiene buena capacidad de almacenamiento")
    #     print("para los 4 patrones dados, o que se necesitan más intentos.")

if __name__ == '__main__':
    # Fijar semilla para reproducibilidad
    np.random.seed(42)
    main()
