import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_mem_log(log_file):
    """
    Parse memory log file and extract memory usage data grouped by Python file name using pandas.
    """
    # Create empty lists to store parsed data
    files = []
    first_mem = []
    last_mem = []
    total_mem = []
    last_file_name = ""
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract filename, free and total memory using regex
            match = re.match(r'([\w/._\-]+\.py)::[\w:]+ on cuda:[\d]+: Free: ([\d.]+), Total: ([\d.]+)', line)
            if match:
                file_name = match.group(1)
                first_memory = float(match.group(2))
                total_memory = float(match.group(3))
                if file_name != last_file_name:
                    last_mem.append(first_memory)
                files.append(match.group(1))
                first_mem.append(first_memory)
                total_mem.append(total_memory)
                last_file_name = file_name
    last_mem.append(first_memory)

def plot_memory_usage(files, free_memory, total_memory, min_free, max_free, memory_range):
    """
    Create a matplotlib chart showing memory usage by Python file.
    """
    # Simplify filenames for the x-axis (remove path)
    short_filenames = [f.split('/')[-1] for f in files]
    
    x = np.arange(len(files))  # the label locations
    width = 0.4  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot memory range as bars
    rects = ax.bar(x, memory_range, width, label='Free Memory Range (GB)', alpha=0.5, color='green')
    
    # Plot free and total memory as lines
    ax.plot(x, free_memory, 'b-', marker='o', linewidth=2, label='Free Memory (GB)')
    ax.plot(x, total_memory, 'r-', marker='s', linewidth=2, label='Total Memory (GB)')
    
    # Create a twin axis on the right for min-max bars
    ax2 = ax.twinx()
    
    # Get the max height of the memory range bars to scale the right axis
    max_bar_height = max(memory_range) if memory_range else 1.0
    
    # Scale factor to adjust min-max bars to fit within a reasonable range
    # This maps the memory values to the range of the right axis
    scale_factor = max_bar_height / (max(max_free) - min(min_free)) if max_free and min_free else 1.0
    
    # Add min-max markers for free memory on the right axis
    for i, (min_val, max_val) in enumerate(zip(min_free, max_free)):
        # Draw the min-max bars scaled to fit the right axis
        ax2.plot([i, i], [min_val, max_val], 'k-', linewidth=2)
        ax2.plot([i - 0.1, i + 0.1], [min_val, min_val], 'k-', linewidth=2)
        ax2.plot([i - 0.1, i + 0.1], [max_val, max_val], 'k-', linewidth=2)
    
    # Add some text for labels, title and custom x-axis tick labels
    ax.set_xlabel('Python Files')
    ax.set_ylabel('Memory (GB)')
    ax2.set_ylabel('Min-Max Memory Range (GB)')
    ax.set_title('Memory Usage by Python File')
    ax.set_xticks(x)
    ax.set_xticklabels(short_filenames, rotation=90)
    
    # Set right y-axis range based on min-max data
    # Set bottom slightly below minimum free memory and top slightly above maximum
    min_y_val = min(min_free) - 0.05 * (max(max_free) - min(min_free))
    max_y_val = max(max_free) + 0.05 * (max(max_free) - min(min_free))
    ax2.set_ylim(min_y_val, max_y_val)
    
    # Add legends
    ax.legend(loc='upper left')
    
    # Ensure the figure fits well
    fig.tight_layout()
    
    plt.savefig('memory_usage_chart.png')
    plt.show()

if __name__ == "__main__":
    files, free_memory, total_memory, min_free, max_free, memory_range = parse_mem_log('/workspace/Megatron-LM/mem.log')
    plot_memory_usage(files, free_memory, total_memory, min_free, max_free, memory_range)
