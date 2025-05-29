#!/usr/bin/env python3
import os
import json
import matplotlib.pyplot as plt

def main():
    # locate loss.json in the same directory as this script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    loss_path = os.path.join(script_dir, 'loss.json')

    # load the JSON data
    with open(loss_path, 'r') as f:
        loss_data = json.load(f)

    # Prepare the plot
    plt.figure(figsize=(8, 5))

    # If it's a dict of lists, plot each key separately
    if isinstance(loss_data, dict):
        for key, values in loss_data.items():
            plt.plot(values, label=key)
    # If it's a simple list of numbers
    elif isinstance(loss_data, list):
        plt.plot(loss_data, label='loss')
    else:
        raise ValueError(f"Unexpected JSON structure: {type(loss_data)}")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
