import sys
from INF2475.perceptron import run_perceptron_example
from INF2475.nn import run_nn_example

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [perceptron|nn]")
        sys.exit(1)

    option = sys.argv[1].lower()
    if option == "perceptron":
        run_perceptron_example()
    elif option == "nn":
        run_nn_example()
    else:
        print("Unknown option. Please choose 'perceptron' or 'nn'.")
        sys.exit(1)

if __name__ == "__main__":
    main()