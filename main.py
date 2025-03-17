import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [perceptron|nn]")
        sys.exit(1)

    option = sys.argv[1].lower()
    if option == "perceptron":
        pass # do something!
    elif option == "nn":
        pass # do something!
    else:
        print("Unknown option. Please choose 'perceptron' or 'nn'.")
        sys.exit(1)

if __name__ == "__main__":
    main()