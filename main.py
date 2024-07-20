from train import train_model
from evaluate import evaluate_model
import os
from model import UNet

def main():
    data_dir = '/Users/udisharora/Desktop/train2'  # Directory where the training data is stored
    test_dir = '/Users/udisharora/Desktop/test'     # Directory where the test data is stored

    # Train the model
    model = train_model(data_dir)

    # Test the model
    model_dir = '/Users/udisharora/project/model'    # Specify your model directory
    model_filename = 'my_model.pth'
    model_path = os.path.join(model_dir, model_filename)
    model = UNet()  # Create a new model instance for evaluation
    model.load_state_dict(torch.load(model_path))
    avg_accuracy, avg_dice = evaluate_model(model, test_dir)

    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average Dice Coefficient: {avg_dice:.4f}')

if __name__ == "__main__":
    main()

