import torch
from torch.utils.data import DataLoader
import os

# Set mode for testing
os.environ['MODE'] = 'test'
import config
import dataset
import model
import json
from utils.criteria import SSIM3D  # Assuming SSIM3D is defined in utils.criteria


def load_checkpoint(model_load, checkpoint_path):
    # Load a checkpoint into the model
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model_load.load_state_dict(checkpoint["state_dict"])
    return model_load


def load_test_dataset():
    # "\"Load and prepare the test dataset
    test_dataset = dataset.TestDataset(config.image_dir, config.label_dir)  # Adjust as per your dataset class
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False)  # Adjust batch_size and other parameters as needed
    return test_loader


def evaluate_model(test_loader, model_eval, device):
    # "\"\"Evaluate the model on the test dataset.\"\"\"
    model_eval.eval()
    ssim_model = SSIM3D().to(device)  # SSIM model for evaluation
    test_ssim_scores = []
    with torch.no_grad():
        for data in test_loader:
            gt = data["gt"].to(device=config.device, non_blocking=True)
            lr = data["lr"].to(device=config.device, non_blocking=True)
            outputs = model_eval(lr)
            ssim_score = ssim_model(outputs, gt)  # Assuming ground truth is inputs
            test_ssim_scores.append(ssim_score.item())
    return test_ssim_scores


def main():
    # Initialize model
    convLSTM_model = model.__dict__[config.d_arch_name](input_dim=config.input_dim,
                                                        hidden_dim=config.hidden_dim,
                                                        kernel_size=config.kernel_size,
                                                        output_size=config.output_size).to(config.device)

    # Load model checkpoint
    convLSTM_model = load_checkpoint(convLSTM_model, config.model_path)

    # Prepare test dataset
    test_loader = load_test_dataset()

    # Evaluate the model
    test_ssim_scores = evaluate_model(test_loader, convLSTM_model, config.device)

    # Log and save the test results
    test_metrics = {
        "test_ssim_scores": test_ssim_scores,
        "average_test_ssim": sum(test_ssim_scores) / len(test_ssim_scores)
    }
    print(test_metrics)
    results_file = os.path.join("results", f"{config.exp_name}_test_results.json")
    with open(results_file, 'w') as f:
        json.dump(test_metrics, f)

    print(f"Test evaluation completed. Results saved to {results_file}")
