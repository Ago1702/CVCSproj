import os
import gdown
import zipfile
from data.datasets import DirectorySequentialDataset
from data.dataloader import TransformDataLoader
from data.datasets import DirectoryRandomDataset
from utils.transform import RandomTransform
import random
import time
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
import torch
from models.nets import SuperEnsemble
from utils.helpers import state_dict_adapter
def tensor_to_image(tensor: torch.Tensor):
    # Clamp values to be in the range [0, 1]
    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor.squeeze(0)
    print(tensor.shape)
    # Convert tensor from (C, H, W) to (H, W, C)
    tensor = tensor.permute(1, 2, 0)  # If the tensor has shape (C, H, W)

    # Convert to NumPy array and scale to [0, 255]
    tensor = tensor.mul(255).byte().cpu().numpy()

    # Convert NumPy array to a PIL image
    image = Image.fromarray(tensor)
    return image


# Function to update the image after the NEXT button is pressed
def update_image():
    global test_dataloader, label, model_tensor,action_var

    # Hide solution labels when "NEXT" is pressed
    solution_label.grid_forget()
    model_solution_label.grid_forget()

    try:
        image_tensor, labels = next(test_dataloader)  # Get one image from the DataLoader

        # Convert the tensor to a PIL image
        random_number = random.randint(0, 1)
        model_tensor = image_tensor[random_number,:,:,:].unsqueeze(0).clone()
        image = tensor_to_image(image_tensor[random_number])
        label = labels[random_number].item()
        # Resize the image to fit within the label's size
        label_width = image_space.winfo_width()  # Get label width in pixels
        label_height = image_space.winfo_height()  # Get label height in pixels

        image = image.resize((label_width, label_height))

        # Convert the PIL image to a format suitable for tkinter
        photo = ImageTk.PhotoImage(image)

        # Update the image displayed in the label
        image_space.configure(image=photo)
        image_space.image = photo  # Keep a reference to the image
        action_var.set(2)

    except StopIteration:
        print("End of DataLoader reached!")


def show_solution():
    global label, model, you_score, model_score, action_var
    # Randomly choose between "IT WAS REAL!" or "IT WAS FAKE!"
    solution = "IT WAS REAL" if label == 0 else "IT WAS FAKE"
    print(label)
    # Set the color based on the solution text
    if solution == "IT WAS REAL!":
        solution_label.config(text=solution, fg="green")  # Green for "REAL"
    else:
        solution_label.config(text=solution, fg="red")  # Red for "FAKE"

    # Display the solution label
    solution_label.grid(row=2, column=0, padx=10, pady=10)

    # Randomly choose "MODEL SAID: REAL" or "MODEL SAID: FAKE!"
    with torch.no_grad():
        output = int(torch.round(torch.sigmoid(model.forward(model_tensor))).item())
        #output = model.forward(model_tensor)
        #print('solution is' + str(output))
    if output == 1:
        model_solution = "MODEL SAID: FAKE!"
        if label == 1:
            model_score+=1
    elif output == 0:
        model_solution = "MODEL SAID: REAL!"
        if label == 0:
            model_score+=1
    if action_var.get() == label:
        you_score+=1
    update_leaderboard()
    # Display the "MODEL SAID" label
    model_solution_label.config(text=model_solution)
    model_solution_label.grid(row=3, column=0, padx=10, pady=10)
    action_var.set(2)


# Function to update leaderboard scores
def update_leaderboard():
    global you_score, model_score, total_label

    # Update leaderboard labels
    you_label.config(text=f"YOU: {you_score}")
    model_label.config(text=f"MODEL: {model_score}")
    total_label.config(text=f"TOTAL: {total_score}")

# Function to create the basic GUI
def create_gui():
    global root, image_space, test_dataloader, solution_label, model_solution_label
    global you_label, model_label, you_score, model_score, action_var, total_score

    # Create the main window
    root = tk.Tk()
    root.title("Image Labeling GUI")

    # Set window size (optional)
    root.geometry("1200x700")

    # Create the frame for buttons and image
    frame = tk.Frame(root)
    frame.pack(side="left", expand=True)

    # Create the image display space
    image_space = tk.Label(frame, width=400, height=400, relief="solid")
    image_space.grid(row=0, column=1, padx=10, pady=10)

    # Variable to track the selected button (Real or Fake)
    action_var = tk.IntVar(value=2)  # Default to "None"

    # Create the REAL and FAKE radiobuttons
    real_button = tk.Radiobutton(frame, text="REAL", variable=action_var, value=0)
    fake_button = tk.Radiobutton(frame, text="FAKE", variable=action_var, value=1)

    # Create the NEXT button
    next_button = tk.Button(frame, text="NEXT", width=15, command=update_image)

    # Create the SOLUTION button
    solution_button = tk.Button(frame, text="SOLUTION", width=15, command=show_solution)

    # Arrange buttons in a grid
    real_button.grid(row=1, column=0, padx=10, pady=10)
    fake_button.grid(row=1, column=1, padx=10, pady=10)
    next_button.grid(row=1, column=2, padx=10, pady=10)
    solution_button.grid(row=1, column=3, padx=10, pady=10)

    # Create the labels for solution and model predictions (initially empty)
    solution_label = tk.Label(frame, text="", font=("Helvetica", 12))
    model_solution_label = tk.Label(frame, text="", font=("Helvetica", 10), fg="black")

    # Create a frame for the leaderboard on the right side
    leaderboard_frame = tk.Frame(root)
    leaderboard_frame.pack(side="right", padx=10, pady=10)

    # Initialize leaderboard scores
    you_score = 0
    model_score = 0
    total_score = 0

    # Create leaderboard table
    leaderboard_label = tk.Label(leaderboard_frame, text="LEADERBOARD", font=("Helvetica", 14))
    leaderboard_label.grid(row=0, column=0, columnspan=2)

    you_label = tk.Label(leaderboard_frame, text=f"YOU: {you_score}", font=("Helvetica", 12))
    you_label.grid(row=1, column=0, padx=10, pady=5)

    model_label = tk.Label(leaderboard_frame, text=f"MODEL: {model_score}", font=("Helvetica", 12))
    model_label.grid(row=2, column=0, padx=10, pady=5)

    total_label = tk.Label(leaderboard_frame, text=f"TOTAL: {total_score}", font=("Helvetica", 12))
    total_label.grid(row=3, column=0, padx=10, pady=5)
    # Run the GUI
    update_image()
    root.mainloop()


if __name__ == '__main__':
    dataset_link = 'https://drive.google.com/uc?id=19nrUNb4U3PCgCDTUGFYNPlK1ZHZwI61S'

    #weights_link = 'https://drive.google.com/uc?id=150nfmRGFLTWo8uQ8W1t6cg73sZotOpXK' #resnet_50
    weights_link = 'https://drive.google.com/uc?id=1dzsHe-BNYUIWzCzJN8Ktskmakh318TfL'

    zip_dataset_folder = 'zip_dataset'
    test_dataset_folder = 'test_dataset'
    zip_name = 'test_dataset.zip'
    weights_folder = 'weights'
    weights_name = 'weights.pth'

    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)
        gdown.download(
            url=weights_link,
            output=weights_folder
        )
        downloaded_filename = os.listdir(weights_folder)[0]
        os.rename(
            os.path.join(
                weights_folder,
                downloaded_filename
            ),
            os.path.join(
                weights_folder,
                weights_name))
    os.makedirs(zip_dataset_folder, exist_ok=True)
    if not os.path.exists(os.path.join(zip_dataset_folder, zip_name)):
        gdown.download(
            url=dataset_link,
            output=zip_dataset_folder
        )
        downloaded_filename = os.listdir(zip_dataset_folder)[0]
        os.rename(
            os.path.join(
                zip_dataset_folder,
                downloaded_filename
            ),
            os.path.join(
                zip_dataset_folder,
                zip_name))
        with zipfile.ZipFile(os.path.join(zip_dataset_folder, zip_name)) as zip_ref:
            zip_ref.extractall()
    test_dataset = DirectoryRandomDataset(dir='test')
    test_dataloader = TransformDataLoader(
        cropping_mode=RandomTransform.GLOBAL_CROP,
        dataset=test_dataset,
        batch_size=2,
        num_workers=0,
        dataset_mode=DirectoryRandomDataset.COUP,
        probability=0.0,
        center_crop=True
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SuperEnsemble(load_checkpoints=False)
    model.eval()
    model.load_state_dict(state_dict_adapter(torch.load(os.path.join(weights_folder,weights_name),weights_only=False,map_location=device)['model'],'module.',''))
    model = model.to(device)
    create_gui()

