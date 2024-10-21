import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
import os

def convert_image_to_rgb(image):
    if image.mode != 'RGB':  # Se l'immagine ha il quarto canale troll viene convertita in RGB
        image = image.convert('RGB')
    return image

def get_embedding(image_tensor, model, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        embedding = model(image_tensor)
        return embedding.flatten()


def show_images_side_by_side(fake_image_path, real_image_path):
    fake_image = Image.open(fake_image_path)
    real_image = Image.open(real_image_path)

    save_dir = "plot-comparison"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Mostra l'immagine fake
    axes[0].imshow(fake_image)
    axes[0].set_title('Fake Image')
    axes[0].axis('off')

    # Mostra l'immagine reale
    axes[1].imshow(real_image)
    axes[1].set_title('Most Similar Real Image')
    axes[1].axis('off')

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(save_dir, f"comparison_{timestamp}.png")
    plt.savefig(save_path)

    # Visualizzo il plot che si chiude automaticamente dopo 5 secondi
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()

if __name__ == '__main__':
    # trasforma ogni immagine di input in 224x224 perché la Resnet50 le vuole così
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    """                             !!! Percorso dei dati nel mio pc, va cambiato!!!                                """
    """                                                    |                                                        """
    """                                                    |                                                        """
    """                                                    V                                                        """
    dataset = datasets.ImageFolder('C:\\Users\\Filip\\Desktop\\imm\\test', transform=transform)

    # il Dataloader dava problemi con le label, non ho capito perché, chatgpt ha fatto sto numero per risolvere
    real_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]
    fake_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]

    real_images = [dataset.samples[i][0] for i in real_indices]
    fake_images = [dataset.samples[i][0] for i in fake_indices]

    # DataLoader
    real_loader = DataLoader([dataset[i] for i in real_indices], batch_size=1, shuffle=False)
    fake_loader = DataLoader([dataset[i] for i in fake_indices], batch_size=1, shuffle=False)

    # Scarico i pesi già allenati della resnet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # l'ultimo layer va tolto perché prendiamo solo l'embedding
    model.fc = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for _ in range(100):
        # Seleziono un'immagine fake casuale
        random_fake_img_path = random.choice(fake_images)
        """
        Carico e trasformo l'immagine fake, ovvero tolgo il canale alpha se presente e poi viene trasformata in 224x224
        """
        fake_image = Image.open(random_fake_img_path)
        fake_image = convert_image_to_rgb(fake_image)
        fake_image_tensor = transform(fake_image).to(device)

        # la do in input alla resnet
        fake_embedding = get_embedding(fake_image_tensor, model, device)

        # Confronto embedding fake con quelli reali e trovo il più simile
        best_similarity = -1
        best_real_image_path = None

        """ Quetsa è la parte che sicuramente rallenta più di tutto il processo di retrieval, infatti va a calcolare 
                    l'embedding di ogni singola immagine reale per poi trovare la più simile 
        """
        for real_image_path in real_images:
            real_image = Image.open(real_image_path)
            real_image = convert_image_to_rgb(real_image)
            real_image_tensor = transform(real_image).to(device)

            real_embedding = get_embedding(real_image_tensor, model, device)

            # Se non ci piace possiamo provare a cambiare la funzione di similarità, a lollone piaceva il cosine
            similarity = torch.nn.functional.cosine_similarity(fake_embedding, real_embedding, dim=0).item()

            if similarity > best_similarity:
                best_similarity = similarity
                best_real_image_path = real_image_path

        # Plotto e salvo il plot
        local_time = time.localtime()
        show_images_side_by_side(random_fake_img_path, best_real_image_path)
