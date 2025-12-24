import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

from model_definitions import ModelA_CNN, ModelB_MLP

st.title('Assignment # 5 Deep Learning')
st.subheader('Submitted by Jawad Amin')
st.divider()

device = torch.device("cpu")

# Load best model name and weights
try:
    with open("best_model_name.txt", "r") as f:
        best_name = f.read().strip()
except FileNotFoundError:
    best_name = "ModelA_CNN"  # fallback if training not run yet

model = ModelA_CNN() if best_name == "ModelA_CNN" else ModelB_MLP()
try:
    state = torch.load("mnist_model.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    st.success(f"Loaded best model: {best_name}")
except FileNotFoundError:
    st.warning("mnist_model.pth not found. Please run training.py first.")
    model.eval()

# Preprocessing (for dataset and uploads)
dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

upload_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# MNIST test dataset for demo predictions
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=dataset_transform)

test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=dataset_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

st.write("Training samples:", len(train_dataset))
st.write("Test samples:", len(test_dataset))


modelA = ModelA_CNN()
modelB = ModelB_MLP()

st.subheader("Model Architectures")

st.write("Model A (CNN):")
st.text(modelA)

st.write("Model B (MLP):")
st.text(modelB)

st.subheader("Try model on 5 random MNIST test images")
indices = random.sample(range(len(test_dataset)), 5)
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, idx in enumerate(indices):
    img, label = test_dataset[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].set_title(f"True: {label}\nPred: {pred}")
    axes[i].axis("off")
st.pyplot(fig)

st.divider()
st.subheader("Upload your own digit image")

uploaded_file = st.file_uploader("Upload PNG/JPG of a single digit (white on dark works best)",
                                 type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=150)

    tensor = upload_transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
        pred = int(probs.argmax())
        confidence = probs[pred] * 100
    st.write(f"Predicted Digit: **{pred}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
    st.bar_chart(probs)
