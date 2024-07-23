import torch
import numpy as np
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import streamlit as st

st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache(allow_output_mutation=True)
def load_model():
    return torch.load("dretino.pt")


model = load_model()


st.write(
    """

    # Web Demo for Self-Supervised Learning on Diabetic Retinopathy Images

    ### [GitHub](https://github.com/Maahi10001/D-Retino)

    """
)

transforms = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=(0.5211, 0.2514, 0.0809),
            std=(0.2653, 0.1499, 0.0861),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

file = st.file_uploader("", type=["png", "jpg"])


with open("./assets/0.png", "rb") as file1:
    btn = st.download_button(
        label="Download Sample-1 Image", data=file1, file_name="0.png", mime="image/png"
    )

with open("./assets/1.png", "rb") as file2:
    btn = st.download_button(
        label="Download Sample-2 Image", data=file2, file_name="1.png", mime="image/png"
    )


@st.cache(allow_output_mutation=True)
def generate_cam():
    target_layer = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
    return cam


cam = generate_cam()


def deprocess_image(img):
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return img


def dretino(img):
    img = transforms(image=img)["image"]
    img = torch.tensor(img)

    img = torch.unsqueeze(img, 0)

    model.eval()
    grayscale_cam = cam(input_tensor=img)

    rgb_img = np.transpose(deprocess_image(img[0].numpy()), (1, 2, 0))
    viz = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)

    logits = model(img)

    preds = torch.nn.Softmax(dim=1)(logits.data)[0]
    res = {"No": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Proliferative": 0}
    for idx, key in enumerate(res.keys()):
        res[key] = preds[idx]

    return res, viz


if file is None:
    st.text("Please upload an image file")
else:
    st.text("Uploaded Image")
    image = np.asarray(Image.open(file))
    if image.shape[-1] == 4:
        image = image.convert("RGB")
    st.image(image, use_column_width=True)
    res, viz = dretino(image)
    fig, ax = plt.subplots()
    ax.barh(list(res.keys()), np.array(list(res.values())))
    ax.grid(visible=True)
    st.text("Predicted Result")
    res_ = {key: float("{:.2f}".format(item.item())) for key, item in res.items()}
    st.write(res_)
    st.write(fig)
    st.text("GradCam Output")
    st.write(
        """
    For more information about GradCam visit 
    [GradCam-Book](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)
    """
    )
    st.image(viz, use_column_width=True)
