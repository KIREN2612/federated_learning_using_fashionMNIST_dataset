
# Federated Learning with FashionMNIST Dataset

This repository demonstrates a simple federated learning example using the FashionMNIST dataset. It includes:

- A Jupyter notebook (`federated_learning.ipynb`) to run the federated training simulation using Flower and PyTorch.
- A trained model saved as `federated_fashionmnist.pt`.
- A Streamlit app (`app.py`) to load the trained model and test predictions on uploaded images.
- A sample test image `sample_image.png` from the FashionMNIST dataset for quick testing.

---

## How to Use This Repository

### 1. Run Federated Training in Google Colab (Recommended)

- Open the `federated_learning.ipynb` notebook in [Google Colab](https://colab.research.google.com/).
- Run all the cells to simulate federated training on FashionMNIST.
- At the end of the training, the model weights will be saved locally as `federated_fashionmnist.pt`.
- Download the `federated_fashionmnist.pt` file to your local machine.

---

### 2. Run the Streamlit App Locally to Test the Model

- Make sure you have Python installed with required dependencies:
```

pip install -r requirements.txt

```
- Place the downloaded `federated_fashionmnist.pt` file in the `FashionMNIST` folder (or update the path accordingly in `app.py`).
- Run the Streamlit app:
```

streamlit run FashionMNIST/app.py

```
- The app will launch in your browser.
- Use the provided `sample_image.png` in the repo or upload your own grayscale FashionMNIST-like images to test the model's prediction.

---

### 3. If You Do NOT Want to Run the Notebook

- The repo already contains a pretrained model file: `FashionMNIST/federated_fashionmnist.pt`.
- Download this `.pt` file directly.
- Update the path to the `.pt` file in `app.py` if necessary.
- Run the Streamlit app as explained above.

---

## Repository Structure

```

.
├── FashionMNIST/
│   ├── app.py                     # Streamlit app to test the model
│   ├── federated\_fashionmnist.pt  # Pretrained model weights
│   ├── sample\_image.png           # Sample FashionMNIST image for testing
│   └── ...
├── federated\_learning.ipynb       # Notebook to run federated training in Colab
├── requirements.txt               # Python dependencies
└── README.md                     # This file

```

---

## Notes

- The model was trained on FashionMNIST dataset using federated learning with Flower.
- The app expects grayscale images sized 28x28 pixels similar to FashionMNIST.
- You can use the sample image provided or your own images preprocessed accordingly.
- Feel free to customize the model or experiment with federated learning parameters by modifying the notebook.

---

If you face any issues or have questions, feel free to open an issue on this repository!

---

Happy federated learning! 🚀
```

---

