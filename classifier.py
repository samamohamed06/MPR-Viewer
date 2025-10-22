# classifier.py
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
from sklearn.neighbors import NearestNeighbors
import pydicom

# ------- feature extractor (ResNet18 pretrained) -------
_device = torch.device("cpu")  # خليه CPU افتراضي، لو عندك GPU غيره لـ "cuda"
_model = None
_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def _init_model():
    global _model
    if _model is None:
        m = models.resnet18(pretrained=True)
        # remove final classification layer -> use penultimate activations
        m = torch.nn.Sequential(*(list(m.children())[:-1]))
        m.eval().to(_device)
        _model = m

def image_to_feature(pil_img):
    """Convert PIL image -> feature vector (1D numpy)"""
    _init_model()
    x = _transform(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model(x)  # shape (1, 512, 1, 1)
    feat = feat.cpu().squeeze().numpy().reshape(-1)
    # normalize
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat

def dicom_to_pil(ds):
    """Convert pydicom Dataset to PIL.Image (grayscale->RGB)"""
    arr = ds.pixel_array.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    else:
        arr = arr * 0
    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    # convert to RGB (3 channels) because ResNet expects 3 channels
    return img.convert("RGB")

# ------- gallery building & k-NN predict -------
class FeatureGallery:
    def __init__(self):
        self.features = None  # numpy (N, F)
        self.labels = []      # list length N
        self.knn = None

    def build_from_folder(self, labeled_root):
        """
        labeled_root/
            organA/
                img1.png / dcm1.dcm / ...
            organB/
                ...
        The function reads images (png/jpg) or DICOM (.dcm) and computes features.
        """
        feats = []
        labs = []
        for organ in os.listdir(labeled_root):
            organ_dir = os.path.join(labeled_root, organ)
            if not os.path.isdir(organ_dir):
                continue
            for f in os.listdir(organ_dir):
                fp = os.path.join(organ_dir, f)
                try:
                    if f.lower().endswith(".dcm"):
                        ds = pydicom.dcmread(fp)
                        pil = dicom_to_pil(ds)
                    else:
                        pil = Image.open(fp).convert("RGB")
                    feat = image_to_feature(pil)
                    feats.append(feat)
                    labs.append(organ)
                except Exception as e:
                    print("Gallery read failed:", fp, e)
        if len(feats) == 0:
            raise RuntimeError("No images found in gallery folder")
        self.features = np.stack(feats, axis=0)
        self.labels = labs
        # build knn (k=3)
        self.knn = NearestNeighbors(n_neighbors=3, metric='cosine').fit(self.features)
        print(f"Built gallery with {len(self.labels)} examples across {len(set(self.labels))} labels.")

    def predict(self, pil_img):
        """predict label for a PIL image: returns (top_label, confidence, neighbors)"""
        if self.knn is None:
            raise RuntimeError("Gallery not built")
        feat = image_to_feature(pil_img).reshape(1, -1)
        dists, idxs = self.knn.kneighbors(feat, n_neighbors=3)
        # cosine distances -> smaller is more similar; compute votes
        idxs = idxs[0]
        neigh_labels = [self.labels[i] for i in idxs]
        # simple voting
        from collections import Counter
        ctr = Counter(neigh_labels)
        top_label, count = ctr.most_common(1)[0]
        # confidence proxy: 1 - average distance
        avg_dist = float(np.mean(dists))
        confidence = float(max(0.0, 1.0 - avg_dist))
        return top_label, confidence, neigh_labels

# helper function for quick prediction on a dicom dataset
def predict_from_dicom_dataset(dsets, gallery: FeatureGallery):
    """
    dsets: list of pydicom Dataset or a single ds
    strategy: pick central axial slice (middle of volume) as PIL, predict
    """
    if isinstance(dsets, list):
        # try to find one ds that has pixel_array shape (H,W) and use middle slice if series
        # if many datasets, pick middle InstanceNumber
        # simplest: take dataset at index len//2
        ds0 = dsets[len(dsets)//2]
    else:
        ds0 = dsets
    pil = dicom_to_pil(ds0)
    return gallery.predict(pil)
