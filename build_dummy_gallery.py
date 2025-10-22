import os, shutil
from pydicom.data import get_testdata_files
from classifier import FeatureGallery
from pathlib import Path

# المسار الأساسي للمشروع
base = Path("E:/Task 2")
gallery_root = base / "gallery_samples"
gallery_root.mkdir(exist_ok=True)

# نعمل فولدرين تجريبيين: "Knee" و "Brain"
src = get_testdata_files("CT_small.dcm")[0]
(knee_dir := gallery_root/"Knee").mkdir(exist_ok=True)
(brain_dir := gallery_root/"Brain").mkdir(exist_ok=True)

for i in range(5):
    shutil.copy(src, knee_dir/f"ct_knee_{i}.dcm")
    shutil.copy(src, brain_dir/f"ct_brain_{i}.dcm")

# نبدأ نبني الجاليري
g = FeatureGallery()
g.build_from_folder(str(gallery_root))

print("✅ Dummy gallery built successfully at:", gallery_root)
