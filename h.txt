import os
import subprocess
import nibabel as nib
import numpy as np
from tkinter import Tk, filedialog
import shutil
import tempfile
import sys
import time
import atexit
from pathlib import Path
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed

TEMP_DIR = None

def cleanup_temp():
    """Clean temporary files on exit"""
    global TEMP_DIR
    if TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            time.sleep(0.5)
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        except:
            pass

atexit.register(cleanup_temp)

# FIXED: More flexible organ matching
MAIN_ORGANS = {
    'heart': {'name': 'Heart', 'expected_volume': (250, 600), 'priority': 10, 'weight': 1.0, 
              'aliases': ['heart', 'heart_myocardium', 'heart_atrium', 'heart_ventricle']},
    'liver': {'name': 'Liver', 'expected_volume': (1200, 2500), 'priority': 10, 'weight': 1.0,
              'aliases': ['liver']},
    'spleen': {'name': 'Spleen', 'expected_volume': (150, 400), 'priority': 9, 'weight': 0.95,
               'aliases': ['spleen']},
    'pancreas': {'name': 'Pancreas', 'expected_volume': (80, 180), 'priority': 9, 'weight': 0.95,
                 'aliases': ['pancreas']},
    'kidney_left': {'name': 'Left Kidney', 'expected_volume': (120, 250), 'priority': 8, 'weight': 0.90,
                    'aliases': ['kidney_left', 'kidney_l']},
    'kidney_right': {'name': 'Right Kidney', 'expected_volume': (120, 250), 'priority': 8, 'weight': 0.90,
                     'aliases': ['kidney_right', 'kidney_r']},
    'lung_left': {'name': 'Left Lung', 'expected_volume': (800, 3000), 'priority': 7, 'weight': 0.85,
                  'aliases': ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_left', 'lung_l']},
    'lung_right': {'name': 'Right Lung', 'expected_volume': (800, 3000), 'priority': 7, 'weight': 0.85,
                   'aliases': ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right', 'lung_right', 'lung_r']},
    'stomach': {'name': 'Stomach', 'expected_volume': (150, 500), 'priority': 8, 'weight': 0.90,
                'aliases': ['stomach']},
    'gallbladder': {'name': 'Gallbladder', 'expected_volume': (20, 80), 'priority': 6, 'weight': 0.75,
                    'aliases': ['gallbladder']},
    'esophagus': {'name': 'Esophagus', 'expected_volume': (30, 100), 'priority': 5, 'weight': 0.70,
                  'aliases': ['esophagus']},
    'duodenum': {'name': 'Duodenum', 'expected_volume': (40, 150), 'priority': 6, 'weight': 0.75,
                 'aliases': ['duodenum']},
    'colon': {'name': 'Colon', 'expected_volume': (200, 800), 'priority': 7, 'weight': 0.85,
              'aliases': ['colon']},
    'small_bowel': {'name': 'Small Bowel', 'expected_volume': (300, 1200), 'priority': 6, 'weight': 0.80,
                    'aliases': ['small_bowel', 'small_intestine']},
    'urinary_bladder': {'name': 'Bladder', 'expected_volume': (50, 500), 'priority': 7, 'weight': 0.85,
                        'aliases': ['urinary_bladder', 'bladder']},
    'prostate': {'name': 'Prostate', 'expected_volume': (15, 40), 'priority': 6, 'weight': 0.75,
                 'aliases': ['prostate']},
    'trachea': {'name': 'Trachea', 'expected_volume': (20, 80), 'priority': 5, 'weight': 0.70,
                'aliases': ['trachea']},
    'thyroid': {'name': 'Thyroid', 'expected_volume': (10, 30), 'priority': 6, 'weight': 0.75,
                'aliases': ['thyroid_gland', 'thyroid']},
    'adrenal_left': {'name': 'Left Adrenal', 'expected_volume': (3, 15), 'priority': 5, 'weight': 0.70,
                     'aliases': ['adrenal_gland_left', 'adrenal_left']},
    'adrenal_right': {'name': 'Right Adrenal', 'expected_volume': (3, 15), 'priority': 5, 'weight': 0.70,
                      'aliases': ['adrenal_gland_right', 'adrenal_right']},
}

def check_totalsegmentator():
    """Check TotalSegmentator installation"""
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return "TotalSegmentator"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "TotalSegmentator", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return f"{sys.executable} -m TotalSegmentator"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        import totalsegmentator
        return f"{sys.executable} -m TotalSegmentator"
    except ImportError:
        pass
    
    return None

def is_dicom_file_fast(filepath):
    """Fast DICOM file check"""
    try:
        filename = os.path.basename(filepath).lower()
        
        if filename.endswith(('.dcm', '.dicom')):
            return True
        
        if '.' not in filename and os.path.getsize(filepath) > 128:
            with open(filepath, 'rb') as f:
                header = f.read(132)
                return len(header) >= 132 and header[128:132] == b'DICM'
    except:
        pass
    
    return False

def extract_dicom_files_fast(source_folder):
    """Extract DICOM files with high speed"""
    print("Scanning for DICOM files...")
    
    dicom_files = []
    
    for root, dirs, files in os.walk(source_folder):
        dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('_')]
        
        for file in files:
            if file.startswith('.') or file.startswith('_'):
                continue
            
            filepath = os.path.join(root, file)
            if is_dicom_file_fast(filepath):
                dicom_files.append(filepath)
    
    if not dicom_files:
        return None, "ERROR: No DICOM files found"
    
    print(f"Found {len(dicom_files)} DICOM files")
    
    clean_dicom_folder = os.path.join(source_folder, "dicom_clean")
    os.makedirs(clean_dicom_folder, exist_ok=True)
    
    print(f"Copying files...")
    
    def copy_file(idx_path):
        idx, dicom_path = idx_path
        try:
            dest_name = f"{idx:05d}.dcm"
            dest_path = os.path.join(clean_dicom_folder, dest_name)
            shutil.copy2(dicom_path, dest_path)
            return True
        except:
            return False
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(copy_file, enumerate(dicom_files)))
    
    return clean_dicom_folder, f"Ready"

def select_folder_fixed():
    """Select folder with fixed dialog"""
    print("\nOpening folder selection dialog...")
    
    root = Tk()
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    root.update()
    
    folder_path = filedialog.askdirectory(
        title="Select DICOM Folder",
        parent=root
    )
    
    root.destroy()
    
    if not folder_path or folder_path.strip() == '':
        print("ERROR: No folder selected")
        return None, None
    
    print(f"Selected: {folder_path}")
    
    clean_folder, message = extract_dicom_files_fast(folder_path)
    
    if clean_folder:
        print(message)
        return folder_path, clean_folder
    else:
        print(message)
        return None, None

def match_organ_name(filename):
    """FIXED: Match filename to organ info with aliases"""
    filename_lower = filename.lower().replace('.nii.gz', '').replace('.nii', '')
    
    # Direct match first
    for key, info in MAIN_ORGANS.items():
        if key in filename_lower:
            return key, info
    
    # Try aliases
    for key, info in MAIN_ORGANS.items():
        for alias in info.get('aliases', []):
            if alias in filename_lower:
                return key, info
    
    return None, None

def list_output_files(output_dir):
    """DEBUG: List all output files"""
    print("\n🔍 DEBUG: Files found in output directory:")
    if not os.path.exists(output_dir):
        print("   Output directory does not exist!")
        return []
    
    files = [f for f in os.listdir(output_dir) if f.endswith(('.nii.gz', '.nii'))]
    
    if not files:
        print("   No .nii.gz or .nii files found!")
    else:
        for f in files:
            print(f"   - {f}")
    
    return files

def calculate_volume_score_enhanced(volume_cm3, expected_range, priority):
    """ULTRA ACCURATE volume scoring"""
    min_vol, max_vol = expected_range
    mid = (min_vol + max_vol) / 2
    
    if min_vol <= volume_cm3 <= max_vol:
        deviation = abs(volume_cm3 - mid) / (max_vol - min_vol)
        score = 1.0 - (deviation * 0.03)
    elif volume_cm3 < min_vol:
        ratio = volume_cm3 / min_vol
        if ratio >= 0.85:
            score = 0.85 + (ratio - 0.85) * 1.0
        elif ratio >= 0.70:
            score = 0.70 + (ratio - 0.70) * 1.0
        elif ratio >= 0.50:
            score = 0.50 + (ratio - 0.50) * 0.9
        elif ratio >= 0.35:
            score = 0.35 + (ratio - 0.35) * 0.7
        elif ratio >= 0.20:
            score = 0.20 + (ratio - 0.20) * 0.5
        else:
            score = ratio * 0.20
    else:
        excess_ratio = (volume_cm3 - max_vol) / max_vol
        if excess_ratio < 0.12:
            score = 0.99 - (excess_ratio * 0.3)
        elif excess_ratio < 0.30:
            score = 0.96 - (excess_ratio * 0.35)
        elif excess_ratio < 0.60:
            score = 0.88 - (excess_ratio * 0.35)
        elif excess_ratio < 1.0:
            score = 0.75 - (excess_ratio * 0.30)
        elif excess_ratio < 2.0:
            score = 0.60 - (excess_ratio * 0.20)
        else:
            score = max(0.10, 0.45 - excess_ratio * 0.15)
    
    priority_boost = (priority / 10.0) * 0.15
    score = min(1.0, score + priority_boost)
    
    return max(0.0, score)

def calculate_anatomical_position_score(data, img_shape, organ_name):
    """ULTRA STRICT anatomical position scoring"""
    if np.sum(data > 0) == 0:
        return 0.2
    
    organ_center = ndimage.center_of_mass(data > 0)
    z, y, x = organ_center
    img_z, img_y, img_x = img_shape
    
    z_ratio = z / img_z
    y_ratio = y / img_y
    x_ratio = x / img_x
    
    score = 0.2
    organ_lower = organ_name.lower()
    
    if 'heart' in organ_lower:
        if 0.28 < z_ratio < 0.52 and 0.38 < y_ratio < 0.62 and 0.42 < x_ratio < 0.58:
            score = 1.0
        elif 0.23 < z_ratio < 0.60 and 0.32 < y_ratio < 0.68 and 0.38 < x_ratio < 0.62:
            score = 0.85
        elif 0.18 < z_ratio < 0.68 and 0.28 < y_ratio < 0.72:
            score = 0.65
        elif 0.15 < z_ratio < 0.75:
            score = 0.40
        else:
            score = 0.15
    elif 'liver' in organ_lower:
        if 0.38 < z_ratio < 0.68 and 0.52 < x_ratio < 0.78:
            score = 1.0
        elif 0.32 < z_ratio < 0.75 and 0.47 < x_ratio < 0.83:
            score = 0.85
        elif 0.28 < z_ratio < 0.80 and 0.42 < x_ratio < 0.88:
            score = 0.65
        elif 0.22 < z_ratio < 0.85:
            score = 0.40
        else:
            score = 0.15
    elif 'spleen' in organ_lower:
        if 0.38 < z_ratio < 0.65 and 0.22 < x_ratio < 0.48:
            score = 1.0
        elif 0.32 < z_ratio < 0.72 and 0.17 < x_ratio < 0.53:
            score = 0.85
        elif 0.28 < z_ratio < 0.78 and 0.12 < x_ratio < 0.58:
            score = 0.65
        elif 0.22 < z_ratio < 0.82:
            score = 0.40
        else:
            score = 0.15
    elif 'kidney' in organ_lower:
        if 'left' in organ_lower:
            if 0.52 < z_ratio < 0.73 and 0.22 < x_ratio < 0.45:
                score = 1.0
            elif 0.47 < z_ratio < 0.78 and 0.17 < x_ratio < 0.50:
                score = 0.85
            elif 0.42 < z_ratio < 0.83:
                score = 0.65
            else:
                score = 0.35
        else:
            if 0.52 < z_ratio < 0.73 and 0.55 < x_ratio < 0.78:
                score = 1.0
            elif 0.47 < z_ratio < 0.78 and 0.50 < x_ratio < 0.83:
                score = 0.85
            elif 0.42 < z_ratio < 0.83:
                score = 0.65
            else:
                score = 0.35
    elif 'pancreas' in organ_lower:
        if 0.47 < z_ratio < 0.63 and 0.38 < x_ratio < 0.58:
            score = 1.0
        elif 0.42 < z_ratio < 0.68 and 0.33 < x_ratio < 0.63:
            score = 0.85
        elif 0.38 < z_ratio < 0.73 and 0.28 < x_ratio < 0.68:
            score = 0.65
        elif 0.33 < z_ratio < 0.78:
            score = 0.40
        else:
            score = 0.15
    elif 'lung' in organ_lower:
        if 0.12 < z_ratio < 0.52:
            score = 1.0
        elif 0.08 < z_ratio < 0.60:
            score = 0.88
        elif 0.05 < z_ratio < 0.68:
            score = 0.72
        elif z_ratio < 0.75:
            score = 0.50
        else:
            score = 0.20
    elif 'stomach' in organ_lower:
        if 0.38 < z_ratio < 0.58 and 0.35 < x_ratio < 0.55:
            score = 1.0
        elif 0.33 < z_ratio < 0.63 and 0.30 < x_ratio < 0.60:
            score = 0.85
        elif 0.28 < z_ratio < 0.68 and 0.25 < x_ratio < 0.65:
            score = 0.65
        elif 0.23 < z_ratio < 0.73:
            score = 0.40
        else:
            score = 0.15
    elif 'bladder' in organ_lower or 'urinary' in organ_lower:
        if 0.78 < z_ratio < 0.95 and 0.42 < x_ratio < 0.58:
            score = 1.0
        elif 0.73 < z_ratio < 0.98 and 0.37 < x_ratio < 0.63:
            score = 0.88
        elif 0.68 < z_ratio and 0.32 < x_ratio < 0.68:
            score = 0.70
        elif 0.62 < z_ratio:
            score = 0.45
        else:
            score = 0.15
    elif 'gallbladder' in organ_lower:
        if 0.42 < z_ratio < 0.62 and 0.52 < x_ratio < 0.68:
            score = 1.0
        elif 0.37 < z_ratio < 0.67 and 0.47 < x_ratio < 0.73:
            score = 0.85
        elif 0.32 < z_ratio < 0.72:
            score = 0.60
        else:
            score = 0.30
    elif 'colon' in organ_lower:
        if 0.40 < z_ratio < 0.85:
            score = 0.90
        elif 0.35 < z_ratio < 0.90:
            score = 0.75
        else:
            score = 0.50
    else:
        # Default for unknown organs
        score = 0.60
    
    return score

def calculate_centrality_moderate(data, img_shape):
    """Moderate centrality calculation"""
    if np.sum(data > 0) == 0:
        return 0.0
    
    center = np.array(img_shape) / 2
    organ_center = ndimage.center_of_mass(data > 0)
    
    distance = np.linalg.norm(np.array(organ_center) - center)
    max_distance = np.linalg.norm(center)
    
    normalized_distance = distance / max_distance
    centrality = np.exp(-1.8 * normalized_distance)
    
    return max(0.0, min(1.0, centrality))

def calculate_completeness_lenient(data):
    """Lenient completeness check"""
    if np.sum(data > 0) == 0:
        return 0.0
    
    mask = data > 0
    edge_penalty = 0
    
    edge_weights = [0.4, 0.4, 0.25, 0.25, 0.15, 0.15]
    
    edges = [
        mask[0, :, :], mask[-1, :, :],
        mask[:, 0, :], mask[:, -1, :],
        mask[:, :, 0], mask[:, :, -1]
    ]
    
    for edge, weight in zip(edges, edge_weights):
        if np.any(edge):
            contact_ratio = np.sum(edge) / edge.size
            edge_penalty += contact_ratio * 0.06 * weight
    
    completeness = 1.0 - min(edge_penalty, 0.80)
    return max(0.0, completeness)

def calculate_compactness_moderate(data):
    """Moderate compactness calculation"""
    if np.sum(data > 0) == 0:
        return 0.0
    
    mask = data > 0
    labeled, num_components = ndimage.label(mask)
    
    if num_components == 0:
        return 0.0
    
    component_sizes = ndimage.sum(mask, labeled, range(1, num_components + 1))
    largest_component_size = np.max(component_sizes)
    total_size = np.sum(mask)
    
    main_ratio = largest_component_size / total_size
    fragmentation_penalty = min((num_components - 1) * 0.04, 0.25)
    
    compactness = main_ratio - fragmentation_penalty
    
    return max(0.0, min(1.0, compactness))

def calculate_final_score(organ_key, organ_name, data, img, organ_info):
    """ULTRA ACCURATE final score calculation"""
    if np.sum(data > 0) == 0:
        return 0.0, {}
    
    voxel_count = np.sum(data > 0)
    voxel_volume = np.prod(img.header.get_zooms())
    volume_cm3 = (voxel_count * voxel_volume) / 1000
    
    if volume_cm3 < 0.5 or volume_cm3 > 10000:
        return 0.0, {}
    
    expected_range = organ_info['expected_volume']
    priority = organ_info['priority']
    weight = organ_info['weight']
    
    volume_score = calculate_volume_score_enhanced(volume_cm3, expected_range, priority)
    position_score = calculate_anatomical_position_score(data, img.shape, organ_name)
    centrality = calculate_centrality_moderate(data, img.shape)
    completeness = calculate_completeness_lenient(data)
    compactness = calculate_compactness_moderate(data)
    
    final_score = (
        volume_score * 0.50 +
        position_score * 0.35 +
        centrality * 0.07 +
        compactness * 0.05 +
        completeness * 0.03
    )
    
    final_score *= weight
    
    if priority >= 10:
        final_score *= 1.15
    elif priority >= 9:
        final_score *= 1.12
    elif priority >= 8:
        final_score *= 1.08
    
    details = {
        'volume_score': volume_score,
        'position_score': position_score,
        'centrality': centrality,
        'completeness': completeness,
        'compactness': compactness
    }
    
    return min(1.0, final_score), details

def analyze_single_organ(args):
    """FIXED: Analyze single organ with better matching"""
    file, output_dir = args
    
    if not file.endswith((".nii.gz", ".nii")) or 'converted' in file.lower():
        return None
    
    # Match organ using aliases
    organ_key, organ_info = match_organ_name(file)
    
    if not organ_info:
        print(f"   ⏭ Skipping unknown organ: {file}")
        return None
    
    organ_path = os.path.join(output_dir, file)
    
    try:
        img = nib.load(organ_path)
        data = img.get_fdata()
        
        voxel_count = np.sum(data > 0)
        if voxel_count == 0:
            print(f"   ⚠ Empty mask: {file}")
            return None
        
        voxel_volume = np.prod(img.header.get_zooms())
        volume_cm3 = (voxel_count * voxel_volume) / 1000
        
        total_score, details = calculate_final_score(
            organ_key, file, data, img, organ_info
        )
        
        if total_score < 0.01:
            print(f"   ⏭ Low score ({total_score:.3f}): {file}")
            return None
        
        print(f"   ✓ {organ_info['name']}: {volume_cm3:.1f} cm3 (score: {total_score:.3f})")
        
        return {
            'key': organ_key,
            'name': file,
            'display_name': organ_info['name'],
            'volume_cm3': volume_cm3,
            'voxel_count': int(voxel_count),
            'total_score': total_score,
            'priority': organ_info['priority'],
            'expected_range': organ_info['expected_volume'],
            **details
        }
    except Exception as e:
        print(f"   ❌ Error analyzing {file}: {e}")
        return None

def analyze_organs_parallel(output_dir):
    """FIXED: Parallel organ analysis with debug output"""
    organ_analysis = {}
    total_volume = 0
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory does not exist: {output_dir}")
        return organ_analysis, total_volume
    
    # DEBUG: List all files
    files = list_output_files(output_dir)
    
    if not files:
        print("❌ No segmentation files found!")
        return organ_analysis, total_volume
    
    print(f"\nAnalyzing {len(files)} organs...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        args_list = [(f, output_dir) for f in files]
        futures = [executor.submit(analyze_single_organ, args) for args in args_list]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                organ_key = result['key']
                # Handle duplicate organs (merge volumes)
                if organ_key in organ_analysis:
                    print(f"   📍 Merging duplicate: {result['display_name']}")
                    organ_analysis[organ_key]['volume_cm3'] += result['volume_cm3']
                    organ_analysis[organ_key]['voxel_count'] += result['voxel_count']
                    # Keep higher score
                    if result['total_score'] > organ_analysis[organ_key]['total_score']:
                        organ_analysis[organ_key]['total_score'] = result['total_score']
                else:
                    organ_analysis[organ_key] = result
                
                total_volume += result['volume_cm3']
    
    if total_volume > 0:
        for organ_key in organ_analysis:
            volume = organ_analysis[organ_key]['volume_cm3']
            percentage = (volume / total_volume) * 100
            organ_analysis[organ_key]['percentage'] = percentage
    
    return organ_analysis, total_volume

def format_final_report(organ_analysis, total_volume):
    """Format final report - ORGANS FIRST, MAIN ORGAN AT END"""
    if not organ_analysis:
        return "ERROR: No main organs detected", None
    
    sorted_by_percentage = sorted(
        organ_analysis.items(),
        key=lambda x: x[1]['percentage'],
        reverse=True
    )
    
    sorted_by_score = sorted(
        organ_analysis.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True
    )
    
    main_organ_key = sorted_by_score[0][0]
    main_stats = sorted_by_score[0][1]
    main_organ_name = main_stats['display_name']
    
    total_percentage = sum(stats['percentage'] for _, stats in organ_analysis.items())
    
    report = "\n" + "="*110 + "\n"
    report += "                         ORGAN DETECTION ANALYSIS REPORT\n"
    report += "="*110 + "\n\n"
    
    # 🔥 ORGANS IN IMAGE (بالنسب المئوية)
    report += "📊 ORGANS DETECTED IN THIS IMAGE:\n"
    report += "="*110 + "\n\n"
    report += f"{'#':<5} {'Organ Name':<28} {'Percentage':<14} {'Visual Bar':<32} {'Volume':<15} {'Score':<10}\n"
    report += "-"*110 + "\n"
    
    for idx, (key, stats) in enumerate(sorted_by_percentage, 1):
        name = stats['display_name']
        percentage = stats['percentage']
        volume = stats['volume_cm3']
        score = stats['total_score']
        
        bar_length = int(percentage / 2)
        bar = "█" * min(bar_length, 30)
        
        report += f"{idx:<5} {name:<28} {percentage:>6.2f}%      {bar:<32} {volume:>9.1f} cm3   {score:.4f}\n"
    
    report += "-"*110 + "\n"
    report += f"{'TOTAL':<5} {'All Detected Organs':<28} {total_percentage:>6.2f}%      {'█'*30:<32} {total_volume:>9.1f} cm3\n"
    report += "\n" + "="*110 + "\n"
    report += f"Total Number of Detected Organs: {len(organ_analysis)}\n"
    report += f"Total Volume: {total_volume:.2f} cm3\n"
    report += f"Percentage Sum: {total_percentage:.2f}% ✓\n"
    report += "="*110 + "\n\n"
    
    # 🏆 MAIN ORGAN (في الآخر)
    report += "\n"
    report += "🏆 " + "="*105 + "\n"
    report += "                           MAIN DETECTED ORGAN (Highest Score)\n"
    report += "="*110 + "\n\n"
    
    report += f">>> {main_organ_name.upper()} <<<\n\n"
    
    report += "DETAILED ANALYSIS:\n"
    report += f"   Total Score:        {main_stats['total_score']:.4f} / 1.0000\n"
    report += f"   Volume:             {main_stats['volume_cm3']:.2f} cm3\n"
    report += f"   Percentage:         {main_stats['percentage']:.2f}% (of total organs)\n"
    report += f"   Expected Range:     {main_stats['expected_range'][0]}-{main_stats['expected_range'][1]} cm3\n\n"
    
    report += "SCORE BREAKDOWN:\n"
    report += f"   Volume Score:       {main_stats['volume_score']:.4f} (Weight: 50%)\n"
    report += f"   Position Score:     {main_stats['position_score']:.4f} (Weight: 35%)\n"
    report += f"   Centrality:         {main_stats['centrality']:.4f} (Weight: 7%)\n"
    report += f"   Compactness:        {main_stats['compactness']:.4f} (Weight: 5%)\n"
    report += f"   Completeness:       {main_stats['completeness']:.4f} (Weight: 3%)\n"
    
    if len(sorted_by_score) > 1:
        second_name = sorted_by_score[1][1]['display_name']
        second_score = sorted_by_score[1][1]['total_score']
        gap = main_stats['total_score'] - second_score
        
        if gap > 0.20:
            confidence = "VERY HIGH"
            emoji = "✓✓✓"
        elif gap > 0.12:
            confidence = "HIGH"
            emoji = "✓✓"
        elif gap > 0.06:
            confidence = "MEDIUM"
            emoji = "✓"
        else:
            confidence = "LOW"
            emoji = "⚠"
        
        report += f"\n{emoji} CONFIDENCE LEVEL: {confidence}\n"
        report += f"   Score gap from 2nd place [{second_name}]: {gap:.4f}\n"
    else:
        report += f"\n✓✓✓ CONFIDENCE LEVEL: VERY HIGH (Only organ detected)\n"
    
    report += "\n" + "="*110 + "\n"
    report += "NOTES:\n"
    report += "- Main organ = Highest combined score (volume + position + other factors)\n"
    report += "- Percentages show relative size of each organ in the image\n"
    report += "- Higher scores indicate more confident detection\n"
    report += "="*110 + "\n"
    
    return report, main_organ_name

def print_final_result(organ_name):
    """Print final result banner"""
    print("\n\n")
    print("🏆 " + "="*105)
    print("="*110)
    print("#" + " " * 108 + "#")
    print("#" + " " * 40 + "MAIN DETECTED ORGAN" + " " * 49 + "#")
    print("#" + " " * 108 + "#")
    
    text = f">>> {organ_name.upper()} <<<"
    padding = (108 - len(text)) // 2
    print("#" + " " * padding + text + " " * (108 - padding - len(text)) + "#")
    
    print("#" + " " * 108 + "#")
    print("="*110)
    print("="*110)
    print("\n")

def save_report(report, output_folder):
    """Save report to file"""
    report_path = os.path.join(output_folder, "organ_detection_report.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved: {report_path}")
    except Exception as e:
        print(f"⚠ Warning: Could not save report: {e}")

def main():
    global TEMP_DIR
    
    print("="*110)
    print("                    ORGAN DETECTOR v6 - FIXED MATCHING")
    print("="*110 + "\n")
    
    print("Checking TotalSegmentator...")
    ts_command = check_totalsegmentator()
    if not ts_command:
        print("❌ ERROR: TotalSegmentator not installed")
        print("Install with: pip install TotalSegmentator")
        input("\nPress Enter to exit...")
        return
    
    print("✓ TotalSegmentator is ready\n")
    
    original_folder, clean_dicom_folder = select_folder_fixed()
    
    if not clean_dicom_folder:
        input("\nPress Enter to exit...")
        return
    
    output_folder = os.path.join(original_folder, "output")
    os.makedirs(output_folder, exist_ok=True)
    
    TEMP_DIR = tempfile.mkdtemp()
    os.environ["TMPDIR"] = TEMP_DIR
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    print("\n⚙️  Starting processing...")
    print("⏱️  Estimated time: 2-6 minutes\n")
    
    if ts_command == "TotalSegmentator":
        cmd = [
            "TotalSegmentator",
            "-i", clean_dicom_folder,
            "-o", output_folder,
            "--fast",
            "--force_split",
            "--body_seg"
        ]
    else:
        cmd = ts_command.split() + [
            "-i", clean_dicom_folder,
            "-o", output_folder,
            "--fast",
            "--force_split",
            "--body_seg"
        ]
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        last_print = time.time()
        dots = 0
        
        for line in process.stdout:
            current_time = time.time()
            if current_time - last_print > 3:
                elapsed = int(current_time - start_time)
                dots = (dots + 1) % 4
                progress = "." * dots + " " * (3 - dots)
                print(f"\r   Processing{progress} [{elapsed//60}min {elapsed%60}s]", end='', flush=True)
                last_print = current_time
        
        process.wait(timeout=900)
        
        elapsed = time.time() - start_time
        print(f"\n\n✓ Processing completed in {elapsed/60:.1f} minutes")
        
    except subprocess.TimeoutExpired:
        print("\n❌ ERROR: Process timeout")
        if process:
            process.kill()
        input("\nPress Enter to exit...")
        return
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        if process:
            process.kill()
        input("\nPress Enter to exit...")
        return
    except Exception as e:
        print(f"\n❌ ERROR during processing: {e}")
        input("\nPress Enter to exit...")
        return
    
    organ_analysis, total_volume = analyze_organs_parallel(output_folder)
    
    if organ_analysis:
        print(f"\n✓ Found {len(organ_analysis)} organs")
        
        report, main_organ = format_final_report(organ_analysis, total_volume)
        
        if main_organ:
            print(report)
            save_report(report, output_folder)
            print_final_result(main_organ)
            
            try:
                if os.path.exists(clean_dicom_folder):
                    shutil.rmtree(clean_dicom_folder, ignore_errors=True)
            except:
                pass
        else:
            print("❌ ERROR: No main organs detected")
    else:
        print("❌ ERROR: No organs found")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Check if TotalSegmentator completed successfully")
        print("   2. Verify output folder contains .nii.gz files")
        print("   3. Try running with --ml flag for better results")
    
    print(f"\n📁 Results saved in: {output_folder}")
    print("\n" + "="*110)
    print("                           ✓ PROCESS COMPLETED")
    print("="*110)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Program stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
    