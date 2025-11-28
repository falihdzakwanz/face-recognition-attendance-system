"""
Debug inference script
- Loads latest CNN checkpoint
- Runs inference on one example test image with resize 160 and 224
- Prints top-5 predictions and confidences for both sizes
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.config_loader import load_config


def topk_probs(probs, class_names, k=5):
    probs = probs.flatten()
    topk = probs.argsort()[-k:][::-1]
    return [(class_names[i] if i < len(class_names) else str(i), float(probs[i])) for i in topk]


def main():
    cfg = load_config('config.yaml')
    models_dir = Path(cfg['paths']['models_dir'])
    cnn_ckpts = list(models_dir.glob('cnn_*/best_model.pth'))
    if not cnn_ckpts:
        print('No CNN checkpoints found in', models_dir)
        return
    latest = max(cnn_ckpts, key=lambda p: p.stat().st_mtime)
    print('Using checkpoint:', latest)

    # Load class names from train dir
    train_dir = Path(cfg['paths']['train_dir'])
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print('Num classes:', len(class_names))

    # Load model via create_model
    from src.models.cnn_facenet import create_model
    model = create_model(cfg, len(class_names), model_type='arcface')
    ck = torch.load(str(latest), map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.eval()

    print('Model class:', model.__class__)
    print('Checkpoint epoch:', ck.get('epoch'))
    print('Checkpoint val_acc:', ck.get('val_acc'))

    # Pick a sample test image
    test_dir = Path(cfg['paths']['test_dir'])
    # find first class with images
    selected_img = None
    for student_dir in sorted(test_dir.iterdir()):
        if student_dir.is_dir():
            imgs = list(student_dir.glob('*.jpg')) + list(student_dir.glob('*.jpeg')) + list(student_dir.glob('*.png'))
            if imgs:
                selected_img = imgs[0]
                selected_student = student_dir.name
                break
    if selected_img is None:
        print('No test images found')
        return

    print('Selected sample:', selected_img, 'label:', selected_student)
    img_bgr = cv2.imread(str(selected_img))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess and infer for sizes 160 and 224
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for size in (160, 224):
        img_resized = cv2.resize(img_rgb, (size, size))
        x = img_resized.astype('float32') / 255.0
        x = (x - mean) / std
        x = np.transpose(x, (2,0,1))[None, ...]
        x_tensor = torch.from_numpy(x)
        with torch.no_grad():
            out = model(x_tensor)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=1).cpu().numpy()
        top5 = topk_probs(probs, class_names, k=5)
        print(f'--- Size {size}x{size} ---')
        for name, p in top5:
            print(f'{name}: {p:.4f}')

if __name__ == '__main__':
    main()
