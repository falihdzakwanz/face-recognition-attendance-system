# Quickstart: Model Testing & Evaluation

## 1. Prepare Environment

- Ensure you have installed all dependencies:
  ```powershell
  pip install -r requirements.txt
  ```
- Activate your virtual environment if needed.

## 2. Prepare Model & Test Data

- Place your trained model in `models/cnn_*/best_model.pth` (or any supported location).
- Organize your test images in folders:
  - `dataset/test/<student_name>/*.jpg`
  - The script will auto-detect the test folder and model.

## 3. Run the Evaluation Script

- Run the test script:
  ```powershell
  python test.py
  ```
- The script will:
  - Find the latest model and test folder automatically
  - Print per-image predictions
  - Show confusion matrix and metrics (accuracy, precision, recall, F1)
  - Save detailed results to CSV
  - Save confusion matrix plot (PNG)

## 4. Output & Results

- You will see:
  - Per-image prediction results
  - Overall metrics (accuracy, precision, recall, F1)
  - Confusion matrix (printed and saved as PNG)
  - Per-class accuracy table
  - CSV file with all results (timestamped)

## 5. Troubleshooting

- If you see errors about missing model or test data, check your folder structure.
- The script is robust and will fallback to any available test/model folder.
- For custom test folders, edit `test.py` or pass your own path.

## Example Output

```
python test.py

🎯 FACE RECOGNITION EVALUATION SYSTEM
Model: models/cnn_arcface_20251127_163448/best_model.pth
Test Directory: dataset/test
Config: config.yaml
...
📈 Accuracy:    0.9000 (90.00%)
🎯 Precision:   0.9000 (90.00%)
🔍 Recall:      0.9000 (90.00%)
⚖️  F1-Score:    0.9000 (90.00%)
Confusion Matrix:
[[10  0]
 [ 1  9]]
...
💾 Detailed results saved: evaluation_results_20251129_160000.csv
✓ Confusion matrix saved: confusion_matrix_20251129_160000.png
```

## 6. Advanced Usage

- You can modify `test.py` to use a specific model or test folder.
- For single image testing, adapt the `predict()` function in `test.py`.

---
