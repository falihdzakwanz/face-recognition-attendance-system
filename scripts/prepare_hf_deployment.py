"""
Prepare files for Hugging Face Spaces deployment
=================================================

This script:
1. Creates deployment directory structure
2. Copies necessary files (no student photos for privacy)
3. Creates empty dataset folders for class names
4. Prepares requirements.txt
"""

import shutil
from pathlib import Path
import json

def prepare_deployment():
    """Prepare deployment package."""
    
    print("=" * 60)
    print("Preparing Hugging Face Spaces Deployment")
    print("=" * 60)
    
    # Root directories
    project_root = Path(__file__).parent.parent
    deploy_dir = project_root / "hf_deployment"
    
    # Clean and create deployment directory
    if deploy_dir.exists():
        print(f"\n⚠ Removing existing deployment directory: {deploy_dir}")
        shutil.rmtree(deploy_dir)
    
    deploy_dir.mkdir()
    print(f"✓ Created deployment directory: {deploy_dir}")
    
    # 1. Copy source code
    print("\n1. Copying source code...")
    shutil.copytree(project_root / "src", deploy_dir / "src")
    print("   ✓ src/")
    
    # 2. Copy config files
    print("\n2. Copying configuration files...")
    files_to_copy = [
        "app.py",
        "config.yaml",
        "README_HF.md",
        "requirements.txt"
    ]
    
    for file in files_to_copy:
        src_file = project_root / file
        if src_file.exists():
            if file == "README_HF.md":
                shutil.copy(src_file, deploy_dir / "README.md")
                print(f"   ✓ {file} → README.md")
            else:
                shutil.copy(src_file, deploy_dir / file)
                print(f"   ✓ {file}")
        else:
            print(f"   ⚠ {file} not found, skipping")
    
    # 3. Create dataset structure (empty folders for class names only)
    print("\n3. Creating dataset structure (no photos for privacy)...")
    train_dir = project_root / "dataset" / "Train"
    deploy_train_dir = deploy_dir / "dataset" / "Train"
    deploy_train_dir.mkdir(parents=True)
    
    if train_dir.exists():
        class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        
        # Create empty folders
        for class_name in class_names:
            (deploy_train_dir / class_name).mkdir()
        
        print(f"   ✓ Created {len(class_names)} empty class folders")
        
        # Save class names to JSON for reference
        class_names_file = deploy_dir / "class_names.json"
        with open(class_names_file, 'w', encoding='utf-8') as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Saved class names to class_names.json")
    else:
        print("   ⚠ Train directory not found")
    
    # 4. Copy model (find latest checkpoint)
    print("\n4. Copying trained model...")
    models_dir = project_root / "models"
    
    if models_dir.exists():
        # Find latest CNN model
        cnn_models = sorted(models_dir.glob("cnn_*/best_model.pth"), 
                           key=lambda p: p.stat().st_mtime, 
                           reverse=True)
        
        if cnn_models:
            latest_model = cnn_models[0]
            model_dir = latest_model.parent
            
            # Copy entire model directory
            deploy_model_dir = deploy_dir / "models" / model_dir.name
            shutil.copytree(model_dir, deploy_model_dir)
            
            print(f"   ✓ Copied: {model_dir.name}/")
            print(f"     Size: {latest_model.stat().st_size / (1024*1024):.1f} MB")
        else:
            print("   ⚠ No trained CNN model found")
    else:
        print("   ⚠ Models directory not found")
    
    # 5. Create .gitattributes for Git LFS
    print("\n5. Creating .gitattributes for Git LFS...")
    gitattributes = deploy_dir / ".gitattributes"
    with open(gitattributes, 'w') as f:
        f.write("*.pth filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.pkl filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
    print("   ✓ .gitattributes created")
    
    # 6. Create deployment instructions
    print("\n6. Creating deployment instructions...")
    instructions = deploy_dir / "DEPLOY_INSTRUCTIONS.txt"
    with open(instructions, 'w', encoding='utf-8') as f:
        f.write("""
╔═══════════════════════════════════════════════════════════════╗
║        Hugging Face Spaces Deployment Instructions           ║
╚═══════════════════════════════════════════════════════════════╝

STEP 1: Install Hugging Face CLI
---------------------------------
pip install huggingface_hub

STEP 2: Login to Hugging Face
------------------------------
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens

STEP 3: Create a new Space
---------------------------
1. Go to: https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - Name: face-recognition-presensi
   - License: MIT
   - SDK: Gradio
   - Hardware: CPU Basic (free) or GPU (paid)

STEP 4: Clone the Space repository
-----------------------------------
git clone https://huggingface.co/spaces/YOUR_USERNAME/face-recognition-presensi
cd face-recognition-presensi

STEP 5: Copy deployment files
------------------------------
# Copy all files from hf_deployment/ to your Space repo
cp -r /path/to/hf_deployment/* .

STEP 6: Initialize Git LFS (for model files)
---------------------------------------------
git lfs install
git lfs track "*.pth"

STEP 7: Push to Hugging Face
-----------------------------
git add .
git commit -m "Initial deployment"
git push

STEP 8: Wait for build
----------------------
Your Space will automatically build and deploy!
URL: https://huggingface.co/spaces/YOUR_USERNAME/face-recognition-presensi

═══════════════════════════════════════════════════════════════

NOTES:
------
✓ Model file (~100-500MB) will be uploaded via Git LFS
✓ Free tier uses CPU (slower inference)
✓ GPU costs ~$0.60/hour (can pause when not in use)
✓ No student photos uploaded (privacy protected)
✓ Space can be set to private if needed

TROUBLESHOOTING:
----------------
• If model upload fails: Use `git lfs push --all origin`
• If build fails: Check requirements.txt dependencies
• For GPU: Upgrade Space in Settings → Hardware

═══════════════════════════════════════════════════════════════
""")
    print("   ✓ DEPLOY_INSTRUCTIONS.txt created")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ Deployment package ready!")
    print("=" * 60)
    print(f"\nLocation: {deploy_dir}")
    print("\nNext steps:")
    print("1. Read: DEPLOY_INSTRUCTIONS.txt")
    print("2. Install HF CLI: pip install huggingface_hub")
    print("3. Login: huggingface-cli login")
    print("4. Create Space at: https://huggingface.co/spaces")
    print("5. Push files to your Space repository")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    prepare_deployment()
