#  Door & Window Detection in Architectural Blueprints

**Final Submission for Palcode.ai Internship Assignment**

✅ Manual Labeling → ✅ Augmentation → ✅ YOLOv8 → ✅ ONNX → ✅ API → ✅ Deployed

---

## 🚀 Project Overview

This project detects `door` and `window` objects in **construction blueprint images**.

* Labeled with **Roboflow**
  👉 [Roboflow Project Link](https://universe.roboflow.com/asp-2t4fy/doors-and-windows-cgn3z/dataset/2)
* Augmented in Roboflow
* Trained **YOLOv8**
* Exported to **ONNX** with NMS
* Flask app with:

  * Web UI (upload images + visualize detections)
  * `/detect` API returning JSON
* Deployed on **Render**
  👉 [https://vision-pylv.onrender.com](https://vision-pylv.onrender.com)

---

## 📂 Repository Structure

```
├── Doors and Windows/          # images/ + labels/ (organized)
├── Proof of work/              # Screenshots, labeling, training proofs
│   ├── Screenshot 2025-05-30 at 3.32.58 PM.png  # Labeling screenshot
│   ├── Results post augmentation.png           # Training results / loss graph
├── Raw dataset/                # Original images
├── static/results/             # Output images with bounding boxes
├── templates/index.html        # Web UI template
├── uploads/                    # Temporary upload folder
├── 16.png                      # Test image
├── Doors and Windows.zip       # ZIP of images + labels
├── README.md                   # This file
├── app.py                      # Flask app with Web UI + /detect API
├── requirements.txt            # Python dependencies
├── test.py                     # (optional test script)
├── vision2.onnx                # Optimized ONNX model
└── x.py                        # (helper/test script)
```

---

## 🏷️ Classes

```
door
window
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/blueprint-door-window-detection.git
cd blueprint-door-window-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Locally

```bash
python app.py
```

Web UI: [http://localhost:8000](http://localhost:8000)
API: [http://localhost:8000/detect](http://localhost:8000/detect)

---

## 🖼️ API Usage

### POST `/detect`

* Accepts: PNG or JPG image
* Returns: JSON with bounding boxes

### Example `curl`:

```bash
curl -X POST https://vision-pylv.onrender.com/detect \
  -F "file=@16.png"
```

### Example Response:

```json
{
  "detections": [
    {
      "label": "door",
      "confidence": 0.91,
      "bbox": [x1, y1, x2, y2]
    },
    {
      "label": "window",
      "confidence": 0.84,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## ⚙️ Pipeline

| Step         | Tool                 |
| ------------ | -------------------- |
| Labeling     | Roboflow             |
| Augmentation | Roboflow             |
| Training     | YOLOv8               |
| Optimization | ONNX export with NMS |
| Inference    | ONNXRuntime          |
| API          | Flask                |
| Deployment   | Render               |

---

## 📸 Proof of Work

### 1️⃣ Labeling Screenshot

📍 `Proof of work/Screenshot 2025-05-30 at 3.32.58 PM.png`

![Labeling](Proof%20of%20work/Screenshot%202025-05-30%20at%203.32.58%E2%80%AFPM.png)

---

### 2️⃣ Training Screenshot / Loss Graph

📍 `Proof of work/Results post augmentation.png`

![Training](Proof%20of%20work/Results%20post%20augmentation.png)

---

### 3️⃣ API Response Example

Example shown on deployed API: [https://vision-pylv.onrender.com](https://vision-pylv.onrender.com)



---

## 🔗 Final Submission Links

✅ GitHub Repo: [https://github.com/yourusername/blueprint-door-window-detection](https://github.com/yourusername/blueprint-door-window-detection)
✅ Public API URL (Render): [https://vision-pylv.onrender.com](https://vision-pylv.onrender.com)
✅ Loom Video: [https://loom.com/share/your-video-link](https://loom.com/share/your-video-link)

---

## Notes

* ONNX runtime ensures fast inference
* Threshold: `confidence > 0.4`
* Handles **non-labeled blueprint sheets** gracefully
* Web UI and API both ready for testing

---

