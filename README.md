# 🧠 Brain Tumor Detection (React + FastAPI)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-61DAFB.svg?logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-Bundler-646CFF.svg?logo=vite)](https://vitejs.dev/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Vercel](https://img.shields.io/badge/Vercel-Deploy-black.svg?logo=vercel)](#)

> An **end-to-end Convolutional Neural Network (CNN) based Brain Tumor Detection System** that classifies MRI brain scans into 4 categories (Glioma, Meningioma, Pituitary, No Tumor). 
> 
> *Recently completely restructured* from a monolithic Flask app into a modern **React.js SPA** with a **FastAPI backend**, designed for seamless **Vercel Deployment**.

## 🚀 Key Features

* ✨ **Modern React UI** — Premium, dark-themed, glassmorphic design built with Tailwind CSS & Framer Motion.
* 🧠 **FastAPI Backend** — High-performance python API serving the deep learning model.
* 🎯 **Multi-class CNN** — Achieves ~96% accuracy across 4 tumor types.
* 📊 **Interactive Results** — Animated confidence bars and visual probability distribution.
* ☁️ **Vercel Ready** — Pre-configured `vercel.json` for serverless deployment.

---

## 🗂️ Project Structure

```
├── api/                   # FastAPI Backend
│   ├── index.py           # Serverless API entrypoint
│   └── requirements.txt   # Python dependencies
├── frontend/              # React.js Frontend (Vite)
│   ├── src/               # React components & styles
│   ├── package.json
│   └── vite.config.js     # Dev proxy config
├── model/                 # Machine Learning
│   ├── model_building.ipynb
│   └── model.h5           # The trained weights (Generate this first!)
├── dataset/               # Original MRI dataset scripts
├── vercel.json            # Vercel Deployment configuration
└── main.py                # (Legacy) Original Flask monolithic app
```

---

## ⚙️ Running Locally

### Prerequisites
1. **Node.js** (v18+) for the React frontend
2. **Python** (3.8+) for the FastAPI backend
3. A trained model file at `model/model.h5`. *(If you don't have it, run `jupyter notebook model/model_building.ipynb` first to train it)*.

### 1. Start the FastAPI Backend
Open a terminal and run:
```bash
# Optional: create a virtual environment
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
cd api
pip install -r requirements.txt

# Start the server
uvicorn index:app --reload --port 8000
```
*The API will be running at `http://localhost:8000`*

### 2. Start the React Frontend
Open **another** terminal and run:
```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```
*The frontend will be running at `http://localhost:5173` and will proxy `/api` calls to port 8000.*

---

## ☁️ Deploying to Vercel

This project is configured natively for Vercel using `vercel.json` which maps `/api/*` to the Python serverless function and everything else to the React static build.

> [!WARNING]
> **Important Note about Model Size:**
> Vercel's Serverless Functions have a strict **50MB size limit** (compressed) for the free tier. 
> The `tensorflow` package combined with your `model.h5` file might exceed this limit. 
> 
> If you encounter a `MaxServerlessFunctionSizeError` during Vercel deployment, you will need to:
> 1. Host the backend separately on a platform like **Render** or **Railway** (which don't have small size limits).
> 2. Change the `VITE_API_URL` environment variable in your Vercel React frontend project to point to your new backend URL.

### Steps to Deploy (assuming model size is within limits):
1. Push this repository to GitHub.
2. Go to your Vercel dashboard and click **Add New Project**.
3. Import this GitHub repository.
4. **Framework Preset**: Vercel should auto-detect Vite.
5. **Root Directory**: Leave as `./` (the root).
6. Click **Deploy**. Vercel will automatically read `vercel.json`, build the frontend, and set up the Python serverless functions.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** This AI-powered system is intended **only for research and educational purposes**.
> 
> - ❌ It should **NOT** be used as a substitute for professional medical diagnosis or treatment
> - ✅ Always consult **qualified healthcare professionals** for medical decisions
> - ✅ This tool is designed to assist, not replace, medical expertise

---

## 👤 Author

**Jaidatt Kale**
- 🔗 [LinkedIn](https://linkedin.com/in/jaidattkale)
- 🔗 [GitHub](https://github.com/jaidatt007)

*Project restructured to React/FastAPI.*
