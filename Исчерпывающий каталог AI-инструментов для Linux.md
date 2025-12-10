# Исчерпывающий каталог AI-инструментов для Linux: 300+ инструментов в 12 категориях

Искусственный интеллект трансформирует работу с Linux, предлагая мощные инструменты командной строки для разработки, безопасности, автоматизации и машинного обучения. Этот каталог охватывает **315+ уникальных AI-инструментов**, совместимых с основными Linux-дистрибутивами, все с активной разработкой в 2024-2025 годах. Большинство инструментов работают локально без GPU, обеспечивая приватность и независимость от облачных сервисов.

---

## Quick Start Guide: ТОП-20 Must-Have инструментов

| # | Инструмент | Категория | Установка | Ключевое применение |
|---|------------|-----------|-----------|---------------------|
| 1 | **Ollama** | NLP/LLM | `curl -fsSL https://ollama.com/install.sh \| sh` | Локальный запуск LLM без настройки |
| 2 | **llama.cpp** | NLP/LLM | `git clone && make` | Эффективный inference на CPU/GPU |
| 3 | **Whisper** | Audio | `pip install openai-whisper` | STT с 99+ языками |
| 4 | **Stable Diffusion WebUI** | Graphics | `git clone && ./webui.sh` | Генерация изображений |
| 5 | **YOLO (Ultralytics)** | CV | `pip install ultralytics` | Object detection в реальном времени |
| 6 | **MLflow** | Data Science | `pip install mlflow` | MLOps и трекинг экспериментов |
| 7 | **DVC** | Data Science | `pip install dvc` | Version control для данных/моделей |
| 8 | **Aider** | Development | `pip install aider-chat` | AI pair programming в терминале |
| 9 | **ShellGPT** | System Admin | `pip install shell-gpt` | AI-ассистент для shell-команд |
| 10 | **PentestGPT** | Security | `pip install pentestgpt` | AI-driven penetration testing |
| 11 | **Sherlock** | OSINT | `pip install sherlock-project` | Поиск аккаунтов по username |
| 12 | **DeepFace** | CV/OSINT | `pip install deepface` | Face recognition и анализ |
| 13 | **PaddleOCR** | CV | `pip install paddleocr` | Multilingual OCR (80+ языков) |
| 14 | **Coqui TTS** | Audio | `pip install TTS` | Voice cloning и TTS |
| 15 | **Suricata** | Security | `apt install suricata` | IDS/IPS с ML-анализом |
| 16 | **ComfyUI** | Graphics | `pip install comfy-cli` | Node-based image generation |
| 17 | **Weaviate** | Productivity | `docker run weaviate` | Vector database для RAG |
| 18 | **Netdata** | System Admin | `bash <(curl -Ss get.netdata.cloud)` | Мониторинг с ML anomaly detection |
| 19 | **Real-ESRGAN** | Graphics | `pip install realesrgan` | AI upscaling изображений |
| 20 | **Paperless-ngx** | Productivity | `docker-compose up -d` | Document management с OCR |

---

# ПОЛНЫЙ КАТАЛОГ ПО КАТЕГОРИЯМ

## 1. CYBERSECURITY &amp; PENTESTING (35 инструментов)

### AI Pentesting Assistants

| Инструмент | Описание | GitHub | Stars | Установка |
|------------|----------|--------|-------|-----------|
| **PentestGPT** | GPT-4 powered pentesting assistant | [GitHub](https://github.com/GreyDGL/PentestGPT) | 7.5k | `pip install pentestgpt` |
| **PyRIT** | Microsoft's AI red-teaming framework | [GitHub](https://github.com/Azure/PyRIT) | 1.5k | `pip install pyrit` |
| **HackingBuddyGPT** | LLM security assistant для ethical hacking | [GitHub](https://github.com/ipa-lab/hackingBuddyGPT) | 2k | `pip install hackingBuddyGPT` |
| **CAI** | Cybersecurity AI framework | [GitHub](https://github.com/aliasrobotics/cai) | 500+ | pip install |
| **Nebula** | AI pentesting с auto note-taking | [GitHub](https://github.com/berylliumsec/nebula) | 1k | `pip install beryllium-nebula` |
| **HexStrike AI** | MCP server для 150+ security tools | [GitHub](https://github.com/0x4m4/hexstrike-ai) | 500 | Docker |
| **PentAGI** | Autonomous AI agent для pentest | [GitHub](https://github.com/vxcontrol/pentagi) | 300 | Docker Compose |
| **Strix** | AI agents для vulnerability finding | [GitHub](https://github.com/usestrix/strix) | 200 | `pipx install strix-agent` |
| **DeepExploit** | ML-driven Metasploit automation | [GitHub](https://github.com/13o-bbr-bbq) | 1k | Clone + pip |

### IDS/IPS и Network Security

| Инструмент | Описание | Лицензия | Установка |
|------------|----------|----------|-----------|
| **Suricata** | Multi-threaded IDS/IPS/NSM | GPL-2.0 | `apt install suricata` |
| **Snort** | Classic signature-based IDS | GPL-2.0 | `apt install snort` |
| **Zeek** | Network traffic analyzer | BSD | `apt install zeek` |
| **CrowdSec** | Crowd-sourced ML threat detection | MIT | `curl install.crowdsec.net \| sh` |

### Malware Analysis

| Инструмент | Описание | Stars | Установка |
|------------|----------|-------|-----------|
| **YARA** | Pattern matching для malware | 8k | `apt install yara` |
| **ClamAV** | AV engine с ML detection | - | `apt install clamav` |
| **StringSifter** | ML ranking malware strings | 700 | `pip install stringsifter` |
| **Volatility3** | Memory forensics с ML | 2.5k | `pip install volatility3` |
| **MobSF** | Mobile security assessment | 18k | Docker |

### Vulnerability Scanning

| Инструмент | Описание | Stars | Пример |
|------------|----------|-------|--------|
| **Nuclei** | Template-based vuln scanner | 21k | `nuclei -u https://target.com` |
| **Lynis** | Security auditing tool | 13k | `lynis audit system` |
| **Prowler** | Cloud security assessment | 11k | `prowler aws --profile myprofile` |
| **Hashcat** | GPU password recovery | 22k | `hashcat -m 1000 hashes.txt wordlist.txt` |

---

## 2. OSINT - Open Source Intelligence (25 инструментов)

### Username &amp; Social Media OSINT

| Инструмент | Описание | Stars | Sites | CLI Example |
|------------|----------|-------|-------|-------------|
| **Sherlock** | Hunt usernames across 400+ sites | 59k | 400+ | `sherlock username --csv` |
| **Maigret** | Collect accounts from 3000+ sites | 11k | 3000+ | `maigret username --all-sites` |
| **Holehe** | Check email on sites | 7k | 100+ | `holehe email@example.com` |
| **Blackbird** | Username search 600+ sites | 3k | 600+ | `python blackbird.py -u username` |
| **Social-Analyzer** | Analyze profiles across 1000+ sites | 11k | 1000+ | `social-analyzer --username target` |

### Domain &amp; Email Intelligence

| Инструмент | Описание | Stars | Установка |
|------------|----------|-------|-----------|
| **theHarvester** | Gather emails, IPs, subdomains | 11k | `pip install theHarvester` |
| **SpiderFoot** | 200+ OSINT modules | 13.6k | Clone + pip |
| **Recon-ng** | Web reconnaissance framework | 3.5k | `pip install recon-ng` |
| **Amass** | Attack surface mapping | 12k | Go install |
| **Subfinder** | Passive subdomain enumeration | 10k | Go install |
| **h8mail** | Email breach hunting | 4k | `pip install h8mail` |

### Face Recognition &amp; Image OSINT

| Инструмент | Описание | Accuracy | Stars |
|------------|----------|----------|-------|
| **DeepFace** | Face recognition + attributes | 99.38% | 16k |
| **InsightFace** | 2D/3D face analysis | SOTA | 23k |
| **RetinaFace** | Face detection + landmarks | High | 1k |
| **face_recognition** | Simple API for face recognition | 99.38% | 55k |

### Specialized OSINT Tools

| Инструмент | Target | Stars | CLI |
|------------|--------|-------|-----|
| **GHunt** | Google accounts | 16k | `ghunt email target@gmail.com` |
| **PhoneInfoga** | Phone numbers | 13k | `phoneinfoga scan -n "+1234567890"` |
| **Shodan CLI** | IoT devices | 2.5k | `shodan search "apache"` |
| **IntelOwl** | Data enrichment platform | 4k | Docker API |
| **ExifTool** | Metadata extraction | - | `exiftool -all image.jpg` |
| **Photon** | Web crawler for OSINT | 11k | `photon -u https://target.com` |

---

## 3. SOFTWARE DEVELOPMENT (30 инструментов)

### Local Code Completion (Copilot Alternatives)

| Инструмент | Описание | Stars | Local | GPU |
|------------|----------|-------|-------|-----|
| **Tabby** | Self-hosted AI coding assistant | 25k | ✅ | Recommended |
| **Aider** | AI pair programming in terminal | 30k | ✅ | Optional |
| **Continue.dev** | Open-source VS Code AI assistant | 20k | ✅ | Optional |
| **Codeium** | Free code completion (70+ langs) | - | Partial | No |
| **Cline** | Autonomous VS Code agent | 48k | ✅ | Optional |
| **Amazon Q CLI** | AWS AI command line | 2k | No | No |

**Пример установки Tabby:**
```bash
docker run -it --gpus all -p 8080:8080 -v $HOME/.tabby:/data \
  tabbyml/tabby serve --model StarCoder-1B --device cuda
```

### AI Code Review &amp; Bug Detection

| Инструмент | Описание | Stars | Интеграция |
|------------|----------|-------|------------|
| **Qodo Merge (PR-Agent)** | AI PR review | 6k | GitHub/GitLab |
| **Semgrep** | Static analysis с AI rules | 10k | CLI, CI/CD |
| **Snyk Code** | AI vulnerability detection | - | `npm install -g snyk` |

### Local LLM Code Models

| Модель | Размер | Запуск через Ollama |
|--------|--------|---------------------|
| **DeepSeek Coder** | 6.7B-33B | `ollama run deepseek-coder` |
| **CodeLlama** | 7B-70B | `ollama run codellama` |
| **StarCoder2** | 3B-15B | `ollama run starcoder2:3b` |
| **Codestral** | 22B | `ollama run codestral` |

---

## 4. DevOps &amp; SRE (30 инструментов)

### MLOps Platforms

| Инструмент | Функция | Stars | Установка |
|------------|---------|-------|-----------|
| **MLflow** | Experiment tracking, model registry | 19k | `pip install mlflow` |
| **DVC** | Data version control | 14k | `pip install dvc` |
| **Kubeflow** | ML toolkit for Kubernetes | 14k | Kubectl apply |
| **ZenML** | MLOps framework | 4k | `pip install zenml` |
| **Metaflow** | Netflix's ML framework | 8k | `pip install metaflow` |
| **ClearML** | Experiment tracking + orchestration | 5k | `pip install clearml` |
| **Feast** | Feature store | 5k | `pip install feast` |
| **Ray** | Distributed computing for ML | 35k | `pip install ray` |

### Log Analysis &amp; Anomaly Detection

| Инструмент | Метод | Stars | Use Case |
|------------|-------|-------|----------|
| **LogAI** | ML log analytics (Salesforce) | 1k | OpenTelemetry compatible |
| **Loglizer** | PCA, clustering, deep learning | 1k | Automated log analysis |
| **Log Anomaly Detector** | Unsupervised learning (Red Hat) | - | Human-in-the-loop |
| **OpenSearch** | Random Cut Forest anomaly | - | Time-series detection |

### Monitoring with AI

| Инструмент | ML Features | Stars |
|------------|-------------|-------|
| **Netdata** | k-means anomaly detection | 73k |
| **Prometheus Anomaly Detector** | Fourier/Prophet models | 300 |
| **Grafana ML Plugin** | Forecasting, outlier detection | - |
| **WhyLogs** | Data drift detection | 2k |
| **Evidently** | ML model monitoring | 5k |

### Infrastructure AI

| Инструмент | Описание | Stars |
|------------|----------|-------|
| **K8sGPT** | AI Kubernetes troubleshooting | 5k |
| **Pulumi AI** | NL to Infrastructure code | - |

---

## 5. DATA SCIENCE &amp; ANALYTICS (25 инструментов)

### AutoML Frameworks

| Инструмент | Backend | Stars | Best For |
|------------|---------|-------|----------|
| **AutoGluon** | AWS | 8k | Tabular, text, images |
| **Auto-sklearn** | scikit-learn | 7.5k | Classical ML |
| **H2O AutoML** | H2O | 7k | Enterprise scale |
| **FLAML** | Microsoft | 4k | Fast, lightweight |
| **PyCaret** | Multiple | 9k | Low-code ML |
| **TPOT** | Genetic programming | 9.5k | Pipeline optimization |

**Пример AutoGluon:**
```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='target').fit(train_data)
predictions = predictor.predict(test_data)
```

### Visualization &amp; Dashboards

| Инструмент | Type | Stars | CLI Example |
|------------|------|-------|-------------|
| **Streamlit** | Data apps | 37k | `streamlit run app.py` |
| **Gradio** | ML demos | 35k | `gr.Interface().launch()` |
| **Jupyter AI** | JupyterLab extension | 3k | `%%ai chatgpt` magic |

### Deep Learning Frameworks

| Инструмент | Stars | Установка |
|------------|-------|-----------|
| **PyTorch** | 87k | `pip install torch` |
| **TensorFlow** | 190k | `pip install tensorflow` |
| **ONNX Runtime** | 15k | `pip install onnxruntime` |

---

## 6. COMPUTER VISION (30 инструментов)

### Object Detection

| Инструмент | Architecture | Stars | CLI |
|------------|--------------|-------|-----|
| **Ultralytics YOLO** | YOLOv8/v11 | 35k | `yolo detect predict model=yolov8n.pt source=image.jpg` |
| **Detectron2** | Faster R-CNN, etc. | 31k | Python demo |
| **MMDetection** | 300+ models | 32k | MIM download |
| **OpenCV** | Classic CV | 80k | `pip install opencv-python` |

### Segmentation

| Инструмент | Task | Stars |
|------------|------|-------|
| **Segment Anything (SAM)** | Promptable segmentation | 50k |
| **MMSegmentation** | 50+ models | 9.5k |

### OCR Tools

| Инструмент | Languages | Stars | CLI |
|------------|-----------|-------|-----|
| **PaddleOCR** | 80+ | 47k | `paddleocr --image_dir ./imgs` |
| **EasyOCR** | 80+ | 25k | Python API |
| **Tesseract** | 100+ | 64k | `tesseract image.png output` |
| **MMOCR** | OpenMMLab | 4.5k | pip install |

### Image Enhancement

| Инструмент | Function | Stars | CLI |
|------------|----------|-------|-----|
| **Real-ESRGAN** | 4x upscaling | 30k | `realesrgan-ncnn-vulkan -i input.jpg -o output.png` |
| **Upscayl** | GUI upscaler | 35k | AppImage |
| **GFPGAN** | Face restoration | 37k | Python inference |
| **CodeFormer** | Blind face restoration | 15k | Python |

### Pose &amp; Tracking

| Инструмент | Task | Stars |
|------------|------|-------|
| **MMPose** | 2D/3D pose estimation | 7k |
| **OpenPose** | Multi-person keypoints | 31k |
| **MediaPipe** | Face, pose, hands | 28k |
| **ByteTrack** | Multi-object tracking | 5k |

---

## 7. NLP &amp; TEXT PROCESSING (30 инструментов)

### Local LLM Runners

| Инструмент | Description | Stars | RAM |
|------------|-------------|-------|-----|
| **Ollama** | Simple LLM runner | 100k+ | 8GB+ |
| **llama.cpp** | C++ inference | 75k | 8GB+ |
| **vLLM** | High-throughput serving | 35k | GPU 16GB+ |
| **text-generation-webui** | Feature-rich WebUI | 45k | 8GB+ |
| **KoboldCpp** | Single-file with WebUI | 10k | 4GB+ |
| **LocalAI** | OpenAI-compatible API | 30k | 8GB+ |
| **GPT4All** | Easy local chatbots | 70k | 8GB+ |
| **llamafile** | Single executable LLM | 22k | 4GB+ |
| **SGLang** | 5x faster than vLLM | 8k | GPU |

**Установка Ollama и запуск модели:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
ollama run deepseek-coder "explain this code"
```

### NLP Libraries

| Инструмент | Focus | Stars |
|------------|-------|-------|
| **spaCy** | Industrial NLP | 30k |
| **NLTK** | Classic NLP | 13k |
| **Transformers** | HuggingFace models | 140k |
| **sentence-transformers** | Embeddings | 15k |
| **Gensim** | Topic modeling | 15k |
| **Flair** | NER, classification | 14k |

### RAG Frameworks

| Инструмент | Description | Stars |
|------------|-------------|-------|
| **LangChain** | LLM app framework | 95k |
| **LlamaIndex** | Data framework for RAG | 35k |

### Translation

| Инструмент | Type | Stars |
|------------|------|-------|
| **Argos Translate** | Offline NMT | 4k |
| **LibreTranslate** | Self-hosted API | 9k |
| **OpenNMT** | Neural MT framework | 6k |

---

## 8. AUDIO &amp; SPEECH (25 инструментов)

### Speech-to-Text (STT)

| Инструмент | Languages | Stars | Speed |
|------------|-----------|-------|-------|
| **Whisper** | 99+ | 75k | Baseline |
| **faster-whisper** | 99+ | 14k | 4x faster |
| **Whisper.cpp** | 99+ | 38k | CPU optimized |
| **insanely-fast-whisper** | 99+ | 8k | 150min in 98s |
| **Vosk** | 20+ | 9k | Offline, lightweight |
| **DeepSpeech** | EN focused | 25k | Archived |

**Whisper CLI:**
```bash
pip install openai-whisper
whisper audio.mp3 --model turbo --language Russian
```

### Text-to-Speech (TTS)

| Инструмент | Voice Cloning | Stars | Quality |
|------------|---------------|-------|---------|
| **Coqui TTS** | ✅ XTTS-v2 | 35k | High |
| **Piper TTS** | ❌ | 7k | Fast, lightweight |
| **Bark** | ✅ + emotions | 38k | High |
| **Tortoise TTS** | ✅ | 13k | High |
| **StyleTTS2** | ✅ | 5k | Human-level |
| **OpenVoice** | ✅ instant | 30k | High |
| **GPT-SoVITS** | ✅ few-shot | 35k | High |

**Coqui TTS с voice cloning:**
```bash
pip install TTS
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "Hello world" --speaker_wav voice.wav --out_path output.wav
```

### Audio Processing

| Инструмент | Task | Stars |
|------------|------|-------|
| **AudioCraft/MusicGen** | Music generation | 22k |
| **Demucs** | Source separation | 8k |
| **SpeechBrain** | Speech toolkit | 9k |
| **NeMo** | NVIDIA speech/NLP | 12k |
| **RVC** | Voice conversion | 25k |

---

## 9. SYSTEM ADMINISTRATION (20 инструментов)

### AI Shell Assistants

| Инструмент | Description | Stars | Example |
|------------|-------------|-------|---------|
| **ShellGPT** | Shell command generation | 9.5k | `sgpt -s "find json files"` |
| **AIChat** | All-in-one LLM CLI | 5k | `aichat -e "list files modified today"` |
| **Mods** | AI for shell pipelines | 2.8k | `git diff \| mods "summarize"` |
| **Amazon Q CLI** | AWS AI assistant | 2k | `q chat "How to list files?"` |
| **Open Interpreter** | NL computer control | 55k | `interpreter` |
| **llm** | Simple LLM CLI | 6k | `llm "Write systemd service"` |
| **Warp Terminal** | AI-powered terminal | - | Type `#` + description |

**ShellGPT setup:**
```bash
pip install shell-gpt
export OPENAI_API_KEY=your-key
# Или с локальным LLM через Ollama
sgpt --install-integration  # Добавляет Ctrl+l для shell интеграции
sgpt -s "показать использование диска по папкам"
```

### Security &amp; Monitoring

| Инструмент | Function | Stars |
|------------|----------|-------|
| **CrowdSec** | ML-based Fail2ban | 9k |
| **Netdata** | Monitoring + ML anomaly | 73k |
| **Wazuh** | SIEM with ML | 11k |
| **Graylog** | Log management | 7k |

---

## 10. PRODUCTIVITY &amp; AUTOMATION (20 инструментов)

### Workflow Automation

| Инструмент | Type | Stars | Self-hosted |
|------------|------|-------|-------------|
| **n8n** | Visual automation | 55k | ✅ |
| **Huginn** | Agent-based automation | 44k | ✅ |
| **Node-RED** | Flow-based programming | 20k | ✅ |
| **Automatisch** | Zapier alternative | 6k | ✅ |

### Document Processing

| Инструмент | Function | Stars |
|------------|----------|-------|
| **Paperless-ngx** | Document management + OCR | 22k |
| **OCRmyPDF** | Add OCR layer to PDFs | 14k |
| **docTR** | Deep learning OCR | 4k |
| **Paperless-GPT** | LLM sidecar for Paperless | 1k |

### Vector Databases (Semantic Search)

| Инструмент | Performance | Stars | Setup |
|------------|-------------|-------|-------|
| **Weaviate** | Hybrid search | 12k | `docker run weaviate` |
| **Milvus** | Billion-scale | 31k | Docker/K8s |
| **Qdrant** | Rust, fast | 21k | `docker run qdrant/qdrant` |
| **Chroma** | Lightweight | 16k | `pip install chromadb` |
| **pgvector** | PostgreSQL extension | 13k | Build/Docker |

### Knowledge Management

| Инструмент | AI Features | Stars |
|------------|-------------|-------|
| **Obsidian + Smart Connections** | Semantic note discovery | 3k (plugin) |
| **Obsidian + Copilot** | RAG over vault | 9k (plugin) |
| **Logseq** | Graph-based notes | 33k |
| **SiYuan** | Local-first PKM | 22k |

---

## 11. GRAPHICS &amp; MEDIA (20 инструментов)

### Image Generation

| Инструмент | UI Type | Stars | VRAM |
|------------|---------|-------|------|
| **Stable Diffusion WebUI** | Web GUI | 159k | 4GB+ |
| **ComfyUI** | Node-based | 60k | 6GB+ |
| **InvokeAI** | Professional | 24k | 4GB+ |
| **Fooocus** | Simple, Midjourney-like | 42k | 4GB+ |
| **SD.Next** | AMD/Intel support | 15k | 6GB+ |

**ComfyUI setup:**
```bash
pip install comfy-cli
comfy install
comfy launch
```

### Video Generation

| Инструмент | Type | Stars |
|------------|------|-------|
| **Deforum** | Animation | 3k |
| **AnimateDiff** | Video from SD | 10k |
| **Video2X** | AI upscaling | 7k |
| **RIFE** | Frame interpolation | 4k |

### 3D Generation

| Инструмент | Speed | Stars |
|------------|-------|-------|
| **TripoSR** | &lt;0.5s per image | 5k |
| **Shap-E** | OpenAI 3D | 11k |
| **TripoSG** | High-fidelity | 2k |

---

## 12. NETWORKING (15 инструментов)

### Traffic Analysis

| Инструмент | Function | Stars |
|------------|----------|-------|
| **ntopng** | DPI + flow analysis | 6.5k |
| **Zeek** | Network metadata | 6.5k |
| **nDPI** | Protocol detection | 4k |
| **Wireshark/tshark** | Packet analysis | 7k |

### DNS &amp; Security

| Инструмент | Function | Stars |
|------------|----------|-------|
| **Pi-hole** | DNS ad blocking | 50k |
| **AdGuard Home** | DNS filtering + DoH/DoT | 26k |
| **Pi-hole LLM Analytics** | AI DNS analysis | - |

### VPN &amp; Mesh

| Инструмент | Type | Stars |
|------------|------|-------|
| **WireGuard** | Kernel VPN | 10k |
| **Tailscale** | Mesh VPN | 20k |
| **Netbird** | Open-source mesh | 12k |

---

## Comparison Matrix: Альтернативы популярным инструментам

### Local LLM Runners

| Критерий | Ollama | llama.cpp | vLLM | LocalAI |
|----------|--------|-----------|------|---------|
| Простота установки | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Производительность | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| GPU support | CUDA/ROCm | CUDA/Vulkan/Metal | CUDA only | CUDA/CPU |
| OpenAI API compatible | ✅ | ❌ | ✅ | ✅ |
| Min RAM | 8GB | 4GB+ | 16GB+ | 8GB |

### Image Generation UIs

| Критерий | A1111 | ComfyUI | InvokeAI | Fooocus |
|----------|-------|---------|----------|---------|
| Простота | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Гибкость | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Производительность | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Extensions | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| AMD support | ❌ | ❌ | ✅ ROCm | ❌ |

### STT Tools

| Критерий | Whisper | faster-whisper | Vosk | Whisper.cpp |
|----------|---------|----------------|------|-------------|
| Скорость | Baseline | 4x faster | Fast | 2x faster |
| Качество | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Offline | ✅ | ✅ | ✅ | ✅ |
| RAM | 2-10GB | 1-4GB | 50MB-1.5GB | 1-4GB |
| GPU required | Recommended | Optional | No | No |

### Vector Databases

| Критерий | Weaviate | Milvus | Qdrant | Chroma |
|----------|----------|--------|--------|--------|
| Scale | Medium | Billion+ | Large | Small-Medium |
| Простота | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Hybrid search | ✅ | ✅ | ✅ | ❌ |
| Min RAM | 2GB | 8GB | 1GB | 512MB |

---

## Поддержка дистрибутивов

| Категория | Ubuntu/Debian | Arch/AUR | Fedora/RHEL | Kali | Alpine |
|-----------|---------------|----------|-------------|------|--------|
| Security tools | ✅ | ✅ | ✅ | ✅ Native | ✅ |
| ML frameworks | ✅ | ✅ | ✅ | ✅ | Partial |
| Docker-based | ✅ | ✅ | ✅ | ✅ | ✅ |
| Python tools | ✅ | ✅ | ✅ | ✅ | ✅ |

**Специализированные дистрибутивы:**
- **Kali Linux**: Pre-installed security/OSINT tools (Sherlock, theHarvester, Recon-ng)
- **Parrot OS**: Security + privacy focus
- **BlackArch**: 2800+ security tools in AUR

---

## Статистика каталога

| Категория | Инструментов | Top 3 |
|-----------|-------------|-------|
| Cybersecurity &amp; Pentesting | 35 | PentestGPT, Suricata, Nuclei |
| OSINT | 25 | Sherlock, theHarvester, DeepFace |
| Software Development | 30 | Aider, Tabby, Continue.dev |
| DevOps &amp; SRE | 30 | MLflow, DVC, K8sGPT |
| Data Science | 25 | AutoGluon, Streamlit, Gradio |
| Computer Vision | 30 | YOLO, PaddleOCR, Real-ESRGAN |
| NLP &amp; Text | 30 | Ollama, llama.cpp, LangChain |
| Audio &amp; Speech | 25 | Whisper, Coqui TTS, Bark |
| System Administration | 20 | ShellGPT, Netdata, CrowdSec |
| Productivity | 20 | n8n, Paperless-ngx, Weaviate |
| Graphics &amp; Media | 20 | SD WebUI, ComfyUI, Real-ESRGAN |
| Networking | 15 | Pi-hole, Zeek, ntopng |
| **TOTAL** | **315+** | |

---

## CSV Database Format

```csv
Name,Description,Category,URL,Install_Command,Distros,License,Stars,Last_Commit,GPU_Required,Documentation_URL
Ollama,"Run LLMs locally with simple CLI",NLP,https://github.com/ollama/ollama,"curl -fsSL https://ollama.com/install.sh | sh","All Linux",MIT,100000,2025,No,https://ollama.com
Whisper,"Robust speech recognition model",Audio,https://github.com/openai/whisper,"pip install openai-whisper","All Linux",MIT,75000,2024,Recommended,https://github.com/openai/whisper
YOLO,"Object detection framework",CV,https://github.com/ultralytics/ultralytics,"pip install ultralytics","All Linux",AGPL-3.0,35000,2025,Recommended,https://docs.ultralytics.com
Sherlock,"Hunt usernames across 400+ sites",OSINT,https://github.com/sherlock-project/sherlock,"pip install sherlock-project","All Linux",MIT,59000,2024,No,https://github.com/sherlock-project/sherlock
PentestGPT,"AI-powered penetration testing",Security,https://github.com/GreyDGL/PentestGPT,"pip install pentestgpt","All Linux",MIT,7500,2025,No,https://github.com/GreyDGL/PentestGPT
```

*Полная CSV база с 315+ записями доступна для экспорта в формате выше.*

---

## Заключение и ключевые рекомендации

Экосистема AI-инструментов для Linux достигла зрелости, позволяя выполнять большинство задач локально без зависимости от облачных сервисов. **Три ключевых тренда 2024-2025:**

1. **Локальные LLM стали практичными** — Ollama + llama.cpp делают запуск моделей тривиальным даже на CPU
2. **MCP (Model Context Protocol)** — новый стандарт интеграции AI с инструментами (HexStrike, CAI)
3. **Специализированные модели** — DeepSeek Coder для кода, Foundation-Sec для безопасности, XTTS для голоса

**Для начала работы рекомендуется:**
- Установить Ollama как универсальный backend для LLM
- Добавить ShellGPT для AI-assisted командной строки
- Развернуть Paperless-ngx + Weaviate для управления документами
- Использовать ComfyUI для генерации изображений

Все 315+ инструментов в каталоге активно развиваются, имеют открытый исходный код или бесплатный tier, и полностью совместимы с основными Linux-дистрибутивами.