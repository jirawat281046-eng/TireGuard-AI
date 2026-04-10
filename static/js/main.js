document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultSection = document.getElementById('result-section');
    const resultCard = document.getElementById('result-card');
    const resultLabel = document.getElementById('result-label');
    const resultExplanation = document.getElementById('result-explanation');
    const confidencePct = document.getElementById('confidence-pct');
    const progressFill = document.getElementById('progress-fill');
    const resetBtn = document.getElementById('reset-btn');
    const statusIcon = document.getElementById('status-icon');

    // Camera Elements
    const startCameraBtn = document.getElementById('start-camera-btn');
    const cameraContainer = document.getElementById('camera-container');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const closeCameraBtn = document.getElementById('close-camera-btn');

    let selectedFile = null;
    let stream = null;

    // Trigger file input on click
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#start-camera-btn')) return;
        fileInput.click();
    });

    // Camera logic
    startCameraBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });
            video.srcObject = stream;
            cameraContainer.classList.remove('hidden');
            uploadContent.classList.add('hidden');
            previewContainer.classList.add('hidden');
            resultSection.classList.add('hidden');
        } catch (err) {
            console.error("Camera error:", err);
            alert("Could not access camera. Please check permissions.");
        }
    });

    closeCameraBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        stopCamera();
    });

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        cameraContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
    }

    captureBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            selectedFile = new File([blob], "capture.jpg", { type: "image/jpeg" });
            imagePreview.src = URL.createObjectURL(blob);
            stopCamera();
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.classList.remove('disabled');
            predictBtn.disabled = false;
        }, 'image/jpeg');
    });

    // Drag and drop handlers

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) {
            alert('โปรดอัปโหลดไฟล์รูปภาพเท่านั้น (JPG, PNG)');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadContent.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            predictBtn.classList.remove('disabled');
            predictBtn.disabled = false;
            resultSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        uploadContent.classList.remove('hidden');
        predictBtn.classList.add('disabled');
        predictBtn.disabled = true;
        resultSection.classList.add('hidden');
    });

    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // UI Loading State
        const btnText = predictBtn.querySelector('.btn-text');
        const loader = predictBtn.querySelector('.loader');
        const icon = predictBtn.querySelector('i');

        predictBtn.disabled = true;
        btnText.textContent = 'กำลังวิเคราะห์...';
        loader.classList.remove('hidden');
        icon.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                showResult(data);
            } else {
                alert('เกิดข้อผิดพลาดในการวิเคราะห์: ' + data.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('เกิดข้อผิดพลาดในการเชื่อมต่อ กรุณาตรวจสอบว่าเซิร์ฟเวอร์เปิดอยู่');
        } finally {
            predictBtn.disabled = false;
            btnText.textContent = 'เริ่มวิเคราะห์สภาพยาง';
            loader.classList.add('hidden');
            icon.classList.remove('hidden');
        }
    });

    function showResult(data) {
        resultSection.classList.remove('hidden');
        resultLabel.textContent = data.label;
        confidencePct.textContent = data.confidence;
        
        const inferenceTimeSpan = document.getElementById('inference-time');
        if (inferenceTimeSpan && data.inference_time_ms) {
            inferenceTimeSpan.textContent = data.inference_time_ms;
        }

        // Update bar width
        setTimeout(() => {
            progressFill.style.width = data.confidence;
        }, 100);

        // Update UI based on class
        if (data.label === 'ปกติ') {
            statusIcon.className = 'status-icon good';
            statusIcon.innerHTML = '<i class="fas fa-check-circle"></i>';
            resultExplanation.textContent = 'ยางเส้นนี้อยู่ในสภาพปกติและปลอดภัยสำหรับการใช้งาน';
        } else if (data.label === 'ไม่สามารถวิเคราะห์ได้') {
            statusIcon.className = 'status-icon warning';
            statusIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            resultExplanation.textContent = ''; // CLEAR text in the red circle area as requested
        } else {
            statusIcon.className = 'status-icon defective';
            statusIcon.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
            resultExplanation.textContent = 'พบความชำรุดหรือรอยเสียหาย ควรตรวจสอบอย่างละเอียดเพื่อความปลอดภัย';
        }

        // Show advice
        const adviceCard = document.getElementById('advice-card');
        const adviceText = document.getElementById('advice-text');
        if (data.advice) {
            adviceCard.classList.remove('hidden');
            adviceText.textContent = data.advice;
        } else {
            adviceCard.classList.add('hidden');
        }

        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    resetBtn.addEventListener('click', () => {
        removeBtn.click();
        resultSection.classList.add('hidden');
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});
