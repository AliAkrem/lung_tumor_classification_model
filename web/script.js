// Lung Tumor Classification - Main JavaScript Application
// Updated to use @gradio/client for API calls

class LungTumorClassifier {
  constructor() {
    this.spaceUrl = "aliakrem/lung_tumor_prediction";
    this.gradioClient = null;
    this.chart = null;
    this.currentImage = null;

    this.initializeElements();
    this.bindEvents();
    this.setupDragAndDrop();

    // Initialize Gradio client
    this.initializeGradioClient();
  }

  initializeElements() {
    // Upload elements
    this.uploadArea = document.getElementById("uploadArea");
    this.imageInput = document.getElementById("imageInput");
    this.imagePreview = document.getElementById("imagePreview");
    this.previewImg = document.getElementById("previewImg");
    this.imageInfo = document.getElementById("imageInfo");
    this.clearImage = document.getElementById("clearImage");
    this.analyzeBtn = document.getElementById("analyzeBtn");

    // Results elements
    this.loadingState = document.getElementById("loadingState");
    this.resultsContent = document.getElementById("resultsContent");
    this.emptyState = document.getElementById("emptyState");
    this.statusIndicator = document.getElementById("statusIndicator");
    this.resultIcon = document.getElementById("resultIcon");
    this.resultClass = document.getElementById("resultClass");
    this.resultConfidence = document.getElementById("resultConfidence");
    this.probabilityBars = document.getElementById("probabilityBars");
    this.probabilityChart = document.getElementById("probabilityChart");

    // Modal elements
    this.modal = document.getElementById("modal");
    this.modalTitle = document.getElementById("modalTitle");
    this.modalBody = document.getElementById("modalBody");
    this.modalClose = document.getElementById("modalClose");
    this.modalBackdrop = document.getElementById("modalBackdrop");
    this.aboutBtn = document.getElementById("aboutBtn");
    this.helpBtn = document.getElementById("helpBtn");
  }

  bindEvents() {
    // Upload events
    this.imageInput.addEventListener("change", (e) =>
      this.handleImageSelect(e),
    );
    this.clearImage.addEventListener("click", () => this.clearImageSelection());
    this.analyzeBtn.addEventListener("click", () => this.analyzeImage());

    // Modal events
    this.aboutBtn.addEventListener("click", () => this.showAboutModal());
    this.helpBtn.addEventListener("click", () => this.showHelpModal());
    this.modalClose.addEventListener("click", () => this.hideModal());
    this.modalBackdrop.addEventListener("click", () => this.hideModal());

    // Upload area click
    this.uploadArea.addEventListener("click", () => this.imageInput.click());
  }

  setupDragAndDrop() {
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      this.uploadArea.addEventListener(eventName, this.preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      this.uploadArea.addEventListener(
        eventName,
        () => {
          this.uploadArea.classList.add("drag-over");
        },
        false,
      );
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.uploadArea.addEventListener(
        eventName,
        () => {
          this.uploadArea.classList.remove("drag-over");
        },
        false,
      );
    });

    this.uploadArea.addEventListener("drop", (e) => this.handleDrop(e), false);
  }

  preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      this.processFile(files[0]);
    }
  }

  handleImageSelect(e) {
    if (e.target.files.length > 0) {
      this.processFile(e.target.files[0]);
    }
  }

  processFile(file) {
    // Validate file type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type)) {
      this.showError("Please upload a valid image file (PNG, JPG, JPEG)");
      return;
    }

    // Validate file size (limit to 10MB)
    if (file.size > 10 * 1024 * 1024) {
      this.showError(
        "File size too large. Please upload an image smaller than 10MB.",
      );
      return;
    }

    this.currentImage = file;
    this.displayImagePreview(file);
    this.analyzeBtn.disabled = false;
    this.updateStatus("Ready to analyze", "ready");
  }

  displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewImg.src = e.target.result;
      this.imagePreview.style.display = "block";
      this.uploadArea.style.display = "none";

      // Update image info
      this.imageInfo.innerHTML = `
        <div class="info-item"><strong>File:</strong> ${file.name}</div>
        <div class="info-item"><strong>Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB</div>
        <div class="info-item"><strong>Type:</strong> ${file.type}</div>
      `;
    };
    reader.readAsDataURL(file);
  }

  clearImageSelection() {
    this.currentImage = null;
    this.imagePreview.style.display = "none";
    this.uploadArea.style.display = "block";
    this.analyzeBtn.disabled = true;
    this.imageInput.value = "";
    this.showEmptyState();
    this.updateStatus("Ready", "ready");
  }

  async analyzeImage() {
    if (!this.currentImage) {
      this.showError("No image selected.");
      return;
    }

    this.showLoadingState();
    this.updateStatus("Analyzing...", "processing");

    try {
      // Ensure Gradio client is connected
      await this.ensureGradioConnection();

      // Convert file to blob for Gradio client
      const imageBlob = await this.fileToBlob(this.currentImage);

      console.log("Calling Gradio API with image...");

      // Use the JSON endpoint for structured results
      const result = await this.gradioClient.predict("/predict_tumor_json", {
        image: imageBlob,
      });

      console.log("Gradio API response:", result);

      if (result && result.data && result.data[0]) {
        const apiResult = result.data[0];

        if (apiResult.error) {
          throw new Error(apiResult.error);
        }

        if (apiResult.class && apiResult.probabilities) {
          this.displayResults(apiResult);
          this.updateStatus("Analysis complete", "success");
        } else {
          throw new Error("Invalid response format from API");
        }
      } else {
        throw new Error("No valid response from Gradio API");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      this.showError(`Analysis failed: ${error.message}`);
      this.updateStatus("Error", "error");
    }
  }

  async fileToBlob(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result;
        const blob = new Blob([arrayBuffer], { type: file.type });
        resolve(blob);
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  }

  async initializeGradioClient() {
    try {
      this.updateStatus("Connecting to API...", "processing");

      // Wait for GradioClient to be available
      let attempts = 0;
      while (!window.GradioClient && attempts < 50) {
        await new Promise((resolve) => setTimeout(resolve, 100));
        attempts++;
      }

      if (!window.GradioClient) {
        throw new Error("Gradio client not loaded");
      }

      console.log("Initializing Gradio client...");
      this.gradioClient = await window.GradioClient.connect(this.spaceUrl);
      console.log("Gradio client connected successfully");
      this.updateStatus("Ready", "ready");
    } catch (error) {
      console.error("Failed to initialize Gradio client:", error);
      this.updateStatus("API connection failed", "error");
      this.showError(
        "Failed to connect to the AI service. Please refresh the page and try again.",
      );
    }
  }

  async ensureGradioConnection() {
    if (!this.gradioClient) {
      await this.initializeGradioClient();
    }
  }

  displayResults(data) {
    // Hide loading and empty states
    this.loadingState.style.display = "none";
    this.emptyState.style.display = "none";
    this.resultsContent.style.display = "block";

    // Update primary result
    this.resultClass.textContent = this.formatClassName(data.class);
    this.resultConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    // Update result icon based on class
    this.resultIcon.textContent = this.getClassIcon(data.class);

    // Create probability bars
    this.createProbabilityBars(data.probabilities);

    // Update chart
    this.createChart(data.probabilities);
  }

  formatClassName(className) {
    return className
      .split(".")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  }

  getClassIcon(className) {
    const icons = {
      adenocarcinoma: "🔴",
      "large.cell.carcinoma": "🟠",
      normal: "✅",
      "squamous.cell.carcinoma": "🟡",
    };
    return icons[className] || "🎯";
  }

  getClassColor(className) {
    const colors = {
      adenocarcinoma: "#e74c3c",
      "large.cell.carcinoma": "#f39c12",
      normal: "#27ae60",
      "squamous.cell.carcinoma": "#f1c40f",
    };
    return colors[className] || "#3498db";
  }

  createProbabilityBars(probabilities) {
    // Sort probabilities by value (descending)
    const sortedProbs = Object.entries(probabilities).sort(
      ([, a], [, b]) => b - a,
    );

    this.probabilityBars.innerHTML = "";

    sortedProbs.forEach(([className, probability]) => {
      const percentage = (probability * 100).toFixed(1);
      const formattedName = this.formatClassName(className);
      const color = this.getClassColor(className);
      const icon = this.getClassIcon(className);

      const barContainer = document.createElement("div");
      barContainer.className = "probability-bar-container";
      barContainer.innerHTML = `
        <div class="probability-label">
          <span class="class-icon">${icon}</span>
          <span class="class-name">${formattedName}</span>
          <span class="probability-value">${percentage}%</span>
        </div>
        <div class="probability-bar">
          <div class="probability-fill" style="width: ${percentage}%; background-color: ${color}"></div>
        </div>
      `;

      this.probabilityBars.appendChild(barContainer);
    });
  }

  createChart(probabilities) {
    const ctx = this.probabilityChart.getContext("2d");

    // Destroy existing chart if it exists
    if (this.chart) {
      this.chart.destroy();
    }

    // Sort probabilities by value (descending)
    const sortedProbs = Object.entries(probabilities).sort(
      ([, a], [, b]) => b - a,
    );

    const labels = sortedProbs.map(([name]) => this.formatClassName(name));
    const data = sortedProbs.map(([, prob]) => (prob * 100).toFixed(1));
    const colors = sortedProbs.map(([name]) => this.getClassColor(name));

    this.chart = new Chart(ctx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [
          {
            data: data,
            backgroundColor: colors,
            borderColor: colors.map((color) => color + "40"),
            borderWidth: 2,
            hoverBorderWidth: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              padding: 20,
              usePointStyle: true,
              font: {
                family: "Inter",
                size: 12,
              },
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                return `${context.label}: ${context.parsed}%`;
              },
            },
          },
        },
        elements: {
          arc: {
            borderRadius: 4,
          },
        },
      },
    });
  }

  showLoadingState() {
    this.loadingState.style.display = "flex";
    this.resultsContent.style.display = "none";
    this.emptyState.style.display = "none";
  }

  showEmptyState() {
    this.emptyState.style.display = "flex";
    this.resultsContent.style.display = "none";
    this.loadingState.style.display = "none";
  }

  updateStatus(message, type = "ready") {
    const statusText = this.statusIndicator.querySelector(".status-text");
    const statusDot = this.statusIndicator.querySelector(".status-dot");

    statusText.textContent = message;

    // Remove existing status classes
    this.statusIndicator.classList.remove(
      "status-ready",
      "status-processing",
      "status-success",
      "status-error",
    );

    // Add new status class
    this.statusIndicator.classList.add(`status-${type}`);
  }

  showError(message) {
    // Create error notification
    const errorDiv = document.createElement("div");
    errorDiv.className = "error-notification";
    errorDiv.innerHTML = `
      <div class="error-content">
        <div class="error-icon">⚠️</div>
        <div class="error-message">${message}</div>
        <button class="error-close">×</button>
      </div>
    `;

    // Add to body
    document.body.appendChild(errorDiv);

    // Close button functionality
    const closeBtn = errorDiv.querySelector(".error-close");
    closeBtn.addEventListener("click", () => {
      errorDiv.remove();
    });

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (errorDiv.parentNode) {
        errorDiv.remove();
      }
    }, 5000);

    // Show empty state
    this.showEmptyState();
  }

  showAboutModal() {
    this.modalTitle.textContent = "About Lung Tumor Classification";
    this.modalBody.innerHTML = `
      <div class="modal-section">
        <h4>🫁 Advanced AI-Powered Medical Imaging</h4>
        <p>This application uses state-of-the-art EfficientNet deep learning architecture to analyze lung CT scan images and classify potential tumor types.</p>
      </div>

      <div class="modal-section">
        <h4>📊 Classification Categories</h4>
        <ul class="class-list">
          <li><span class="class-icon">🔴</span> <strong>Adenocarcinoma:</strong> Most common type of lung cancer</li>
          <li><span class="class-icon">🟠</span> <strong>Large Cell Carcinoma:</strong> Fast-growing cancer type</li>
          <li><span class="class-icon">✅</span> <strong>Normal:</strong> Healthy lung tissue</li>
          <li><span class="class-icon">🟡</span> <strong>Squamous Cell Carcinoma:</strong> Cancer in thin, flat cells</li>
        </ul>
      </div>

      <div class="modal-section">
        <h4>🔬 Model Performance</h4>
        <div class="performance-stats">
          <div class="stat-item">
            <span class="stat-label">Architecture:</span>
            <span class="stat-value">EfficientNet</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Input Size:</span>
            <span class="stat-value">299×299 pixels</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">Classes:</span>
            <span class="stat-value">4 categories</span>
          </div>
        </div>
      </div>

      <div class="modal-section">
        <h4>⚠️ Important Disclaimer</h4>
        <p class="disclaimer">This tool is for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.</p>
      </div>
    `;
    this.showModal();
  }

  showHelpModal() {
    this.modalTitle.textContent = "How to Use";
    this.modalBody.innerHTML = `
      <div class="modal-section">
        <h4>📋 Step-by-Step Guide</h4>
        <ol class="help-steps">
          <li><strong>Upload Image:</strong> Click the upload area or drag and drop a lung CT scan image (PNG, JPG, JPEG format)</li>
          <li><strong>Review Preview:</strong> Check that your image has been loaded correctly in the preview</li>
          <li><strong>Analyze:</strong> Click the "Analyze Image" button to start the AI classification</li>
          <li><strong>View Results:</strong> Review the prediction results, confidence scores, and probability distribution</li>
        </ol>
      </div>

      <div class="modal-section">
        <h4>📁 Supported Formats</h4>
        <ul class="format-list">
          <li>PNG (.png)</li>
          <li>JPEG (.jpg, .jpeg)</li>
          <li>Maximum file size: 10MB</li>
        </ul>
      </div>

      <div class="modal-section">
        <h4>📊 Understanding Results</h4>
        <ul class="result-guide">
          <li><strong>Primary Prediction:</strong> The most likely classification with confidence percentage</li>
          <li><strong>Probability Bars:</strong> Detailed breakdown of all class probabilities</li>
          <li><strong>Visual Chart:</strong> Graphical representation of the prediction distribution</li>
        </ul>
      </div>

      <div class="modal-section">
        <h4>🔧 Troubleshooting</h4>
        <ul class="troubleshooting">
          <li>Ensure your image is clear and properly formatted</li>
          <li>Check your internet connection for API access</li>
          <li>Try refreshing the page if you encounter errors</li>
          <li>Contact support if issues persist</li>
        </ul>
      </div>
    `;
    this.showModal();
  }

  showModal() {
    this.modal.style.display = "flex";
    document.body.style.overflow = "hidden";
  }

  hideModal() {
    this.modal.style.display = "none";
    document.body.style.overflow = "auto";
  }
}

// Initialize the application when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.lungTumorApp = new LungTumorClassifier();
});

// Add error notification styles
const style = document.createElement("style");
style.textContent = `
  .error-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: #fee;
    border: 1px solid #fcc;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    z-index: 10000;
    max-width: 400px;
    animation: slideIn 0.3s ease-out;
  }

  .error-content {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .error-icon {
    font-size: 20px;
  }

  .error-message {
    flex: 1;
    color: #d63384;
    font-weight: 500;
  }

  .error-close {
    background: none;
    border: none;
    font-size: 18px;
    cursor: pointer;
    color: #d63384;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  .probability-bar-container {
    margin-bottom: 12px;
  }

  .probability-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
    font-size: 14px;
  }

  .class-icon {
    margin-right: 8px;
  }

  .class-name {
    flex: 1;
    font-weight: 500;
  }

  .probability-value {
    font-weight: 600;
    color: #333;
  }

  .probability-bar {
    height: 8px;
    background-color: #e9ecef;
    border-radius: 4px;
    overflow: hidden;
  }

  .probability-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .status-ready .status-dot {
    background-color: #28a745;
  }

  .status-processing .status-dot {
    background-color: #ffc107;
    animation: pulse 2s infinite;
  }

  .status-success .status-dot {
    background-color: #28a745;
  }

  .status-error .status-dot {
    background-color: #dc3545;
  }

  @keyframes pulse {
    0% {
      transform: scale(0.95);
      box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.7);
    }

    70% {
      transform: scale(1);
      box-shadow: 0 0 0 10px rgba(255, 193, 7, 0);
    }

    100% {
      transform: scale(0.95);
      box-shadow: 0 0 0 0 rgba(255, 193, 7, 0);
    }
  }
`;
document.head.appendChild(style);
