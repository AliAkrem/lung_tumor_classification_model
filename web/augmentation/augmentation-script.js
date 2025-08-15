// Image Augmentation Script - Zed Design System Implementation

class ImageAugmentationApp {
  constructor() {
    this.spaceUrl = "aliakrem/aug";
    this.gradioClient = null;
    this.currentImage = null;
    this.augmentationResults = null;

    // DOM Elements
    this.uploadArea = document.getElementById("uploadArea");
    this.imageInput = document.getElementById("imageInput");
    this.imagePreview = document.getElementById("imagePreview");
    this.previewImg = document.getElementById("previewImg");
    this.imageInfo = document.getElementById("imageInfo");
    this.clearImageBtn = document.getElementById("clearImage");
    this.augmentBtn = document.getElementById("augmentBtn");

    this.statusIndicator = document.getElementById("statusIndicator");
    this.statusText = this.statusIndicator.querySelector(".status-text");
    this.loadingState = document.getElementById("loadingState");
    this.resultsContent = document.getElementById("resultsContent");
    this.emptyState = document.getElementById("emptyState");

    this.augmentationGrid = document.getElementById("augmentationGrid");
    this.downloadAllBtn = document.getElementById("downloadAllBtn");

    // Modal elements
    this.modal = document.getElementById("modal");
    this.modalBackdrop = document.getElementById("modalBackdrop");
    this.modalTitle = document.getElementById("modalTitle");
    this.modalBody = document.getElementById("modalBody");
    this.modalClose = document.getElementById("modalClose");
    this.aboutBtn = document.getElementById("aboutBtn");
    this.helpBtn = document.getElementById("helpBtn");

    // Image modal for fullscreen view
    this.createImageModal();

    this.init();
  }

  init() {
    this.setupEventListeners();
    this.initializeGradioClient();
    this.updateStatus("ready", "Ready");
  }

  async initializeGradioClient() {
    try {
      this.updateStatus("processing", "Connecting to API...");

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
      this.updateStatus("ready", "Ready");
    } catch (error) {
      console.error("Failed to initialize Gradio client:", error);
      this.updateStatus("error", "API connection failed");
      this.showError(
        "Failed to connect to the AI service. Please refresh the page and try again.",
      );
    }
  }

  setupEventListeners() {
    // File upload events
    this.uploadArea.addEventListener("click", () => this.imageInput.click());
    this.uploadArea.addEventListener(
      "dragover",
      this.handleDragOver.bind(this),
    );
    this.uploadArea.addEventListener(
      "dragleave",
      this.handleDragLeave.bind(this),
    );
    this.uploadArea.addEventListener("drop", this.handleDrop.bind(this));
    this.imageInput.addEventListener(
      "change",
      this.handleFileSelect.bind(this),
    );

    // Control buttons
    this.clearImageBtn.addEventListener("click", this.clearImage.bind(this));
    this.augmentBtn.addEventListener(
      "click",
      this.generateAugmentations.bind(this),
    );
    this.downloadAllBtn.addEventListener("click", this.downloadAll.bind(this));

    // Modal events
    this.aboutBtn.addEventListener("click", () => this.showModal("about"));
    this.helpBtn.addEventListener("click", () => this.showModal("help"));
    this.modalClose.addEventListener("click", this.hideModal.bind(this));
    this.modalBackdrop.addEventListener("click", this.hideModal.bind(this));

    // Keyboard shortcuts
    document.addEventListener("keydown", this.handleKeydown.bind(this));
  }

  handleDragOver(e) {
    e.preventDefault();
    this.uploadArea.classList.add("drag-over");
  }

  handleDragLeave(e) {
    e.preventDefault();
    this.uploadArea.classList.remove("drag-over");
  }

  handleDrop(e) {
    e.preventDefault();
    this.uploadArea.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      this.processFile(files[0]);
    }
  }

  handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
      this.processFile(file);
    }
  }

  processFile(file) {
    if (!this.isValidImageFile(file)) {
      this.showError("Please select a valid image file (PNG, JPG, JPEG)");
      return;
    }

    this.currentImage = file;
    this.displayImagePreview(file);
    this.augmentBtn.disabled = false;
    this.updateStatus("ready", "Image Ready");
  }

  isValidImageFile(file) {
    const validTypes = ["image/png", "image/jpeg", "image/jpg"];
    return validTypes.includes(file.type);
  }

  displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewImg.src = e.target.result;
      this.imagePreview.style.display = "block";

      // Update image info
      const sizeKB = (file.size / 1024).toFixed(1);
      this.imageInfo.textContent = `${file.name} • ${sizeKB} KB`;
    };
    reader.readAsDataURL(file);
  }

  clearImage() {
    this.currentImage = null;
    this.imageInput.value = "";
    this.imagePreview.style.display = "none";
    this.augmentBtn.disabled = true;
    this.hideResults();
    this.updateStatus("ready", "Ready");
  }

  async generateAugmentations() {
    if (!this.currentImage) {
      this.showError("No image selected.");
      return;
    }

    this.showLoading();
    this.updateStatus("processing", "Processing...");

    try {
      // Test connection first
      if (!this.gradioClient) {
        this.updateStatus("processing", "Connecting to API...");
        await this.initializeGradioClient();
      }

      // Convert file to blob for Gradio client - same as main script
      const imageBlob = await this.fileToBlob(this.currentImage);

      console.log("Calling Gradio API with image...");
      console.log("Space URL:", this.spaceUrl);
      console.log("Image blob size:", imageBlob.size, "bytes");

      const result = await this.gradioClient.predict(
        "/create_individual_augmentations",
        {
          image: imageBlob,
        },
      );

      console.log("Gradio API response:", result);

      if (result && result.data) {
        // The API returns an array of 8 image objects with url properties
        if (Array.isArray(result.data) && result.data.length >= 1) {
          // Extract URLs from the response objects
          const imageUrls = result.data
            .map((item) => {
              if (typeof item === "string") {
                return item;
              } else if (item && item.url) {
                return item.url;
              } else if (item && item.path) {
                // Fallback to path if url not available
                return item.path;
              }
              return null;
            })
            .filter((url) => url !== null);

          console.log("Extracted image URLs:", imageUrls);
          this.augmentationResults = imageUrls;
          this.displayAugmentations(imageUrls);
          this.updateStatus("success", "Complete");
        } else {
          console.log("Unexpected data format:", result.data);
          throw new Error(
            `Invalid response format - got ${result.data.length} items, expected at least 1`,
          );
        }
      } else {
        console.log("No data in response:", result);
        throw new Error("No valid response from Gradio API");
      }
    } catch (error) {
      console.error("Augmentation error:", error);
      console.error("Error details:", {
        name: error.name,
        message: error.message,
        stack: error.stack,
      });

      let errorMessage = "Failed to generate augmentations";
      if (error.message.includes("ERR_FILE_NOT_FOUND")) {
        errorMessage = "API connection issue - please try again";
      } else if (error.message.includes("fetch")) {
        errorMessage = "Network error - check your connection";
      } else {
        errorMessage = `Error: ${error.message}`;
      }

      this.showError(errorMessage);
      this.updateStatus("error", "Failed");
    }
  }

  displayAugmentations(results) {
    this.hideLoading();
    this.showResults();

    // Map results to image elements
    const imageElements = [
      "originalImage",
      "rotationImage",
      "widthShiftImage",
      "heightShiftImage",
      "shearImage",
      "zoomImage",
      "brightnessImage",
      "channelShiftImage",
    ];

    imageElements.forEach((elementId, index) => {
      const img = document.getElementById(elementId);
      const container = img.closest(".augmentation-image-container");
      const item = img.closest(".augmentation-item");

      if (results[index]) {
        container.classList.add("loading");

        img.onload = () => {
          container.classList.remove("loading");
          item.classList.add("loaded");
        };

        img.onerror = () => {
          container.classList.remove("loading");
          item.classList.add("error");
        };

        img.src = results[index];
      } else {
        item.classList.add("error");
      }
    });

    // Setup download buttons
    this.setupDownloadButtons();

    // Setup image click for fullscreen
    this.setupImageClickHandlers();
  }

  setupDownloadButtons() {
    const downloadButtons = document.querySelectorAll(".download-btn");

    downloadButtons.forEach((btn, index) => {
      btn.addEventListener("click", (e) => {
        e.stopbigation();
        if (this.augmentationResults && this.augmentationResults[index]) {
          this.downloadImage(this.augmentationResults[index], btn.dataset.type);
        }
      });
    });
  }

  setupImageClickHandlers() {
    const imageContainers = document.querySelectorAll(
      ".augmentation-image-container",
    );

    imageContainers.forEach((container) => {
      container.addEventListener("click", () => {
        const img = container.querySelector("img");
        if (img.src && !img.src.endsWith("")) {
          this.showImageModal(img.src, img.alt);
        }
      });
    });
  }

  downloadImage(imageUrl, type) {
    if (!imageUrl) return;

    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = `augmented_${type}_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  async downloadAll() {
    if (!this.augmentationResults) return;

    const types = [
      "original",
      "rotation",
      "width-shift",
      "height-shift",
      "shear",
      "zoom",
      "brightness",
      "channel-shift",
    ];

    // Download each image with a small delay
    for (let i = 0; i < this.augmentationResults.length; i++) {
      if (this.augmentationResults[i]) {
        setTimeout(() => {
          this.downloadImage(this.augmentationResults[i], types[i]);
        }, i * 200);
      }
    }
  }

  createImageModal() {
    const modal = document.createElement("div");
    modal.className = "image-modal";
    modal.id = "imageModal";

    modal.innerHTML = `
            <div class="image-modal-content">
                <button class="image-modal-close" id="imageModalClose">×</button>
                <img id="imageModalImg" src="" alt="">
            </div>
        `;

    document.body.appendChild(modal);

    this.imageModal = modal;
    this.imageModalImg = document.getElementById("imageModalImg");
    this.imageModalClose = document.getElementById("imageModalClose");

    this.imageModalClose.addEventListener(
      "click",
      this.hideImageModal.bind(this),
    );
    this.imageModal.addEventListener("click", (e) => {
      if (e.target === this.imageModal) {
        this.hideImageModal();
      }
    });
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

  async ensureGradioConnection() {
    if (!this.gradioClient) {
      console.log("No client found, initializing...");
      await this.initializeGradioClient();
    } else {
      console.log("Client already connected");
    }
  }

  showImageModal(src, alt) {
    this.imageModalImg.src = src;
    this.imageModalImg.alt = alt;
    this.imageModal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  hideImageModal() {
    this.imageModal.classList.remove("active");
    document.body.style.overflow = "";
  }

  showLoading() {
    this.loadingState.style.display = "flex";
    this.resultsContent.style.display = "none";
    this.emptyState.style.display = "none";
    this.augmentBtn.disabled = true;
  }

  hideLoading() {
    this.loadingState.style.display = "none";
    this.augmentBtn.disabled = false;
  }

  showResults() {
    this.resultsContent.style.display = "block";
    this.emptyState.style.display = "none";
  }

  hideResults() {
    this.resultsContent.style.display = "none";
    this.emptyState.style.display = "block";

    // Clear all images
    const images = this.augmentationGrid.querySelectorAll("img");
    images.forEach((img) => {
      img.src = "";
    });

    // Reset all items
    const items = this.augmentationGrid.querySelectorAll(".augmentation-item");
    items.forEach((item) => {
      item.classList.remove("loaded", "error");
    });
  }

  updateStatus(type, text) {
    this.statusText.textContent = text;
    this.statusIndicator.className = `status-indicator ${type}`;
  }

  showError(message) {
    // Create a temporary error notification
    const errorDiv = document.createElement("div");
    errorDiv.className = "error-notification";
    errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: var(--accent-error);
            color: var(--text-inverse);
            padding: var(--space-4) var(--space-6);
            border-radius: var(--radius-lg);
            z-index: 1000;
            max-width: 300px;
            animation: slideInRight 0.3s ease-out;
        `;
    errorDiv.textContent = message;

    document.body.appendChild(errorDiv);

    setTimeout(() => {
      errorDiv.style.animation = "slideOutRight 0.3s ease-in forwards";
      setTimeout(() => document.body.removeChild(errorDiv), 300);
    }, 5000);
  }

  showModal(type) {
    let title = "";
    let content = "";

    if (type === "about") {
      title = "About Image Augmentation";
      content = `
                <div style="line-height: 1.6;">
                    <p style="margin-bottom: 16px;">
                        This tool generates 8 different types of image augmentations using advanced computer vision techniques.
                    </p>
                    <h4 style="margin: 20px 0 12px 0; color: var(--accent-primary);">Augmentation Types:</h4>
                    <ul style="margin-left: 20px;">
                        <li><strong>Random Rotation:</strong> Rotates images by random angles</li>
                        <li><strong>Width Shift:</strong> Horizontally shifts image content</li>
                        <li><strong>Height Shift:</strong> Vertically shifts image content</li>
                        <li><strong>Shear:</strong> Applies geometric shearing transformations</li>
                        <li><strong>Zoom:</strong> Zooms in/out on image content</li>
                        <li><strong>Brightness:</strong> Adjusts image brightness levels</li>
                        <li><strong>Channel Shift:</strong> Shifts color channel intensities</li>
                    </ul>
                    <p style="margin-top: 20px; font-size: 14px; color: var(--text-secondary);">
                        Powered by Gradio API • Real-time processing
                    </p>
                </div>
            `;
    } else if (type === "help") {
      title = "How to Use";
      content = `
                <div style="line-height: 1.6;">
                    <h4 style="margin: 0 0 12px 0; color: var(--accent-primary);">Getting Started:</h4>
                    <ol style="margin-left: 20px;">
                        <li style="margin-bottom: 8px;">Upload an image by clicking the upload area or dragging & dropping</li>
                        <li style="margin-bottom: 8px;">Click "Generate Augmentations" to process your image</li>
                        <li style="margin-bottom: 8px;">View the 8 different augmented versions in the grid</li>
                        <li style="margin-bottom: 8px;">Click on any image to view it in fullscreen</li>
                        <li style="margin-bottom: 8px;">Download individual images or all at once</li>
                    </ol>

                    <h4 style="margin: 20px 0 12px 0; color: var(--accent-primary);">Tips:</h4>
                    <ul style="margin-left: 20px;">
                        <li>Supported formats: PNG, JPG, JPEG</li>
                        <li>Best results with clear, well-lit images</li>
                        <li>Use keyboard shortcuts: Esc to close modals</li>
                        <li>Processing typically takes 10-30 seconds</li>
                    </ul>
                </div>
            `;
    }

    this.modalTitle.textContent = title;
    this.modalBody.innerHTML = content;
    this.modal.style.display = "flex";
    document.body.style.overflow = "hidden";
  }

  hideModal() {
    this.modal.style.display = "none";
    document.body.style.overflow = "";
  }

  handleKeydown(e) {
    // Escape key closes modals
    if (e.key === "Escape") {
      if (this.modal.style.display === "flex") {
        this.hideModal();
      }
      if (this.imageModal.classList.contains("active")) {
        this.hideImageModal();
      }
    }

    // Enter key triggers augmentation if image is selected
    if (e.key === "Enter" && this.currentImage && !this.augmentBtn.disabled) {
      this.generateAugmentations();
    }
  }
}

// Add required CSS animations
const style = document.createElement("style");
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the app when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new ImageAugmentationApp();
});
