// Lung Segmentation - Main JavaScript Application

class LungSegmenter {
  constructor() {
    this.spaceUrl = "aliakrem/seg";
    this.gradioClient = null;
    this.currentImage = null;
    this.segmentationResults = null;
    this.debug = true; // Enable debugging
    this.apiTimeout = 30000; // 30 seconds timeout for API calls

    // DOM Elements
    this.initializeElements();
    this.bindEvents();
    this.setupDragAndDrop();

    // Initialize Gradio client
    this.initializeGradioClient();

    // Log initialization
    if (this.debug) console.log("LungSegmenter initialized");
  }

  initializeElements() {
    // Upload elements
    this.uploadArea = document.getElementById("uploadArea");
    this.imageInput = document.getElementById("imageInput");
    this.imagePreview = document.getElementById("imagePreview");
    this.previewImg = document.getElementById("previewImg");
    this.imageInfo = document.getElementById("imageInfo");
    this.clearImage = document.getElementById("clearImage");
    this.segmentBtn = document.getElementById("segmentBtn");

    // Make sure image input is properly initialized
    if (!this.imageInput) {
      console.error("Image input element not found!");
      // Create one if not found
      this.imageInput = document.createElement("input");
      this.imageInput.type = "file";
      this.imageInput.id = "imageInput";
      this.imageInput.accept = "image/*";
      this.imageInput.className = "file-input";
      if (this.uploadArea) {
        this.uploadArea.appendChild(this.imageInput);
      }
    }

    // Also get the browse button
    this.browseButton = document.getElementById("browseButton");

    // Results elements
    this.loadingState = document.getElementById("loadingState");
    this.resultsContent = document.getElementById("resultsContent");
    this.emptyState = document.getElementById("emptyState");
    this.statusIndicator = document.getElementById("statusIndicator");
    this.statusText = this.statusIndicator.querySelector(".status-text");

    // Segmentation images
    this.originalImage = document.getElementById("originalImage");
    this.combinedImage = document.getElementById("combinedImage");
    this.leftLungImage = document.getElementById("leftLungImage");
    this.darkRegionsImage = document.getElementById("darkRegionsImage");
    this.mediastinumImage = document.getElementById("mediastinumImage");
    this.rightLungImage = document.getElementById("rightLungImage");

    // Download button
    this.downloadAllBtn = document.getElementById("downloadAllBtn");

    // Modal elements
    this.modal = document.getElementById("modal");
    this.modalBackdrop = document.getElementById("modalBackdrop");
    this.modalTitle = document.getElementById("modalTitle");
    this.modalBody = document.getElementById("modalBody");
    this.modalClose = document.getElementById("modalClose");
    this.aboutBtn = document.getElementById("aboutBtn");
    this.helpBtn = document.getElementById("helpBtn");

    // Image modal
    this.imageModal = document.getElementById("imageModal");
    this.imageModalImg = document.getElementById("imageModalImg");
    this.imageModalClose = document.getElementById("imageModalClose");
  }

  bindEvents() {
    // Button Events
    this.clearImage.addEventListener("click", () => this.clearImageSelection());
    this.segmentBtn.addEventListener("click", () => this.segmentImage());
    this.downloadAllBtn.addEventListener("click", () => this.downloadAll());

    // Add click handler for the browse button
    if (this.browseButton) {
      this.browseButton.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.imageInput) {
          this.imageInput.click();
        }
      });
    }

    // Modal events
    this.aboutBtn.addEventListener("click", () => this.showAboutModal());
    this.helpBtn.addEventListener("click", () => this.showHelpModal());
    this.modalClose.addEventListener("click", () => this.hideModal());
    this.modalBackdrop.addEventListener("click", () => this.hideModal());

    // Image modal events
    this.imageModalClose.addEventListener("click", () => this.hideImageModal());
    this.imageModal.addEventListener("click", (e) => {
      if (e.target === this.imageModal) {
        this.hideImageModal();
      }
    });

    // Setup image click events for fullscreen view
    this.setupImageClickHandlers();

    // Setup download buttons
    this.setupDownloadButtons();
  }

  setupImageClickHandlers() {
    const imageContainers = document.querySelectorAll(
      ".segment-image-container",
    );
    imageContainers.forEach((container) => {
      container.addEventListener("click", () => {
        const img = container.querySelector("img");
        if (img && img.src && !img.src.endsWith("")) {
          this.showImageModal(img.src, img.alt);
        }
      });
    });
  }

  setupDownloadButtons() {
    const downloadButtons = document.querySelectorAll(".download-btn");
    downloadButtons.forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation(); // Prevent opening the fullscreen view
        const type = btn.dataset.type;
        const img = this.getImageByType(type);
        if (img && img.src && !img.src.endsWith("")) {
          this.downloadImage(img.src, type);
        }
      });
    });
  }

  getImageByType(type) {
    switch (type) {
      case "original":
        return this.originalImage;
      case "combined":
        return this.combinedImage;
      case "left-lung":
        return this.leftLungImage;
      case "dark-regions":
        return this.darkRegionsImage;
      case "mediastinum":
        return this.mediastinumImage;
      case "right-lung":
        return this.rightLungImage;
      default:
        return null;
    }
  }

  setupDragAndDrop() {
    const preventDefaults = (e) => {
      e.preventDefault();
      e.stopPropagation();
      return false;
    };

    // Prevent default drag behaviors
    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
      this.uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ["dragenter", "dragover"].forEach((eventName) => {
      this.uploadArea.addEventListener(
        eventName,
        () => {
          this.uploadArea.classList.add("drag-over");
        },
        false,
      );
    });

    // Remove highlight when item is dragged away or dropped
    ["dragleave", "drop"].forEach((eventName) => {
      this.uploadArea.addEventListener(
        eventName,
        () => {
          this.uploadArea.classList.remove("drag-over");
        },
        false,
      );
    });

    // Handle dropped files
    this.uploadArea.addEventListener("drop", this.handleDrop.bind(this), false);

    // Handle click on the upload area (not on child elements)
    this.uploadArea.addEventListener("click", (e) => {
      // Only trigger if it's directly on the upload area
      if (e.target === this.uploadArea) {
        e.preventDefault();
        if (this.imageInput) {
          this.imageInput.click();
        }
      }
    });

    // Add keyboard accessibility for the upload area
    this.uploadArea.addEventListener("keydown", (e) => {
      // Trigger on Enter or Space key
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        if (this.imageInput) {
          this.imageInput.click();
        }
      }
    });

    // Handle file selection from input
    this.imageInput.addEventListener(
      "change",
      this.handleImageSelect.bind(this),
      false,
    );
  }

  preventDefaults(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }
    return false;
  }

  // Method to manually trigger file input click
  triggerFileInput() {
    if (this.imageInput) {
      this.imageInput.click();
    }
  }

  handleDrop(e) {
    if (!e || !e.dataTransfer) return;

    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length) {
      this.processFile(files[0]);
    }
  }

  handleImageSelect(e) {
    if (e && e.target && e.target.files && e.target.files.length) {
      this.processFile(e.target.files[0]);
    }
  }

  processFile(file) {
    if (!file.type.match("image.*")) {
      this.showError(
        "Please select a valid image file",
        false,
        "Supported formats: PNG, JPG, JPEG",
      );
      return;
    }

    this.currentImage = file;
    this.displayImagePreview(file);
    this.segmentBtn.disabled = false;
    this.updateStatus("Ready to segment", "ready");
  }

  displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      this.previewImg.src = e.target.result;
      this.imagePreview.style.display = "block";

      // Show file info
      const sizeKB = Math.round(file.size / 1024);
      this.imageInfo.textContent = `${file.name} • ${sizeKB} KB`;
    };
    reader.readAsDataURL(file);
  }

  clearImageSelection() {
    this.currentImage = null;
    this.imageInput.value = "";
    this.imagePreview.style.display = "none";
    this.segmentBtn.disabled = true;
    this.hideResults();
    this.hideErrorContainer();
    this.updateStatus("Ready", "ready");
  }

  async segmentImage() {
    if (!this.currentImage) {
      this.showError(
        "No image selected.",
        false,
        "Please upload an image before attempting segmentation.",
      );
      return;
    }

    this.showLoadingState();
    this.updateStatus("Processing...", "processing");

    try {
      // Ensure Gradio client is connected
      await this.ensureGradioConnection();

      if (this.debug)
        console.log("Image info:", {
          name: this.currentImage.name,
          type: this.currentImage.type,
          size: `${Math.round(this.currentImage.size / 1024)} KB`,
        });

      // Convert file to blob for Gradio client
      const imageBlob = await this.fileToBlob(this.currentImage);

      if (this.debug)
        console.log("Calling Gradio API endpoint '/predict' with image...");

      // Try to use the endpoint specified in the space docs
      try {
        // Call the segmentation API with timeout
        const apiPromise = this.gradioClient.predict("/predict", {
          input_image: imageBlob,
        });

        // Set a timeout for the API call
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("API request timed out after 30 seconds")),
            30000,
          ),
        );

        const result = await Promise.race([apiPromise, timeoutPromise]);

        if (this.debug)
          console.log("Gradio API raw response:", JSON.stringify(result));

        if (result && result.data) {
          // Handle both array format and object format responses
          if (Array.isArray(result.data)) {
            if (this.debug)
              console.log(
                "Processing array response with",
                result.data.length,
                "items",
              );
            this.segmentationResults = result.data;
            this.displaySegmentationResults(result.data);
            this.updateStatus("Segmentation complete", "success");
          } else if (typeof result.data === "object") {
            // If it's an object with numeric keys (like {0: url1, 1: url2})
            const dataArray = Object.values(result.data);
            if (this.debug)
              console.log(
                "Processing object response converted to array with",
                dataArray.length,
                "items",
              );

            if (dataArray.length > 0) {
              this.segmentationResults = dataArray;
              this.displaySegmentationResults(dataArray);
              this.updateStatus("Segmentation complete", "success");
            } else {
              throw new Error("Empty data object in API response");
            }
          } else {
            if (this.debug)
              console.log("Unexpected data format:", typeof result.data);
            throw new Error("Unexpected data format in API response");
          }
        } else {
          if (this.debug) console.log("Invalid response structure:", result);
          throw new Error("No valid response from Gradio API");
        }
      } catch (endpointError) {
        // Try alternative endpoint as fallback
        if (this.debug)
          console.log(
            "Primary endpoint failed, trying fallback endpoint",
            endpointError,
          );

        const alternativeResult = await this.gradioClient.predict(0, [
          imageBlob,
        ]);

        if (this.debug)
          console.log("Alternative endpoint response:", alternativeResult);

        if (alternativeResult && alternativeResult.length > 0) {
          this.segmentationResults = alternativeResult;
          this.displaySegmentationResults(alternativeResult);
          this.updateStatus("Segmentation complete", "success");
        } else {
          throw new Error("Fallback API call failed: " + endpointError.message);
        }
      }
    } catch (error) {
      console.error("Segmentation error:", error);
      this.showError(
        "Segmentation failed",
        true,
        `Error: ${error.message}. Please try a different image or try again later.`,
      );
      this.updateStatus("Error", "error");

      if (this.debug) {
        // Display more detailed error information in the UI for debugging
        this.showErrorContainer();
        const errorDetails = document.createElement("div");
        errorDetails.className = "error-details";
        errorDetails.innerHTML = `
          <h4>Technical Error Details:</h4>
          <pre>${error.stack || error.toString()}</pre>
          <p>Please report this error along with the image you tried to upload.</p>
        `;
        const errorContent = document.querySelector(
          ".error-container .error-content",
        );
        if (errorContent) errorContent.appendChild(errorDetails);
      }
    }
  }

  async fileToBlob(file) {
    return new Promise((resolve, reject) => {
      try {
        const reader = new FileReader();
        reader.onload = () => {
          try {
            const arrayBuffer = reader.result;
            const blob = new Blob([arrayBuffer], { type: file.type });
            console.log("File converted to blob successfully", {
              type: file.type,
              size: blob.size,
            });
            resolve(blob);
          } catch (err) {
            console.error("Error creating blob:", err);
            reject(err);
          }
        };
        reader.onerror = (err) => {
          console.error("FileReader error:", err);
          reject(err);
        };
        reader.readAsArrayBuffer(file);
      } catch (err) {
        console.error("Unexpected error in fileToBlob:", err);
        reject(err);
      }
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

      if (this.debug) console.log(`GradioClient loading attempts: ${attempts}`);

      if (!window.GradioClient) {
        throw new Error("Gradio client not loaded after multiple attempts");
      }

      if (this.debug)
        console.log(
          "Initializing Gradio client with space URL:",
          this.spaceUrl,
        );

      // Create connection with more options for debugging
      this.gradioClient = await window.GradioClient.connect(this.spaceUrl, {
        hf_token: null, // No token needed for public spaces
        status_callback: (status) => {
          if (this.debug) console.log("Gradio connection status:", status);
          // Update UI based on connection status
          if (status === "connected") {
            this.updateStatus("Connected to API", "ready");
          } else if (status === "connecting") {
            this.updateStatus("Connecting to API...", "processing");
          } else if (status === "error") {
            this.updateStatus("Connection error", "error");
          }
        },
        max_retries: 5,
        verbose: this.debug,
      });

      // Log available endpoints and API info
      if (this.gradioClient) {
        if (this.debug) {
          console.log("Gradio client initialized:", {
            endpoints: this.gradioClient.endpoints || "No endpoints available",
            config: this.gradioClient.config || "No config available",
            session_hash: this.gradioClient.session_hash || "No session hash",
          });
        }

        // Check if we have the required endpoint
        const hasRequiredEndpoint = this.gradioClient.endpoints?.some(
          (endpoint) => endpoint.includes("/predict") || endpoint === 0,
        );

        if (!hasRequiredEndpoint && this.debug) {
          console.warn(
            "Required endpoint '/predict' not found in available endpoints!",
          );
        }
      }

      if (this.debug) console.log("Gradio client connected successfully");
      this.updateStatus("Ready", "ready");
    } catch (error) {
      console.error("Failed to initialize Gradio client:", error);
      this.updateStatus("API connection failed", "error");

      // Show more detailed error message
      const errorMessage = error.message || "Unknown error";
      const errorDetails = this.debug
        ? `<br><small>${errorMessage}</small>`
        : "";

      this.showError(
        "Failed to connect to the AI service. Please refresh the page and try again.",
        true,
        errorDetails ? `Technical details: ${errorDetails}` : null,
      );

      // Try reconnecting after a delay
      setTimeout(() => {
        if (this.debug) console.log("Attempting to reconnect to Gradio...");
        this.initializeGradioClient();
      }, 5000);
    }
  }

  async ensureGradioConnection() {
    if (!this.gradioClient) {
      await this.initializeGradioClient();
    }
  }

  displaySegmentationResults(results) {
    this.hideLoadingState();
    this.showResults();

    if (this.debug) {
      console.log("Processing segmentation results:", results);
      console.log(
        "Results type:",
        Array.isArray(results) ? "Array" : typeof results,
      );
      console.log(
        "Results length:",
        Array.isArray(results) ? results.length : "N/A",
      );
    }

    // Map results to image elements
    // Relax the constraint to handle partial results
    if (results && (Array.isArray(results) ? results.length > 0 : true)) {
      const imageElements = [
        this.originalImage,
        this.combinedImage,
        this.leftLungImage,
        this.darkRegionsImage,
        this.mediastinumImage,
        this.rightLungImage,
      ];

      // Convert results to array if it's not already
      let resultsArray = [];

      // Handle various response formats
      if (Array.isArray(results)) {
        resultsArray = results;
      } else if (results && typeof results === "object") {
        if (Object.keys(results).length > 0) {
          // If it's an object with numeric keys, convert to array
          resultsArray = Object.values(results);
        } else {
          resultsArray = [results];
        }
      } else if (results) {
        resultsArray = [results];
      }

      if (this.debug) console.log("Processing results array:", resultsArray);

      // Process available results (handle cases with fewer than 6 images)
      const availableResults = Math.min(
        resultsArray.length,
        imageElements.length,
      );

      if (this.debug)
        console.log(
          `Processing ${availableResults} out of ${imageElements.length} possible segments`,
        );

      let successfulLoads = 0;
      let failedLoads = 0;
      const totalImages = availableResults;

      // Create a summary element to show loading progress
      const progressSummary = document.createElement("div");
      progressSummary.className = "loading-progress";
      progressSummary.innerHTML = `<span>Loading segments: 0/${totalImages}</span>`;
      this.resultsContent.prepend(progressSummary);

      for (let index = 0; index < availableResults; index++) {
        const img = imageElements[index];
        const resultItem = resultsArray[index];

        if (!img) {
          if (this.debug)
            console.error(`Image element at index ${index} not found in DOM`);
          continue;
        }

        if (resultItem) {
          const container = img.closest(".segment-image-container");
          const item = img.closest(".segment-item");

          if (!container || !item) {
            if (this.debug)
              console.error(
                `Container or item elements not found for image at index ${index}`,
              );
            continue;
          }

          // Add loading state
          container.classList.add("loading");

          // Handle different data formats
          let imageUrl = resultItem;
          let imageType = "string";

          // If the result is an object with a 'url' or 'data' property
          if (typeof resultItem === "object") {
            if (resultItem.url || resultItem.data) {
              imageUrl = resultItem.url || resultItem.data;
              imageType = "object with url/data";
            } else if (resultItem.image) {
              imageUrl = resultItem.image;
              imageType = "object with image";
            } else if (resultItem.blob) {
              imageUrl = URL.createObjectURL(resultItem.blob);
              imageType = "object with blob";
            }
          }

          if (this.debug)
            console.log(
              `Processing segment ${index}: ${imageType}`,
              imageUrl.substring ? imageUrl.substring(0, 50) + "..." : imageUrl,
            );

          // Set image source
          img.src = imageUrl;

          // Handle image load event
          img.onload = () => {
            successfulLoads++;
            container.classList.remove("loading");
            item.classList.add("loaded");

            // Update progress summary
            progressSummary.innerHTML = `<span>Loading segments: ${successfulLoads}/${totalImages}</span>`;

            if (this.debug)
              console.log(
                `Segment ${index} loaded successfully (${successfulLoads}/${totalImages})`,
              );

            // Update success status when all images are loaded
            if (successfulLoads === totalImages) {
              this.updateStatus("All segments loaded successfully", "success");
              // Remove progress summary when all loaded successfully
              progressSummary.remove();
            } else if (successfulLoads + failedLoads === totalImages) {
              // If all images have attempted to load (some success, some failure)
              this.updateStatus(
                `${successfulLoads}/${totalImages} segments loaded`,
                "warning",
              );
            }
          };

          // Handle image load error
          img.onerror = (e) => {
            container.classList.remove("loading");
            item.classList.add("error");
            failedLoads++;

            // Update progress summary
            progressSummary.innerHTML = `<span>Loading segments: ${successfulLoads}/${totalImages} (${failedLoads} failed)</span>`;

            if (this.debug) {
              console.error(`Failed to load segment image ${index}`, {
                url: imageUrl.substring
                  ? imageUrl.substring(0, 100) + "..."
                  : imageUrl,
                error: e,
              });

              // Add error indicator to the container
              const errorMsg = document.createElement("div");
              errorMsg.className = "segment-error-message";
              errorMsg.textContent = "Failed to load image";
              container.appendChild(errorMsg);
            }

            // If all images have attempted to load (success or failure)
            if (successfulLoads + failedLoads === totalImages) {
              if (failedLoads === totalImages) {
                this.updateStatus("All segments failed to load", "error");
              } else {
                this.updateStatus(
                  `${successfulLoads}/${totalImages} segments loaded`,
                  "warning",
                );
              }
            }
          };
        } else {
          if (this.debug) console.log(`No result item for index ${index}`);
        }
      }

      this.updateStatus("Segmentation complete", "success");
    } else {
      const errorMsg = "No segmentation results received from the API";
      if (this.debug) console.error(errorMsg, results);
      this.showError(
        errorMsg,
        true,
        "The API didn't return any valid image segments. Please try a different image.",
      );
      this.updateStatus("Error processing results", "error");
      this.showErrorContainer();
    }
  }

  downloadImage(imageUrl, type) {
    if (!imageUrl) return;

    const link = document.createElement("a");
    link.href = imageUrl;
    link.download = `segment-${type}-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  downloadAll() {
    if (!this.segmentationResults) return;

    const types = [
      "original",
      "combined",
      "left-lung",
      "dark-regions",
      "mediastinum",
      "right-lung",
    ];

    // Download each image with a small delay
    types.forEach((type, index) => {
      setTimeout(() => {
        const img = this.getImageByType(type);
        if (img && img.src && !img.src.endsWith("")) {
          this.downloadImage(img.src, type);
        }
      }, index * 300);
    });
  }

  showLoadingState() {
    this.loadingState.style.display = "flex";
    this.resultsContent.style.display = "none";
    this.emptyState.style.display = "none";
    this.segmentBtn.disabled = true;
  }

  hideLoadingState() {
    this.loadingState.style.display = "none";
    this.segmentBtn.disabled = false;
  }

  showResults() {
    this.resultsContent.style.display = "block";
    this.emptyState.style.display = "none";

    // Hide error container when showing successful results
    this.hideErrorContainer();
  }

  hideResults() {
    this.resultsContent.style.display = "none";
    this.emptyState.style.display = "block";

    // Clear all segment images
    const segmentImages = document.querySelectorAll(
      ".segment-image-container img",
    );
    segmentImages.forEach((img) => {
      img.src = "";
    });

    // Reset all segment items
    const segmentItems = document.querySelectorAll(".segment-item");
    segmentItems.forEach((item) => {
      item.classList.remove("loaded", "error");
    });
  }

  updateStatus(text, state) {
    this.statusText.textContent = text;

    // Update status indicator
    this.statusIndicator.className = "status-indicator";
    if (state) {
      this.statusIndicator.classList.add(state);
    }
  }

  showError(message, isPermanent = false, details = null) {
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
      max-width: 400px;
      box-shadow: var(--shadow-lg);
    `;

    // Add icon and error message
    errorDiv.innerHTML = `
      <div style="display: flex; align-items: flex-start;">
        <div style="margin-right: 12px; font-size: 24px;">❌</div>
        <div>
          <div style="font-weight: 600; margin-bottom: 4px;">${message}</div>
          ${details ? `<div style="font-size: 12px; opacity: 0.9;">${details}</div>` : ""}
        </div>
      </div>
    `;

    // Add close button
    const closeButton = document.createElement("button");
    closeButton.innerHTML = "×";
    closeButton.style.cssText = `
      position: absolute;
      top: 6px;
      right: 6px;
      background: none;
      border: none;
      color: white;
      font-size: 18px;
      cursor: pointer;
      opacity: 0.7;
    `;
    closeButton.addEventListener("click", () => {
      removeErrorDiv();
    });
    errorDiv.appendChild(closeButton);

    document.body.appendChild(errorDiv);

    // Also show error in the error container if it exists (for permanent display)
    if (isPermanent && this.debug) {
      this.showErrorContainer();
      const errorContent = document.querySelector(
        ".error-container .error-content",
      );
      if (errorContent) {
        const permanentError = document.createElement("div");
        permanentError.className = "error-message";
        permanentError.innerHTML = `
          <div class="error-title">${message}</div>
          ${details ? `<div class="error-details">${details}</div>` : ""}
        `;
        errorContent.appendChild(permanentError);
      }
    }

    // Function to remove the error div with animation
    const removeErrorDiv = () => {
      errorDiv.style.opacity = "0";
      errorDiv.style.transform = "translateX(100%)";
      errorDiv.style.transition = "all 0.3s ease-out";

      setTimeout(() => {
        if (document.body.contains(errorDiv)) {
          document.body.removeChild(errorDiv);
        }
      }, 300);
    };

    // Auto-remove after timeout unless it's permanent
    if (!isPermanent) {
      setTimeout(() => {
        removeErrorDiv();
      }, 5000);
    }
  }

  showErrorContainer() {
    const errorContainer = document.querySelector(".error-container");
    if (errorContainer) {
      errorContainer.style.display = "block";

      // Clear previous content if it's getting too long
      const errorContent = errorContainer.querySelector(".error-content");
      if (errorContent && errorContent.children.length > 5) {
        // Keep only the 5 most recent errors
        while (errorContent.children.length > 5) {
          errorContent.removeChild(errorContent.firstChild);
        }
      }
    }
  }

  hideErrorContainer() {
    const errorContainer = document.querySelector(".error-container");
    if (errorContainer) {
      errorContainer.style.display = "none";

      // Clear content
      const errorContent = errorContainer.querySelector(".error-content");
      if (errorContent) {
        errorContent.innerHTML = "";
      }
    }
  }

  showImageModal(src, alt) {
    this.imageModalImg.src = src;
    this.imageModalImg.alt = alt || "Segment image";
    this.imageModal.style.display = "flex";
    this.imageModal.classList.add("active");
    document.body.style.overflow = "hidden";
  }

  hideImageModal() {
    this.imageModal.classList.remove("active");
    setTimeout(() => {
      this.imageModal.style.display = "none";
      document.body.style.overflow = "";
    }, 300);
  }

  showAboutModal() {
    this.modalTitle.textContent = "About Lung Segmentation";
    this.modalBody.innerHTML = `
      <div style="line-height: 1.6;">
        <p style="margin-bottom: 16px;">
          The Lung Segmentation tool uses a architecture to segment different areas of lung CT scans.
        </p>

        <h4 style="margin: 20px 0 12px 0; color: var(--accent-primary);">Segment Types</h4>
        <ul style="margin-left: 20px;">
          <li><strong>Left Lung:</strong> Identifies the left lung region</li>
          <li><strong>Right Lung:</strong> Identifies the right lung region</li>
          <li><strong>Mediastinum:</strong> Identifies the central compartment of the thoracic cavity</li>
          <li><strong>Dark Regions:</strong> Identifies abnormally dark regions within the lungs</li>
          <li><strong>Combined:</strong> Shows all segmentations overlaid together</li>
        </ul>

        <h4 style="margin: 20px 0 12px 0; color: var(--accent-primary);">Clinical Applications</h4>
        <p>
          Precise lung segmentation assists radiologists and pulmonologists in:
        </p>
        <ul style="margin-left: 20px;">
          <li>Identifying abnormal tissue</li>
          <li>Measuring lung volumes</li>
          <li>Planning radiation therapy</li>
          <li>Monitoring disease progression</li>
        </ul>

        <p style="margin-top: 20px; font-size: 14px; color: var(--text-secondary);">
           For research purposes only
        </p>
      </div>
    `;
    this.showModal();
  }

  showHelpModal() {
    this.modalTitle.textContent = "How to Use";
    this.modalBody.innerHTML = `
      <div style="line-height: 1.6;">
        <h4 style="margin: 0 0 12px 0; color: var(--accent-primary);">Getting Started</h4>
        <ol style="margin-left: 20px;">
          <li style="margin-bottom: 8px;">Upload a lung CT scan image using the upload area</li>
          <li style="margin-bottom: 8px;">Click "Segment Image" to process your image</li>
          <li style="margin-bottom: 8px;">View the different segmentation results in the grid</li>
          <li style="margin-bottom: 8px;">Click on any segment to view it in fullscreen</li>
          <li style="margin-bottom: 8px;">Download individual segments or all at once</li>
        </ol>

        <h4 style="margin: 20px 0 12px 0; color: var(--accent-primary);">Tips</h4>
        <ul style="margin-left: 20px;">
          <li>For best results, use clear CT scan images with good contrast</li>
          <li>Processing typically takes 10-30 seconds depending on image size</li>
          <li>DICOM files should be converted to PNG or JPG before uploading</li>
          <li>Click any segment to view it in full screen</li>
          <li>Use the download buttons to save individual segments</li>
        </ul>

        <h4 style="margin: 20px 0 12px 0; color: var(--accent-primary);">Understanding Results</h4>
        <ul style="margin-left: 20px;">
          <li><strong>Original Image:</strong> Your uploaded CT scan</li>
          <li><strong>All Segments Combined:</strong> Overlay of all segmentations</li>
          <li><strong>Left/Right Lung:</strong> Individual lung segmentations</li>
          <li><strong>Mediastinum:</strong> Central thoracic compartment</li>
          <li><strong>Dark Regions:</strong> Areas of low density within lungs</li>
        </ul>
      </div>
    `;
    this.showModal();
  }

  showModal() {
    this.modal.style.display = "flex";
  }

  hideModal() {
    this.modal.style.display = "none";
  }
}

// Initialize application when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  window.app = new LungSegmenter();
});
