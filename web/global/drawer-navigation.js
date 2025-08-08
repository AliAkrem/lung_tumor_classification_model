// Drawer Navigation Component - Zed Design System Implementation

class DrawerNavigation {
  constructor() {
    this.isOpen = false;
    this.currentPath = window.location.pathname;

    this.init();
  }

  init() {
    this.createDrawerHTML();
    this.bindEvents();
    this.setActiveState();
    this.handleKeyboardNavigation();
  }

  getCorrectPath(path) {
    // Determine if we're in a subdirectory
    const currentPath = window.location.pathname;
    const isInAugDir =
      currentPath.includes("/aug/") || window.location.href.includes("/aug/");

    // Check which directory we're in
    if (currentPath.includes("/augmentation/")) {
      // We're in augmentation directory
      if (path === "classification/index.html") {
        return "../classification/index.html";
      } else if (path === "augmentation/index.html") {
        return "index.html";
      }
    } else if (currentPath.includes("/classification/")) {
      // We're in classification directory
      if (path === "classification/index.html") {
        return "index.html";
      } else if (path === "augmentation/index.html") {
        return "../augmentation/index.html";
      }
    } else {
      // We're in root directory
      return path;
    }
    return path;
  }

  createDrawerHTML() {
    // Find header and add toggle button to it
    const headerContent = document.querySelector(".header-content .logo");
    if (headerContent) {
      const toggleButton = document.createElement("button");
      toggleButton.className = "drawer-toggle";
      toggleButton.setAttribute("aria-label", "Open navigation menu");
      toggleButton.setAttribute("aria-expanded", "false");
      toggleButton.innerHTML = '<span class="drawer-toggle-icon">☰</span>';

      // Insert at the beginning of logo section
      headerContent.insertBefore(toggleButton, headerContent.firstChild);
      this.toggleButton = toggleButton;
    }

    // Create drawer (no overlay needed for push behavior)
    const drawer = document.createElement("div");
    drawer.className = "drawer";
    drawer.setAttribute("role", "navigation");
    drawer.setAttribute("aria-label", "Main navigation");

    drawer.innerHTML = `
      <div class="drawer-header">
        <h2 class="drawer-title">Navigation</h2>
        <button class="drawer-close" aria-label="Close navigation">×</button>
      </div>

      <div class="drawer-content">
        <div class="drawer-nav-section">
          <h3 class="drawer-nav-title">AI Tools</h3>
          <ul class="drawer-nav-list">
            <li class="drawer-nav-item">
              <a href="#" class="drawer-nav-link" data-page="classification" data-target="classification/index.html">
                <div class="drawer-nav-icon">🫁</div>
                <div class="drawer-nav-content">
                  <div class="drawer-nav-label">Tumor Classification</div>
                  <div class="drawer-nav-description">Analyze CT scans for lung tumors</div>
                </div>
                <div class="drawer-nav-status">Active</div>
              </a>
            </li>
            <li class="drawer-nav-item">
              <a href="#" class="drawer-nav-link" data-page="augmentation" data-target="augmentation/index.html">
                <div class="drawer-nav-icon">🔄</div>
                <div class="drawer-nav-content">
                  <div class="drawer-nav-label">Image Augmentation</div>
                  <div class="drawer-nav-description">Generate multiple image variations</div>
                </div>
                <div class="drawer-nav-status">Active</div>
              </a>
            </li>
          </ul>
        </div>

        <div class="drawer-nav-section">
          <h3 class="drawer-nav-title">Resources</h3>
          <ul class="drawer-nav-list">
            <li class="drawer-nav-item">
              <a href="#" class="drawer-nav-link" id="drawer-about">
                <div class="drawer-nav-icon">ℹ️</div>
                <div class="drawer-nav-content">
                  <div class="drawer-nav-label">About</div>
                  <div class="drawer-nav-description">Learn about this application</div>
                </div>
              </a>
            </li>
            <li class="drawer-nav-item">
              <a href="#" class="drawer-nav-link" id="drawer-help">
                <div class="drawer-nav-icon">❓</div>
                <div class="drawer-nav-content">
                  <div class="drawer-nav-label">Help</div>
                  <div class="drawer-nav-description">Get help and documentation</div>
                </div>
              </a>
            </li>
          </ul>
        </div>
      </div>

      <div class="drawer-footer">
        <div class="drawer-footer-content">
          Powered by EfficientNet & Gradio<br>
          <strong>AI Diagnostics Platform</strong>
        </div>
      </div>
    `;

    // Add drawer to page
    document.body.appendChild(drawer);

    // Store references
    this.drawer = drawer;
    this.closeButton = drawer.querySelector(".drawer-close");
    this.navLinks = drawer.querySelectorAll(".drawer-nav-link");

    // Fix navigation links after HTML creation
    this.fixNavigationLinks();
  }

  fixNavigationLinks() {
    this.navLinks.forEach((link) => {
      const target = link.getAttribute("data-target");
      if (target) {
        const correctPath = this.getCorrectPath(target);
        link.setAttribute("href", correctPath);
      }
    });
  }

  bindEvents() {
    // Toggle button events
    this.toggleButton.addEventListener("click", () => {
      this.toggle();
    });

    // Close button events
    this.closeButton.addEventListener("click", () => {
      this.close();
    });

    // Navigation link events
    this.navLinks.forEach((link) => {
      link.addEventListener("click", (e) => {
        // Handle internal links
        if (link.id === "drawer-about" || link.id === "drawer-help") {
          e.preventDefault();
          this.handleModalLink(link.id);
          this.close();
        } else {
          // Add loading state for page navigation
          this.addLoadingState(link);
          // Let the browser handle the navigation
          setTimeout(() => this.close(), 100);
        }
      });
    });

    // Escape key to close
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.isOpen) {
        this.close();
      }
    });

    // Handle browser back/forward
    window.addEventListener("popstate", () => {
      this.currentPath = window.location.pathname;
      this.setActiveState();
    });
  }

  toggle() {
    if (this.isOpen) {
      this.close();
    } else {
      this.open();
    }
  }

  open() {
    this.isOpen = true;
    this.toggleButton.classList.add("active");
    this.toggleButton.classList.add("pressed");
    this.drawer.classList.add("active");
    document.body.classList.add("drawer-open");

    // Update ARIA attributes
    this.toggleButton.setAttribute("aria-expanded", "true");

    // Focus first navigation item
    const firstNavLink = this.drawer.querySelector(".drawer-nav-link");
    if (firstNavLink) {
      setTimeout(() => firstNavLink.focus(), 300);
    }

    // Remove pressed animation class
    setTimeout(() => {
      this.toggleButton.classList.remove("pressed");
    }, 200);

    // Add first-open class for animation
    if (!this.drawer.classList.contains("first-opened")) {
      this.drawer.classList.add("first-open");
      this.drawer.classList.add("first-opened");
      setTimeout(() => {
        this.drawer.classList.remove("first-open");
      }, 300);
    }
  }

  close() {
    this.isOpen = false;
    this.toggleButton.classList.remove("active");
    this.drawer.classList.remove("active");
    document.body.classList.remove("drawer-open");

    // Update ARIA attributes
    this.toggleButton.setAttribute("aria-expanded", "false");

    // Return focus to toggle button
    this.toggleButton.focus();
  }

  setActiveState() {
    this.navLinks.forEach((link) => {
      link.classList.remove("active");

      const target = link.getAttribute("data-target");
      const page = link.getAttribute("data-page");

      if (target || page) {
        const currentPath = this.normalizePath(this.currentPath);

        if (this.isActiveForPage(currentPath, page, target)) {
          link.classList.add("active");
        }
      }
    });
  }

  normalizePath(path) {
    if (!path) return "";

    // Remove leading slash and normalize
    path = path.replace(/^\/+/, "");

    // Handle index.html as root
    if (path === "index.html" || path === "") {
      return "index";
    }

    // Handle nested paths
    if (path.includes("/index.html")) {
      return path.replace("/index.html", "");
    }

    return path.replace(".html", "");
  }

  isActiveForPage(currentPath, page, target) {
    // Check for classification page
    if (page === "classification") {
      return (
        currentPath.includes("/classification/") ||
        currentPath === "" ||
        currentPath === "index" ||
        (currentPath.endsWith("index.html") &&
          !currentPath.includes("/augmentation/")) ||
        (currentPath.endsWith("/") && !currentPath.includes("/augmentation/"))
      );
    }

    // Check for augmentation page
    if (page === "augmentation") {
      return (
        currentPath.includes("/augmentation/") ||
        window.location.href.includes("/augmentation/")
      );
    }

    // Fallback to target-based matching
    if (target) {
      const linkPath = this.normalizePath(target);
      return currentPath === linkPath;
    }

    return false;
  }

  addLoadingState(link) {
    link.classList.add("loading");

    // Remove loading state after timeout
    setTimeout(() => {
      link.classList.remove("loading");
    }, 2000);
  }

  handleModalLink(linkId) {
    // Try to trigger existing modal functionality
    if (linkId === "drawer-about") {
      const aboutBtn = document.getElementById("aboutBtn");
      if (aboutBtn) {
        aboutBtn.click();
      }
    } else if (linkId === "drawer-help") {
      const helpBtn = document.getElementById("helpBtn");
      if (helpBtn) {
        helpBtn.click();
      }
    }
  }

  handleKeyboardNavigation() {
    // Handle tab navigation within drawer
    this.drawer.addEventListener("keydown", (e) => {
      if (!this.isOpen) return;

      const focusableElements = this.drawer.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.key === "Tab") {
        // Trap focus within drawer
        if (e.shiftKey) {
          if (document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
          }
        } else {
          if (document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
          }
        }
      }
    });

    // Arrow key navigation between nav items
    const navLinks = Array.from(this.navLinks);

    this.drawer.addEventListener("keydown", (e) => {
      if (!this.isOpen) return;

      const currentIndex = navLinks.indexOf(document.activeElement);

      if (currentIndex >= 0) {
        let newIndex = currentIndex;

        switch (e.key) {
          case "ArrowDown":
            e.preventDefault();
            newIndex = (currentIndex + 1) % navLinks.length;
            break;
          case "ArrowUp":
            e.preventDefault();
            newIndex = (currentIndex - 1 + navLinks.length) % navLinks.length;
            break;
          case "Home":
            e.preventDefault();
            newIndex = 0;
            break;
          case "End":
            e.preventDefault();
            newIndex = navLinks.length - 1;
            break;
        }

        if (newIndex !== currentIndex) {
          navLinks[newIndex].focus();
        }
      }
    });
  }

  // Public methods
  refresh() {
    this.currentPath = window.location.pathname;
    this.setActiveState();
  }

  getActiveItem() {
    return this.drawer.querySelector(".drawer-nav-link.active");
  }

  destroy() {
    // Clean up event listeners and remove elements
    if (this.toggleButton) {
      this.toggleButton.remove();
    }
    if (this.drawer) {
      this.drawer.remove();
    }

    // Remove drawer-open class
    document.body.classList.remove("drawer-open");
  }
}

// Initialize drawer navigation when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  const drawerNav = new DrawerNavigation();

  // Make drawer navigation globally available
  window.drawerNavigation = drawerNav;

  console.log("Drawer navigation initialized");
});

// Handle page visibility changes
document.addEventListener("visibilitychange", () => {
  if (document.hidden && window.drawerNavigation) {
    window.drawerNavigation.close();
  }
});

// Export for module use if needed
if (typeof module !== "undefined" && module.exports) {
  module.exports = DrawerNavigation;
}
