# Lung Tumor Prediction Web Application

This web application provides tools for lung tumor classification and image augmentation.

## Directory Structure

The web application follows a modular structure:

```
lung_tumor_prediction/web/
├── global/                   # Shared resources
│   ├── styles.css            # Global CSS styles
│   ├── drawer-navigation.css # Navigation drawer styles
│   └── drawer-navigation.js  # Navigation drawer functionality
│
├── classification/           # Tumor classification tool
│   ├── index.html            # Classification page
│   └── script.js             # Classification functionality
│
├── augmentation/             # Image augmentation tool
│   ├── index.html            # Augmentation page
│   ├── augmentation-script.js # Augmentation functionality
│   └── augmentation-styles.css # Augmentation-specific styles
│
├── segmentation/             # Lung segmentation tool
│   ├── index.html            # Segmentation page
│   ├── segmentation-script.js # Segmentation functionality
│   └── segmentation-styles.css # Segmentation-specific styles
│
├── utils/                    # Utility files and tools
│   └── test-api.html         # API testing utility
│
├── index.html                # Root redirect page
├── README.md                 # Documentation
└── .htaccess                 # Server configuration (Apache)
```

## Pages

1. **Tumor Classification** (`/classification/index.html`)
   - Upload CT scan images for tumor classification
   - View detailed classification results with confidence scores
   - Interactive chart visualization

2. **Image Augmentation** (`/augmentation/index.html`)
   - Generate 8 different augmented versions of an image
   - View all augmentations in a grid layout
   - Download individual or all augmented images

3. **Lung Segmentation** (`/segmentation/index.html`)
   - Upload CT scan images for lung region segmentation
   - View segmented regions (left lung, right lung, mediastinum, dark regions)
   - Download individual segments or all at once

## Navigation

The application uses a slide-out drawer navigation accessible via the hamburger menu (☰) in the header. The drawer pushes the content when opened and provides access to both tools.

## Technologies

- **Frontend**: Pure HTML, CSS, and JavaScript (no frameworks)
- **API Integration**: Gradio client for AI model communication
- **Design System**: Custom Zed-inspired design system
- **Visualization**: Chart.js for data visualization

## Development

To run the application locally:

1. Clone the repository
2. Navigate to the `web` directory
3. Serve with any static file server (e.g., `python -m http.server`)

## File Descriptions

- **global/styles.css**: Main styling following the Zed design system
- **global/drawer-navigation.css**: Styles for the drawer navigation component
- **global/drawer-navigation.js**: JavaScript for drawer functionality and navigation
- **classification/script.js**: Handles tumor classification logic and API integration
- **augmentation/augmentation-script.js**: Handles image augmentation logic and API integration
- **augmentation/augmentation-styles.css**: Specific styles for the augmentation grid
- **segmentation/segmentation-script.js**: Handles lung segmentation logic and API integration
- **segmentation/segmentation-styles.css**: Specific styles for the segmentation grid
- **utils/test-api.html**: Utility for testing API connections and responses