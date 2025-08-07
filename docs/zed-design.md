# Zed.dev Interface Design Guidelines

## Typography System

### Font Hierarchy

Zed uses a carefully crafted typography system built around the custom Zed Plex font family, designed specifically for optimal code editing and interface clarity.

#### Primary Typefaces
- **UI Font**: `Zed Plex Sans` - Clean, readable sans-serif for interface elements
- **Buffer Font**: `Zed Plex Mono` - Monospaced font optimized for code editing
- **Terminal Font**: `Zed Plex Mono` - Consistent monospace across all code contexts

#### System Font Fallbacks
- **macOS**: `.SystemUIFont` (SF Pro) as fallback for UI elements
- **Cross-platform**: System-appropriate fallbacks maintain consistency

### Font Size Hierarchy

#### Standard Font Sizes
```json
{
  "ui_font_size": 16,          // Interface elements
  "buffer_font_size": 15,      // Code editor
  "terminal_font_size": 15,    // Terminal panel
  "agent_font_size": 15        // AI agent panel
}
```

#### Font Weight Scale
- **Regular (400)**: Default weight for most text
- **Medium (500)**: Emphasis and headers
- **Bold (600-700)**: Strong emphasis and active states
- **Light (300)**: Secondary information and subtle text

### Line Height Standards

Zed provides three standardized line height options optimized for different reading contexts:

- **Comfortable (1.618)**: Golden ratio spacing for reduced eye strain
- **Standard (1.3)**: Compact spacing for information density
- **Custom**: User-defined values for specific preferences

#### Line Height Configuration
```json
{
  "buffer_line_height": "comfortable",
  "terminal": {
    "line_height": "comfortable"
  }
}
```

### Typography Features

#### Programming Ligatures
- **Default Enabled**: Automatic character combination for common programming symbols
- **Examples**: `=>` becomes →, `!=` becomes ≠, `>=` becomes ≥
- **Customizable**: Can be disabled via `"calt": false` in font features

#### Font Feature Settings
```json
{
  "buffer_font_features": {
    "calt": true,    // Enable contextual alternates (ligatures)
    "liga": true,    // Enable standard ligatures
    "kern": true     // Enable kerning
  }
}
```

## Color System

### Theme Architecture

Zed implements a sophisticated color system that adapts to user preferences and environmental conditions while maintaining excellent readability and accessibility.

#### Dual Mode Support
```json
{
  "theme": {
    "dark": "One Dark",
    "light": "One Light",
    "mode": "system"  // Follows OS preference
  }
}
```

### Color Categories

#### Semantic Color Usage
- **Primary Text**: High-contrast colors for main content
- **Secondary Text**: Medium-contrast for supporting information
- **Accent Colors**: Brand and highlight colors used sparingly
- **Status Colors**: Consistent color coding for states and feedback

#### Git Status Colors
- **Added Files**: Green indicators for new content
- **Modified Files**: Orange/yellow for changed content
- **Deleted Files**: Red indicators for removed content
- **Untracked Files**: Gray for uncommitted files

#### Diagnostic Colors
- **Errors**: Red indicators with appropriate contrast ratios
- **Warnings**: Yellow/orange for potential issues
- **Information**: Blue for helpful hints
- **Success**: Green for positive confirmations

### Color Accessibility

#### Contrast Requirements
- **Minimum 4.5:1**: Text to background contrast ratio
- **Minimum 3:1**: UI component contrast ratios
- **Enhanced Contrast**: Available high-contrast mode options

#### Color-blind Considerations
- **Multiple Indicators**: Never rely solely on color for information
- **Shape and Typography**: Use icons, text, and positioning as primary indicators
- **Pattern Support**: Alternative visual patterns for color-coded information

## Spacing System

### Spatial Hierarchy

Zed employs a consistent spacing system that creates clear visual hierarchies and maintains comfortable information density across all interface elements.

#### Base Spacing Units
- **4px**: Minimum spacing increment
- **8px**: Standard small spacing
- **12px**: Medium spacing
- **16px**: Standard spacing unit
- **20px**: Large spacing
- **24px**: Extra-large spacing
- **32px**: Section breaks

### Panel Spacing

#### Project Panel Configuration
```json
{
  "project_panel": {
    "entry_spacing": "comfortable",  // or "standard"
    "indent_size": 20,              // Hierarchical indentation
    "default_width": 240            // Panel width
  }
}
```

#### Spacing Options
- **Comfortable**: Generous spacing for reduced visual density
- **Standard**: Balanced spacing for optimal information display

### Editor Spacing

#### Gutter Spacing
```json
{
  "gutter": {
    "min_line_number_digits": 4  // Reserve space for line numbers
  }
}
```

#### Scroll Margins
```json
{
  "vertical_scroll_margin": 3,    // Lines above/below cursor
  "horizontal_scroll_margin": 5   // Characters left/right of cursor
}
```

### Content Spacing

#### Indent Guides
```json
{
  "indent_guides": {
    "line_width": 1,              // Guide line thickness
    "active_line_width": 1        // Active guide thickness
  }
}
```

#### Panel Indentation
- **Indent Size**: 20px standard for hierarchical content
- **Nested Elements**: Consistent indentation multipliers
- **Visual Guides**: Optional indent guide lines for structure clarity

### Responsive Spacing

#### Panel Dimensions
```json
{
  "agent": {
    "default_width": 640,   // Right/left docked
    "default_height": 320   // Bottom docked
  },
  "terminal": {
    "default_width": 640,
    "default_height": 320
  }
}
```

#### Adaptive Sizing
- **Content-aware**: Panels adjust to content requirements
- **User Preferences**: Customizable default sizes
- **Minimum Constraints**: Ensure usability at small sizes

### Spacing Best Practices

#### Consistency Rules
- **8px Grid**: All spacing should align to 8-pixel grid system
- **Progressive Scale**: Larger elements use proportionally larger spacing
- **Context Sensitivity**: Spacing adjusts based on content type and density

#### Visual Rhythm
- **Vertical Rhythm**: Consistent line-height creates visual flow
- **Horizontal Rhythm**: Consistent margins and padding
- **Component Spacing**: Standardized gaps between interface elements
