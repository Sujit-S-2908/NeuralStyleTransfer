# Neural Style Transfer Application

A Streamlit web application that implements Neural Style Transfer, allowing users to combine the content of one image with the artistic style of another.

## Overview

This application uses deep learning techniques to transfer the style of one image (like a famous painting) onto the content of another image (like a photograph). The implementation is based on the original Neural Style Transfer paper by Gatys et al. and uses pre-trained VGG networks (VGG16 and VGG19) to extract content and style features.

## Features

-   **User-friendly Interface**: Easy-to-use Streamlit web interface
-   **Multiple Image Format Support**: Handles various image formats (JPG, PNG, BMP, WEBP, TIFF, GIF)
-   **Customizable Parameters**: Adjust content weight, style weight, and TV weight
-   **Multiple Optimization Methods**: Choose between L-BFGS and Adam optimizers
-   **Different Initialization Options**: Content, style, or random initialization
-   **Real-time Progress Tracking**: Monitor the optimization process
-   **High-quality Output**: Download results in PNG or JPEG format
-   **Comprehensive Tips**: Guidance for achieving the best results with different image types

## Installation

### Prerequisites

-   Python 3.6+
-   PyTorch
-   CUDA-capable GPU (optional but recommended for faster processing)

### Setup

1. Clone the repository:

    ```
    git clone https://github.com/Sudharshan-3904/NeuralStyleTransfer.git
    cd NeuralStyleTransfer
    ```

2. Create a virtual environment (optional but recommended):

    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Start the Streamlit application:

    ```
    streamlit run main.py
    ```

2. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Upload a content image and a style image using the sidebar

4. Adjust the parameters as desired:

    - **Image Height**: Controls the resolution of the processed images
    - **Content Weight**: Influences how much the content of the original image is preserved
    - **Style Weight**: Determines the strength of the style transfer
    - **TV Weight**: Controls the smoothness of the generated image
    - **Optimizer**: Choose between L-BFGS (better quality) or Adam (faster)
    - **Model**: Select VGG16 or VGG19 for feature extraction
    - **Initialization Method**: Start from content image, style image, or random noise
    - **Saving Frequency**: Control how often intermediate results are saved

5. Click "Generate Styled Image" to start the process

6. Once complete, view and download the result

## Parameter Recommendations

### For L-BFGS optimizer:

-   **Content initialization**: Content weight = 1e5, Style weight = 3e4, TV weight = 1e0
-   **Style initialization**: Content weight = 1e5, Style weight = 1e1, TV weight = 1e-1
-   **Random initialization**: Content weight = 1e5, Style weight = 1e3, TV weight = 1e0

### For Adam optimizer:

-   **Content initialization**: Content weight = 1e5, Style weight = 1e5, TV weight = 1e-1
-   **Style initialization**: Content weight = 1e5, Style weight = 1e2, TV weight = 1e-1
-   **Random initialization**: Content weight = 1e5, Style weight = 1e2, TV weight = 1e-1

## Tips for Different Image Types

### Content Images:

-   Photos with clear subjects work best
-   Images with good contrast and well-defined features transfer better
-   For portraits, higher content weight preserves facial features
-   For landscapes, lower content weight allows more stylistic freedom

### Style Images:

-   Abstract paintings work well for dramatic style transfers
-   For subtle effects, use style images with similar color palettes to your content
-   High-contrast style images with distinct patterns create more pronounced effects
-   Experiment with different style image resolutions

### Resolution Settings:

-   Start with lower resolutions (300-400px) for faster iterations
-   Once you find parameters you like, increase resolution for final output
-   Very high resolutions (>800px) may require more iterations and memory

## Project Structure

```
neural-style-transfer/
├── main.py                  # Main application file with Streamlit UI
├── requirements.txt         # Python dependencies
├── data/                    # Directory for storing images
│   ├── content-images/      # Content images uploaded by users
│   ├── style-images/        # Style images uploaded by users
│   └── output-images/       # Generated output images
├── models/                  # Neural network model definitions
│   └── definitions/
│       └── vgg_nets.py      # VGG16 and VGG19 model implementations
└── utils/                   # Utility functions
    ├── utils.py             # Image processing and model utilities
    └── video_utils.py       # Utilities for creating videos from results
```

## How It Works

1. **Feature Extraction**: The application uses pre-trained VGG networks to extract content and style features from the input images.

2. **Loss Function**: Three loss components are calculated:

    - Content loss: Measures how different the content features are between the generated image and the content image
    - Style loss: Measures how different the style features are between the generated image and the style image
    - Total variation loss: Encourages spatial smoothness in the generated image

3. **Optimization**: The application optimizes the pixels of the generated image to minimize the weighted sum of these losses.

## Technical Details

-   **Content Representation**: Uses mid-level features from the VGG network (typically from layer 'conv4_2' or 'relu4_2')
-   **Style Representation**: Uses Gram matrices of features from multiple layers of the VGG network
-   **Optimization Methods**:
    -   L-BFGS: Slower but generally produces better results
    -   Adam: Faster but may require more parameter tuning

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

-   The original Neural Style Transfer paper by Gatys et al.: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
-   PyTorch for providing the deep learning framework and pre-trained VGG models
-   Streamlit for the interactive web application framework
