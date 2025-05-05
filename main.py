import utils.utils as utils

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import io
import streamlit as st
from PIL import Image
import time


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = neural_net(optimizing_img)

    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def preprocess_image(image_path, target_height, device):
    """
    Preprocesses the image by resizing it to the target height while maintaining aspect ratio.
    Handles various image formats and potential issues with corrupted images.
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Open and validate image
        try:
            img = Image.open(image_path)
            # Verify the image can be read
            img.verify()
            # Reopen after verify (which closes the image)
            img = Image.open(image_path)
            # Convert to RGB (handles grayscale, RGBA, etc.)
            img = img.convert('RGB')
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {e}")

        # Get image dimensions and resize
        width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {width}x{height}")

        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)

        # Use LANCZOS for high-quality downsampling
        try:
            img = img.resize((target_width, target_height), Image.LANCZOS)
        except Exception:
            # Fallback to BICUBIC if LANCZOS fails
            img = img.resize((target_width, target_height), Image.BICUBIC)

        # Create a unique temporary file name to avoid conflicts
        temp_dir = os.path.dirname(image_path)
        temp_filename = f"temp_{int(time.time())}_{os.path.basename(image_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)

        # Save as PNG to preserve quality for processing
        img.save(temp_path, format="PNG")

        # Pass the temporary file path to utils.prepare_img
        processed_img = utils.prepare_img(temp_path, target_height, device)

        # Clean up the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            # Non-critical error, just log it
            print(f"Warning: Could not remove temporary file {temp_path}")

        return processed_img
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")


def validate_image(image_path):
    """
    Validates if an image file is readable and can be processed.
    Returns a tuple (is_valid, error_message)
    """
    if not os.path.exists(image_path):
        return False, f"Image file not found: {image_path}"

    try:
        with Image.open(image_path) as img:
            # Try to verify the image
            img.verify()

            # Reopen to check if we can access image data
            img = Image.open(image_path)
            img.load()

            # Check if image has valid dimensions
            if img.width <= 0 or img.height <= 0:
                return False, f"Invalid image dimensions: {img.width}x{img.height}"

            # Check if image mode is supported
            if img.mode not in ['RGB', 'RGBA', 'L', 'CMYK', 'YCbCr', 'LAB', 'HSV']:
                return False, f"Unsupported image mode: {img.mode}"

            return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    # Validate input images
    content_valid, content_error = validate_image(content_img_path)
    if not content_valid:
        print(f"Content image validation failed: {content_error}")
        return None

    style_valid, style_error = validate_image(style_img_path)
    if not style_valid:
        print(f"Style image validation failed: {style_error}")
        return None

    # Create a unique output directory name
    timestamp = int(time.time())
    out_dir_name = f'combined_{os.path.splitext(os.path.basename(content_img_path))[0]}_{os.path.splitext(os.path.basename(style_img_path))[0]}_{timestamp}'
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        content_img = preprocess_image(content_img_path, config['height'], device)
        style_img = preprocess_image(style_img_path, config['height'], device)
    except ValueError as e:
        print(f"Image preprocessing failed: {e}")
        return None

    # Adjust initialization dynamically based on input images
    if config['init_method'] == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    elif config['init_method'] == 'style':
        init_img = style_img
    else:
        raise ValueError(f"Unknown initialization method: {config['init_method']}")

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    #
    # Start of optimization procedure
    #
    # Get the progress callback if available
    progress_callback = config.get('progress_callback', None)

    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        total_iterations = num_of_iterations[config['optimizer']]

        for cnt in range(total_iterations):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)

            with torch.no_grad():
                # Print progress
                log_message = f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                print(log_message)

                # Update progress if callback is available
                if progress_callback and cnt % 10 == 0:  # Update every 10 iterations to avoid UI slowdown
                    progress_callback(cnt, total_iterations, f"Optimizing with Adam ({cnt}/{total_iterations})")

                # Save intermediate results
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, total_iterations, should_display=False)

    elif config['optimizer'] == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0
        total_iterations = num_of_iterations['lbfgs']

        def closure():
            nonlocal cnt

            if torch.is_grad_enabled():
                optimizer.zero_grad()

            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)

            if total_loss.requires_grad:
                total_loss.backward()

            with torch.no_grad():
                # Print progress
                log_message = f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                print(log_message)

                # Update progress if callback is available
                if progress_callback and cnt % 5 == 0:  # Update every 5 iterations for LBFGS
                    progress_callback(cnt, total_iterations, f"Optimizing with L-BFGS ({cnt}/{total_iterations})")

                # Save intermediate results
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, total_iterations, should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path


def main():
    st.title("Neural Style Transfer App")
    st.write("Upload a content image and a style image to create an artistic neural style transfer.")

    # Create necessary directories if they don't exist
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')

    for directory in [default_resource_dir, content_images_dir, style_images_dir, output_img_dir]:
        os.makedirs(directory, exist_ok=True)

    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    # Sidebar for uploading images and setting parameters
    with st.sidebar:
        st.header("Input Images")
        content_img_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "gif"])
        style_img_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff", "gif"])

        st.header("Parameters")
        height = st.slider("Image Height", min_value=100, max_value=800, value=400, step=50,
                           help="Height to resize images before processing")

        st.subheader("Loss Weights")
        content_weight = st.number_input("Content Weight", min_value=1.0, max_value=1e7, value=1e5,
                                         format="%.1e", help="Weight factor for content loss")
        style_weight = st.number_input("Style Weight", min_value=1.0, max_value=1e7, value=3e4,
                                       format="%.1e", help="Weight factor for style loss")
        tv_weight = st.number_input("Total Variation Weight", min_value=1e-3, max_value=1e3, value=1.0,
                                    format="%.1e", help="Weight factor for total variation loss")

        st.subheader("Algorithm Settings")
        optimizer = st.selectbox("Optimizer", options=["lbfgs", "adam"], index=0)
        model = st.selectbox("Model", options=["vgg19", "vgg16"], index=0)
        init_method = st.selectbox("Initialization Method", options=["content", "style", "random"], index=0)
        saving_freq = st.number_input("Saving Frequency", min_value=-1, max_value=50, value=-1,
                                      help="Saving frequency for intermediate images (-1 means only final)")

    # Main area for displaying images
    col1, col2 = st.columns(2)

    content_img_path = None
    style_img_path = None

    # Handle content image upload
    if content_img_file is not None:
        try:
            # Get file extension from uploaded file
            file_ext = os.path.splitext(content_img_file.name)[1].lower()
            if not file_ext:
                file_ext = '.jpg'  # Default extension if none is found

            # Open and validate the image
            content_img = Image.open(content_img_file)

            # Convert to RGB (handles grayscale, RGBA, etc.)
            content_img = content_img.convert('RGB')

            # Generate a unique filename with the original extension
            content_img_name = f"uploaded_content_{int(time.time())}{file_ext}"
            content_img_path = os.path.join(content_images_dir, content_img_name)

            # Save the image with appropriate quality
            if file_ext in ['.jpg', '.jpeg']:
                content_img.save(content_img_path, format="JPEG", quality=95)
            elif file_ext == '.png':
                content_img.save(content_img_path, format="PNG")
            elif file_ext == '.webp':
                content_img.save(content_img_path, format="WEBP", quality=95)
            elif file_ext == '.gif':
                content_img.save(content_img_path, format="GIF")
            elif file_ext == '.bmp':
                content_img.save(content_img_path, format="BMP")
            elif file_ext in ['.tiff', '.tif']:
                content_img.save(content_img_path, format="TIFF")
            else:
                # Default to PNG for unknown formats
                content_img_path = os.path.join(content_images_dir, f"uploaded_content_{int(time.time())}.png")
                content_img.save(content_img_path, format="PNG")

            # Display the image
            with col1:
                st.subheader("Content Image")
                st.image(content_img, width=300)
                st.caption(f"Original size: {content_img.width}x{content_img.height}")
        except Exception as e:
            st.error(f"Error loading content image: {e}")
            content_img_path = None

    # Handle style image upload
    if style_img_file is not None:
        try:
            # Get file extension from uploaded file
            file_ext = os.path.splitext(style_img_file.name)[1].lower()
            if not file_ext:
                file_ext = '.jpg'  # Default extension if none is found

            # Open and validate the image
            style_img = Image.open(style_img_file)

            # Convert to RGB (handles grayscale, RGBA, etc.)
            style_img = style_img.convert('RGB')

            # Generate a unique filename with the original extension
            style_img_name = f"uploaded_style_{int(time.time())}{file_ext}"
            style_img_path = os.path.join(style_images_dir, style_img_name)

            # Save the image with appropriate quality
            if file_ext in ['.jpg', '.jpeg']:
                style_img.save(style_img_path, format="JPEG", quality=95)
            elif file_ext == '.png':
                style_img.save(style_img_path, format="PNG")
            elif file_ext == '.webp':
                style_img.save(style_img_path, format="WEBP", quality=95)
            elif file_ext == '.gif':
                style_img.save(style_img_path, format="GIF")
            elif file_ext == '.bmp':
                style_img.save(style_img_path, format="BMP")
            elif file_ext in ['.tiff', '.tif']:
                style_img.save(style_img_path, format="TIFF")
            else:
                # Default to PNG for unknown formats
                style_img_path = os.path.join(style_images_dir, f"uploaded_style_{int(time.time())}.png")
                style_img.save(style_img_path, format="PNG")

            # Display the image
            with col2:
                st.subheader("Style Image")
                st.image(style_img, width=300)
                st.caption(f"Original size: {style_img.width}x{style_img.height}")
        except Exception as e:
            st.error(f"Error loading style image: {e}")
            style_img_path = None

    # Process button
    process_button = st.button("Generate Styled Image", disabled=(content_img_path is None or style_img_path is None))

    if process_button and content_img_path and style_img_path:
        try:
            # Create configuration dictionary
            optimization_config = {
                'content_img_name': os.path.basename(content_img_path),
                'style_img_name': os.path.basename(style_img_path),
                'height': height,
                'content_weight': content_weight,
                'style_weight': style_weight,
                'tv_weight': tv_weight,
                'optimizer': optimizer,
                'model': model,
                'init_method': init_method,
                'saving_freq': saving_freq,
                'content_images_dir': content_images_dir,
                'style_images_dir': style_images_dir,
                'output_img_dir': output_img_dir,
                'img_format': img_format
            }

            # Create a progress placeholder
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a function to update progress
            def update_progress(iteration, total, message=""):
                progress = min(iteration / total, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"{message} - {int(progress * 100)}% complete")

            with st.spinner("Preparing for neural style transfer..."):
                # Display device information
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                progress_placeholder.info(f"Using device: {device} - {'GPU acceleration available' if torch.cuda.is_available() else 'CPU only (this might be slow)'}")

                # Add a callback to the config to track progress
                optimization_config['progress_callback'] = update_progress

                # Run neural style transfer
                results_path = neural_style_transfer(optimization_config)
                if results_path is None:
                    st.error("Failed to process images. Please check the logs for details.")
                    return

                # Complete the progress bar
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                progress_placeholder.success("Neural style transfer completed successfully!")

                # Find the latest output image
                output_files = sorted([f for f in os.listdir(results_path) if f.endswith(img_format[1])])
                if output_files:
                    output_img_path = os.path.join(results_path, output_files[-1])

                    try:
                        # Open the output image
                        output_img = Image.open(output_img_path)

                        # Display the result
                        st.subheader("Result Image")
                        st.image(output_img)

                        # Create a high-quality version for download
                        download_img = output_img.copy()
                        download_buffer = io.BytesIO()
                        download_img.save(download_buffer, format="PNG")
                        download_buffer.seek(0)

                        # Provide download button with PNG format for better quality
                        st.download_button(
                            label="Download Result (PNG)",
                            data=download_buffer,
                            file_name=f"nst_result_{int(time.time())}.png",
                            mime="image/png"
                        )

                        # Also provide JPEG download option (smaller file size)
                        jpeg_buffer = io.BytesIO()
                        download_img.save(jpeg_buffer, format="JPEG", quality=95)
                        jpeg_buffer.seek(0)

                        st.download_button(
                            label="Download Result (JPEG)",
                            data=jpeg_buffer,
                            file_name=f"nst_result_{int(time.time())}.jpg",
                            mime="image/jpeg"
                        )

                        # Show image information
                        st.caption(f"Result size: {output_img.width}x{output_img.height}")

                    except Exception as e:
                        st.error(f"Error displaying output image: {e}")
                else:
                    st.error("Failed to generate output image.")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

    # Display tips in an expander
    with st.expander("Tips for good results"):
        st.markdown("""
        ### Parameter suggestions:

        **For LBFGS optimizer:**
        - Content initialization: Content weight = 1e5, Style weight = 3e4, TV weight = 1e0
        - Style initialization: Content weight = 1e5, Style weight = 1e1, TV weight = 1e-1
        - Random initialization: Content weight = 1e5, Style weight = 1e3, TV weight = 1e0

        **For Adam optimizer:**
        - Content initialization: Content weight = 1e5, Style weight = 1e5, TV weight = 1e-1
        - Style initialization: Content weight = 1e5, Style weight = 1e2, TV weight = 1e-1
        - Random initialization: Content weight = 1e5, Style weight = 1e2, TV weight = 1e-1

        ### Tips for different image types:

        **Content images:**
        - Photos with clear subjects work best
        - Images with good contrast and well-defined features transfer better
        - For portraits, higher content weight preserves facial features
        - For landscapes, lower content weight allows more stylistic freedom

        **Style images:**
        - Abstract paintings work well for dramatic style transfers
        - For subtle effects, use style images with similar color palettes to your content
        - High-contrast style images with distinct patterns create more pronounced effects
        - Experiment with different style image resolutions - sometimes smaller style images produce better results

        **Resolution settings:**
        - Start with lower resolutions (300-400px) for faster iterations
        - Once you find parameters you like, increase resolution for final output
        - Very high resolutions (>800px) may require more iterations and memory

        **Optimization tips:**
        - LBFGS generally produces better results but is slower
        - Adam is faster but may require more parameter tuning
        - Content initialization preserves more details from the original image
        - Style initialization creates more dramatic style effects
        - Random initialization can sometimes find unique solutions

        Experiment with different weight combinations to achieve the desired effect!
        """)

if __name__ == "__main__":
    main()

