import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os
import argparse
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
    """
    try:
        img = Image.open(image_path).convert('RGB')
        aspect_ratio = img.width / img.height
        target_width = int(target_height * aspect_ratio)
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Save the resized image to a temporary file
        temp_path = os.path.join(os.path.dirname(image_path), f"temp_{os.path.basename(image_path)}")
        img.save(temp_path)
        
        # Pass the temporary file path to utils.prepare_img
        return utils.prepare_img(temp_path, target_height, device)
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")


def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        content_img = preprocess_image(content_img_path, config['height'], device)
        style_img = preprocess_image(style_img_path, config['height'], device)
    except ValueError as e:
        print(e)
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
    if config['optimizer'] == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[config['optimizer']]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif config['optimizer'] == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

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
        content_img_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
        style_img_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])
        
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
            content_img = Image.open(content_img_file).convert('RGB')
            content_img_name = f"uploaded_content_{int(time.time())}.jpg"
            content_img_path = os.path.join(content_images_dir, content_img_name)
            content_img.save(content_img_path)
            with col1:
                st.subheader("Content Image")
                st.image(content_img, width=300)
        except Exception as e:
            st.error(f"Error loading content image: {e}")
            content_img_path = None
    
    # Handle style image upload
    if style_img_file is not None:
        try:
            style_img = Image.open(style_img_file).convert('RGB')
            style_img_name = f"uploaded_style_{int(time.time())}.jpg"
            style_img_path = os.path.join(style_images_dir, style_img_name)
            style_img.save(style_img_path)
            with col2:
                st.subheader("Style Image")
                st.image(style_img, width=300)
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
            
            with st.spinner("Generating styled image... This might take a while depending on your settings."):
                # Run neural style transfer
                results_path = neural_style_transfer(optimization_config)
                if results_path is None:
                    st.error("Failed to process images. Please check the logs for details.")
                    return
                
                # Find the latest output image
                output_files = sorted([f for f in os.listdir(results_path) if f.endswith(img_format[1])])
                if output_files:
                    output_img_path = os.path.join(results_path, output_files[-1])
                    output_img = Image.open(output_img_path)
                    
                    st.subheader("Result Image")
                    st.image(output_img)
                    
                    # Provide download button
                    with open(output_img_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Result",
                            data=file,
                            file_name=f"nst_result_{int(time.time())}.jpg",
                            mime="image/jpeg"
                        )
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
        
        Experiment with different weight combinations to achieve the desired effect!
        """)

if __name__ == "__main__":
    main()

