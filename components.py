"""
Shared components and utilities for the scan post processing app.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from dash import dcc
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import json

# Configuration
logo = "https://github.com/user-attachments/assets/c1ff143b-4365-4fd1-880f-3e97aab5c302"
page_viewer_height = 600

# Setup data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")

def get_config_file_path():
    """Get the path to the configuration file."""
    config_path = os.path.join(script_dir, "config.json")
    return config_path

def load_config():
    """Load configuration from file, return defaults if file doesn't exist."""
    config_path = get_config_file_path()
    default_config = {
        "images_directory": r"C:\Users\adiad\Downloads\jpgs",  # Fallback to original path
        "app_name": "Scan Post Processing"
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
    except Exception:
        pass
    
    return default_config

def get_pages_directory():
    """Get the configured pages directory."""
    config = load_config()
    return config.get("images_directory", r"C:\Users\adiad\Downloads\jpgs")

def get_pages_list():
    """Get list of pages from the configured directory."""
    pages_dir = get_pages_directory()
    
    if not pages_dir or not os.path.exists(pages_dir):
        return [""]  # Return empty list if directory doesn't exist
    
    try:
        pages = [p for p in os.listdir(pages_dir) if p.lower().endswith(('.jpg', '.jpeg'))]
        pages = [""] + pages  # add empty option
        pages.sort()
        return pages
    except Exception:
        return [""]

# Get list of pages (dynamic based on configuration)
pages_dir = get_pages_directory()
pages = get_pages_list()

def create_empty_figure():
    """Create an empty figure for when no page is selected."""
    empty_fig = px.imshow(np.ones((page_viewer_height, int(8.5/11*page_viewer_height), 3), dtype=np.uint8)*255)
    empty_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=page_viewer_height)
    empty_fig.update_xaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][4],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image bottom border
    )
    empty_fig.update_yaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][4],
        mirror=True,
    )
    return empty_fig

def create_page_select(page_context):
    """Create a page selection dropdown with unique ID for different pages.
    
    Args:
        page_context: String identifier for the page (e.g., 'viewer', 'editor')
    """
    current_pages = get_pages_list()
    
    if len(current_pages) <= 1:  # Only empty option or no options
        return dmc.Stack([
            dmc.Select(
                label="Jump to page",
                placeholder="No images found",
                id=f"{page_context}-page-select",
                value="",
                data=[{"value": "", "label": "No images found"}],
                w=140,
                mb=5,
                disabled=True
            ),
            dmc.Anchor(
                "Configure in Setup",
                href="/setup",
                size="xs",
                c="blue"
            )
        ], gap=2)
    
    # Start with first non-empty page if available
    initial_value = current_pages[1] if len(current_pages) > 1 else current_pages[0]
    
    return dmc.Select(
        label="Jump to page",
        placeholder="Select one",
        id=f"{page_context}-page-select",
        value=initial_value,
        data=[{"value": page, "label": page} for page in current_pages],
        w=140,
        mb=10,
    )

def create_navigation_buttons(page_context):
    """Create previous/next navigation buttons with unique IDs for different pages.
    
    Args:
        page_context: String identifier for the page (e.g., 'viewer', 'editor')
    """
    return dmc.ActionIconGroup(
        [
            dmc.ActionIcon(
                variant="default",
                size="lg",
                children=DashIconify(icon="ion:chevron-back", width=15),
                id=f"{page_context}-prev-page-btn"
            ),
            dmc.ActionIcon(
                variant="default",
                size="lg",
                children=DashIconify(icon="ion:chevron-forward", width=15),
                id=f"{page_context}-next-page-btn"
            ),
        ],
        orientation="horizontal",
        mt=15
    )

def create_app_header():
    """Create the app header with logo and title."""
    return dmc.AppShellHeader(
        dmc.Group(
            [
                dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
                dmc.Image(src=logo, h=40, flex=0),
                dmc.Title("Scan Post Processing App", c="blue"),
            ],
            h="100%",
            px="md",
        )
    )

def create_page_navigation():
    """Create navigation links for switching between pages."""
    return dmc.Group([
        dmc.Anchor("Automated Result Viewer", href="/", underline=False, c="blue"),
        dmc.Text("•", c="gray"),
        dmc.Anchor("Manual Result Editor", href="/editor", underline=False, c="blue"),
    ], gap="sm")

def get_image_path(page, step):
    """Get the image path for a given page and processing step."""
    pages_directory = get_pages_directory()
    if step == 1:
        return os.path.join(pages_directory, page)
    elif step == 2:
        return os.path.join(data_dir, "2_rotate", page)
    elif step == 3:
        return os.path.join(data_dir, "3_crop", page)
    elif step == 4:
        return os.path.join(data_dir, "4_margin", page)
    elif step == 5:
        return os.path.join(data_dir, "5_resize", page)
    elif step == 6:
        return os.path.join(data_dir, "6_final", page)
    else:
        return os.path.join(pages_directory, page)

def create_error_figure(message="Configuration required"):
    """Create an error figure with a message."""
    error_fig = px.imshow(np.ones((page_viewer_height, int(8.5/11*page_viewer_height), 3), dtype=np.uint8)*240)
    error_fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        height=page_viewer_height,
        annotations=[
            dict(
                text=message,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                xanchor="center", yanchor="middle",
                font=dict(size=16, color="red"),
                showarrow=False
            )
        ]
    )
    error_fig.update_xaxes(showticklabels=False, showline=False)
    error_fig.update_yaxes(showticklabels=False, showline=False)
    return error_fig

def create_page_figure(page, step=0, show_bboxes=False):
    """Create a plotly figure for displaying a page image."""
    if not isinstance(page, str) or not page:
        return create_empty_figure()
    
    # Check if pages directory is configured and exists
    pages_directory = get_pages_directory()
    if not pages_directory or not os.path.exists(pages_directory):
        return create_error_figure("Please configure image directory in Setup")
    
    img_path = get_image_path(page, step)
    
    if not os.path.exists(img_path):
        if step == 1:  # Original image should exist
            return create_error_figure(f"Image not found: {page}")
        else:
            return create_empty_figure()  # Processed images might not exist yet
    
    try:
        image = Image.open(img_path)
    except Exception as e:
        return create_error_figure(f"Error loading image: {str(e)}")
    fig = px.imshow(np.asarray(image))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=page_viewer_height)
    fig.update_xaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][5],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image bottom border
    )
    fig.update_yaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][5],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image left border
    )
    
    # Add bounding boxes if requested
    if show_bboxes:
        df_path = os.path.join(data_dir, "df.pkl")
        if os.path.exists(df_path):
            df = pd.read_pickle(df_path)
            bbox_line_style = {"color": dmc.DEFAULT_THEME["colors"]["blue"][2]}
            
            if (step == 1) and (page in df.index):
                # draw bounding boxes around each line of text
                bboxes = df.at[page, "line_bboxes"]
                for bbox in bboxes:
                    fig.add_shape(type="line", x0=bbox[0][0], y0=bbox[0][1], x1=bbox[1][0], y1=bbox[1][1], line=bbox_line_style)
                    fig.add_shape(type="line", x0=bbox[1][0], y0=bbox[1][1], x1=bbox[2][0], y1=bbox[2][1], line=bbox_line_style)
                    fig.add_shape(type="line", x0=bbox[2][0], y0=bbox[2][1], x1=bbox[3][0], y1=bbox[3][1], line=bbox_line_style)
                    fig.add_shape(type="line", x0=bbox[3][0], y0=bbox[3][1], x1=bbox[0][0], y1=bbox[0][1], line=bbox_line_style)
            
            if (step == 2) and (page in df.index):
                # draw bounding box around all text
                x_min, y_min, x_max, y_max = df.at[page, "all_text_bbox"]
                fig.add_shape(type="line", x0=x_min, y0=y_min, x1=x_max, y1=y_min, line=bbox_line_style)
                fig.add_shape(type="line", x0=x_max, y0=y_min, x1=x_max, y1=y_max, line=bbox_line_style)
                fig.add_shape(type="line", x0=x_max, y0=y_max, x1=x_min, y1=y_max, line=bbox_line_style)
                fig.add_shape(type="line", x0=x_min, y0=y_max, x1=x_min, y1=y_min, line=bbox_line_style)
    
    return fig

def create_page_info_card(page, step=0):
    """Create an info card with page details."""
    if not isinstance(page, str) or not page:
        return None
    
    img_path = get_image_path(page, step)
    
    if not os.path.exists(img_path):
        return None
    
    image = Image.open(img_path)
    height = image.height
    width = image.width
    
    info_lines = [
        dmc.Title("Page Info", order=6),
        dmc.Group([dmc.Text("Image Height:", fw=500), dmc.Text(f"{height} px")], gap="xs"),
        dmc.Group([dmc.Text("Image Width:", fw=500), dmc.Text(f"{width} px")], gap="xs"),
    ]
    
    # Add step-specific information
    df_path = os.path.join(data_dir, "df.pkl")
    if os.path.exists(df_path):
        df = pd.read_pickle(df_path)
        
        if (step == 1) and (page in df.index):
            bboxes = df.at[page, "line_bboxes"]
            info_lines += [
                dmc.Group([dmc.Text("N Bounding Boxes:", fw=500), dmc.Text(str(len(bboxes)))], gap="xs"),
                dmc.Group([dmc.Text("Bounding boxes represent lines of text. Each line's angle from horizontal is measured. The page will be rotated based on the median angle from horizontal.")], gap="xs")
            ]
        
        if (step == 2) and (page in df.index):
            angle = df.at[page, "angle"]
            info_lines += [
                dmc.Group([dmc.Text("Rotation Angle:", fw=500), dmc.Text(f"{angle}°")], gap="xs"),
                dmc.Group([dmc.Text("The bounding box should encapsulate all text on the page. It will be used to crop out all white space.")], gap="xs"),
            ]
    
    return dmc.Card(dmc.Group(info_lines, gap="xs"))


def create_rotated_page_figure_with_manual_crop(page, rotation_angle=0, step=0, show_auto_bboxes=False, 
                                               show_manual_bbox=False, manual_bbox=None, dragmode="zoom"):
    """Create a plotly figure for displaying a page image with custom rotation and manual crop capability."""
    if not isinstance(page, str) or not page:
        return create_empty_figure()
    
    img_path = get_image_path(page, step)
    
    if not os.path.exists(img_path):
        return create_empty_figure()
    
    # Load and rotate the image
    image = Image.open(img_path)
    
    # Apply rotation if specified
    if rotation_angle != 0:
        # Rotate image with expansion and white background
        rotated_image = image.rotate(
            angle=rotation_angle,
            expand=True,
            fillcolor='white',
            resample=Image.BICUBIC
        )
    else:
        rotated_image = image
    
    fig = px.imshow(np.asarray(rotated_image))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), 
        height=page_viewer_height,
        dragmode=dragmode
    )
    fig.update_xaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][5],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image bottom border
    )
    fig.update_yaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][5],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image left border
    )
    
    # Add automatic bounding boxes if requested (only for original non-rotated images)
    if show_auto_bboxes and rotation_angle == 0:
        df_path = os.path.join(data_dir, "df.pkl")
        if os.path.exists(df_path):
            df = pd.read_pickle(df_path)
            
            if page in df.index:
                # Only show the overall text bbox (not individual line bboxes)
                all_text_bbox = df.at[page, "all_text_bbox"]
                if (isinstance(all_text_bbox, (list, tuple)) and len(all_text_bbox) == 4 
                    and all(isinstance(coord, (int, float)) for coord in all_text_bbox)):
                    x_min, y_min, x_max, y_max = all_text_bbox
                    overall_bbox_style = {"color": dmc.DEFAULT_THEME["colors"]["red"][4], "width": 2}
                    fig.add_shape(type="line", x0=x_min, y0=y_min, x1=x_max, y1=y_min, line=overall_bbox_style)
                    fig.add_shape(type="line", x0=x_max, y0=y_min, x1=x_max, y1=y_max, line=overall_bbox_style)
                    fig.add_shape(type="line", x0=x_max, y0=y_max, x1=x_min, y1=y_max, line=overall_bbox_style)
                    fig.add_shape(type="line", x0=x_min, y0=y_max, x1=x_min, y1=y_min, line=overall_bbox_style)
    
    # Add manual bounding box if requested
    if show_manual_bbox and manual_bbox:
        x0, y0, x1, y1 = manual_bbox['x0'], manual_bbox['y0'], manual_bbox['x1'], manual_bbox['y1']
        manual_bbox_style = {"color": dmc.DEFAULT_THEME["colors"]["green"][4], "width": 2}
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=manual_bbox_style,
            fillcolor="rgba(0,255,0,0.1)"
        )
    
    return fig


def create_rotated_page_figure(page, rotation_angle=0, step=0, show_bboxes=False):
    """Create a plotly figure for displaying a page image with custom rotation applied."""
    if not isinstance(page, str) or not page:
        return create_empty_figure()
    
    img_path = get_image_path(page, step)
    
    if not os.path.exists(img_path):
        return create_empty_figure()
    
    # Load and rotate the image
    image = Image.open(img_path)
    
    # Apply rotation if specified
    if rotation_angle != 0:
        # Rotate image with expansion and white background
        rotated_image = image.rotate(
            angle=rotation_angle,
            expand=True,
            fillcolor='white',
            resample=Image.BICUBIC
        )
    else:
        rotated_image = image
    
    fig = px.imshow(np.asarray(rotated_image))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=page_viewer_height)
    fig.update_xaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][5],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image bottom border
    )
    fig.update_yaxes(
        showticklabels=False,
        showline=True,
        linewidth=2,
        linecolor=dmc.DEFAULT_THEME["colors"]["blue"][5],
        mirror=True,
        title="" # this is necessary so the parent container doesn't clip the image left border
    )
    
    # Add bounding boxes if requested (only for original non-rotated images to avoid coordinate transformation complexity)
    if show_bboxes and rotation_angle == 0:
        df_path = os.path.join(data_dir, "df.pkl")
        if os.path.exists(df_path):
            df = pd.read_pickle(df_path)
            
            if page in df.index:
                # Only show the overall text bbox (not individual line bboxes)
                all_text_bbox = df.at[page, "all_text_bbox"]
                if (isinstance(all_text_bbox, (list, tuple)) and len(all_text_bbox) == 4 
                    and all(isinstance(coord, (int, float)) for coord in all_text_bbox)):
                    x_min, y_min, x_max, y_max = all_text_bbox
                    overall_bbox_style = {"color": dmc.DEFAULT_THEME["colors"]["red"][4], "width": 2}
                    fig.add_shape(type="line", x0=x_min, y0=y_min, x1=x_max, y1=y_min, line=overall_bbox_style)
                    fig.add_shape(type="line", x0=x_max, y0=y_min, x1=x_max, y1=y_max, line=overall_bbox_style)
                    fig.add_shape(type="line", x0=x_max, y0=y_max, x1=x_min, y1=y_max, line=overall_bbox_style)
                    fig.add_shape(type="line", x0=x_min, y0=y_max, x1=x_min, y1=y_min, line=overall_bbox_style)
    
    return fig


def float_img_to_top(img, target_height):
    """Float image to top by adding white space at bottom."""
    from PIL import ImageOps
    delta_height = target_height - img.height
    return ImageOps.expand(img, border=(0, 0, 0, delta_height), fill="white")


def float_img_to_center(img, target_height):
    """Float image to center by adding white space at top and bottom."""
    from PIL import ImageOps
    delta_height = target_height - img.height
    top_border = delta_height // 2
    bottom_border = delta_height - top_border
    return ImageOps.expand(img, border=(0, top_border, 0, bottom_border), fill="white")


def float_img_to_bottom(img, target_height):
    """Float image to bottom by adding white space at top."""
    from PIL import ImageOps
    delta_height = target_height - img.height
    return ImageOps.expand(img, border=(0, delta_height, 0, 0), fill="white")


def get_standard_dimensions():
    """
    Calculate the standard dimensions used for image processing.
    This replicates the logic from jpgs_to_pdf.py lines 118-140.
    
    Returns:
        tuple: (final_width, final_height) 
    """
    import pandas as pd
    
    desired_aspect_ratio = 9 / 5.5  # width / height (same as in jpgs_to_pdf.py)
    
    # Load the dataframe with text bounding boxes
    df_path = os.path.join(data_dir, "df.pkl")
    if not os.path.exists(df_path):
        # Fallback to a reasonable default if dataframe doesn't exist
        default_width = 1000
        return default_width, int(default_width * desired_aspect_ratio)
    
    df = pd.read_pickle(df_path)
    
    # Determine the minimum width of text on all pages (ignoring first and last pages of chapters)
    # This replicates the logic from jpgs_to_pdf.py
    min_width = float('inf')
    
    for page in df.index:
        # For simplicity, we'll analyze all pages since we don't have chapter information here
        # In the original script, first and last pages of chapters were ignored
        line_bboxes = df.at[page, "line_bboxes"]
        if len(line_bboxes) == 0:
            continue
            
        # Calculate width from line bboxes
        x_coords = [point[0] for bbox in line_bboxes for point in bbox]
        if x_coords:
            min_x, max_x = min(x_coords), max(x_coords)
            width = max_x - min_x
            if width < min_width:
                min_width = width
    
    # If no valid width found, use a reasonable default
    if min_width == float('inf'):
        min_width = 1000
    
    final_width = int(min_width)
    final_height = int(min_width * desired_aspect_ratio)
    
    return final_width, final_height


def process_image_with_manual_adjustments(page, manual_adjustments):
    """
    Process an image with manual adjustments including rotation, cropping, margins, and float position.
    
    Args:
        page: The image page filename
        manual_adjustments: Dictionary with manual adjustment parameters:
            - rotation: rotation angle in degrees
            - top_bottom_margin: top/bottom margin percentage
            - left_right_margin: left/right margin percentage  
            - float_position: "top", "center", or "bottom"
            - manual_crop: boolean for manual crop mode
            - manual_bbox: manual bounding box coordinates (if manual_crop is True)
    
    Returns:
        PIL Image object with applied adjustments
    """
    from PIL import Image, ImageOps
    import math
    
    # Load original image
    pages_directory = get_pages_directory()
    if not pages_directory or not os.path.exists(pages_directory):
        raise ValueError("Image directory not configured or does not exist")
    
    original_path = os.path.join(pages_directory, page)
    if not os.path.exists(original_path):
        raise ValueError(f"Image file not found: {page}")
    
    try:
        img = Image.open(original_path)
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")
    
    # Apply rotation if specified
    rotation = manual_adjustments.get("rotation", 0)
    if rotation != 0:
        img = img.rotate(-rotation, expand=True, fillcolor="white")
    
    # Apply cropping
    if manual_adjustments.get("manual_crop", False) and "manual_bbox" in manual_adjustments:
        # Use manual bounding box
        bbox = manual_adjustments["manual_bbox"]
        img = img.crop((bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']))
    else:
        # Use automatic text boundary detection (if available)
        df_path = os.path.join(data_dir, "df.pkl")
        if os.path.exists(df_path):
            import pandas as pd
            df = pd.read_pickle(df_path)
            if page in df.index:
                all_text_bbox = df.at[page, "all_text_bbox"]
                if (isinstance(all_text_bbox, (list, tuple)) and len(all_text_bbox) == 4 
                    and all(isinstance(coord, (int, float)) for coord in all_text_bbox)):
                    x_min, y_min, x_max, y_max = all_text_bbox
                    img = img.crop((x_min, y_min, x_max, y_max))
    
    # Apply margins
    top_bottom_margin = manual_adjustments.get("top_bottom_margin", 0) / 100.0
    left_right_margin = manual_adjustments.get("left_right_margin", 0) / 100.0
    
    if top_bottom_margin > 0 or left_right_margin > 0:
        tb_border = int(top_bottom_margin * img.height)
        lr_border = int(left_right_margin * img.width)
        img = ImageOps.expand(img, border=(lr_border, tb_border, lr_border, tb_border), fill="white")
    
    # Follow the same standardization logic as jpgs_to_pdf.py
    # 1. Determine final_width and final_height from the dataframe analysis
    desired_aspect_ratio = 9 / 5.5  # width / height (same as in jpgs_to_pdf.py)
    final_width, final_height = get_standard_dimensions()
    
    # 2. Resize the image to final_width while maintaining aspect ratio (like line 237 in jpgs_to_pdf.py)
    height = int(final_width / img.width * img.height)
    img = img.resize((final_width, height), resample=Image.LANCZOS)
    
    # 3. Apply float position only if height < final_height (like lines 240-262 in jpgs_to_pdf.py)
    float_position = manual_adjustments.get("float_position", "center")
    
    if height < final_height:
        if float_position == "top":
            img = float_img_to_top(img, final_height)
        elif float_position == "bottom":
            img = float_img_to_bottom(img, final_height)
        else:  # center (default)
            img = float_img_to_center(img, final_height)
    else:
        # If height >= final_height, resize to exact dimensions (like line 262 in jpgs_to_pdf.py)
        img = img.resize((final_width, final_height), resample=Image.LANCZOS)
    
    return img


def save_manual_adjustments(page, rotation=None, crop_bbox=None, tb_margin=None, lr_margin=None, float_pos=None):
    """Save manual adjustments for a page to the dataframe."""
    df_path = os.path.join(data_dir, "df.pkl")
    
    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Warning: DataFrame not found at {df_path}")
        return
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return
    
    try:
        # Initialize manual columns if they don't exist
        manual_columns = ['manual_rotation', 'manual_crop_bbox', 'manual_tb_margin', 'manual_lr_margin', 'manual_float']
        for col in manual_columns:
            if col not in df.columns:
                df[col] = None
        
        # Ensure the page exists; create a blank row if needed
        if page not in df.index:
            df.loc[page] = None

        # Update the values for the specified page
        if rotation is not None:
            df.loc[page, 'manual_rotation'] = rotation
        if crop_bbox is not None:
            df.loc[page, 'manual_crop_bbox'] = crop_bbox
        if tb_margin is not None:
            df.loc[page, 'manual_tb_margin'] = tb_margin
        if lr_margin is not None:
            df.loc[page, 'manual_lr_margin'] = lr_margin
        if float_pos is not None:
            df.loc[page, 'manual_float'] = float_pos

        # Save the updated dataframe
        df.to_pickle(df_path)
        print(f"Saved manual adjustments for {page}")
    except Exception as e:
        print(f"Error saving manual adjustments for {page}: {e}")


def load_manual_adjustments(page):
    """Load manual adjustments for a page from the dataframe."""
    df_path = os.path.join(data_dir, "df.pkl")
    
    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Warning: DataFrame not found at {df_path}")
        return {}
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return {}
    
    if page not in df.index:
        return {}
    
    # Get manual values if they exist
    manual_values = {}
    manual_columns = {
        'manual_rotation': 'rotation',
        'manual_crop_bbox': 'crop_bbox', 
        'manual_tb_margin': 'tb_margin',
        'manual_lr_margin': 'lr_margin',
        'manual_float': 'float_pos'
    }
    
    try:
        for df_col, key in manual_columns.items():
            if df_col in df.columns:
                value = df.loc[page, df_col]
                # More robust null checking that doesn't depend on pandas methods
                if value is not None and str(value).lower() != 'nan' and str(value) != 'None':
                    manual_values[key] = value
    except Exception as e:
        print(f"Error loading manual adjustments for {page}: {e}")
        return {}
    
    return manual_values


def clear_manual_adjustments(page):
    """Clear all manual adjustments for a page from the dataframe."""
    df_path = os.path.join(data_dir, "df.pkl")
    
    try:
        df = pd.read_pickle(df_path)
    except FileNotFoundError:
        print(f"Warning: DataFrame not found at {df_path}")
        return
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return
    
    try:
        if page in df.index:
            manual_columns = ['manual_rotation', 'manual_crop_bbox', 'manual_tb_margin', 'manual_lr_margin', 'manual_float']
            for col in manual_columns:
                if col in df.columns:
                    df.loc[page, col] = None
            
            # Save the updated dataframe
            df.to_pickle(df_path)
            print(f"Cleared manual adjustments for {page}")
    except Exception as e:
        print(f"Error clearing manual adjustments for {page}: {e}")
