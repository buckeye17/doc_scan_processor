"""
Batch image processing module for document scanning post-processing.
Extracted from prototype/jpgs_to_pdf_ppm_rotation.py for reuse in the application.
"""
import os
import shutil
import cv2
import easyocr
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm
import json
import threading
import time


class ProgressState:
    """Manages progress state for batch processing with multiple progress items."""
    
    def __init__(self, state_file):
        self.state_file = state_file
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def add_step(self, step_id, message, step_type="text"):
        """
        Add a new progress step.
        
        Args:
            step_id: Unique identifier for this step
            message: Description of the step
            step_type: "text" for simple text line, "progress" for progress bar
        """
        with self.lock:
            state = self._read_state()
            if "steps" not in state:
                state["steps"] = []
            
            state["steps"].append({
                "id": step_id,
                "type": step_type,
                "message": message,
                "status": "active",  # "active", "complete"
                "start_time": time.time(),
                "elapsed_time": 0,
                "current": 0,
                "total": 0,
                "percentage": 0
            })
            state["status"] = "processing"
            self._write_state(state)
    
    def update_step(self, step_id, message=None, current=None, total=None, status=None):
        """
        Update an existing step.
        
        Args:
            step_id: Identifier of the step to update
            message: Optional new message
            current: Optional current progress value (for progress bars)
            total: Optional total progress value (for progress bars)
            status: Optional new status ("active", "complete")
        """
        with self.lock:
            state = self._read_state()
            if "steps" not in state:
                return
            
            for step in state["steps"]:
                if step["id"] == step_id:
                    if message is not None:
                        step["message"] = message
                    if current is not None:
                        step["current"] = current
                    if total is not None:
                        step["total"] = total
                    if current is not None and total is not None and total > 0:
                        step["percentage"] = int((current / total) * 100)
                    if status is not None:
                        step["status"] = status
                    
                    # Update elapsed time
                    step["elapsed_time"] = time.time() - step["start_time"]
                    break
            
            self._write_state(state)
    
    def complete_step(self, step_id, message=None):
        """Mark a step as complete with optional final message."""
        with self.lock:
            state = self._read_state()
            if "steps" not in state:
                return
            
            for step in state["steps"]:
                if step["id"] == step_id:
                    step["status"] = "complete"
                    step["elapsed_time"] = time.time() - step["start_time"]
                    if message is not None:
                        step["message"] = message
                    break
            
            self._write_state(state)
    
    def set_complete(self, message="Processing complete!"):
        """Mark entire processing as complete."""
        with self.lock:
            state = self._read_state()
            state["status"] = "complete"
            state["message"] = message
            state["total_elapsed_time"] = time.time() - self.start_time
            self._write_state(state)
    
    def set_error(self, message):
        """Mark processing as error."""
        with self.lock:
            state = self._read_state()
            state["status"] = "error"
            state["message"] = message
            state["total_elapsed_time"] = time.time() - self.start_time
            self._write_state(state)
    
    def _read_state(self):
        """Read current state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {"steps": [], "status": "idle", "message": "", "total_elapsed_time": 0}
    
    def _write_state(self, state):
        """Write state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def get(self):
        """Get current progress state."""
        with self.lock:
            return self._read_state()
    
    def clear(self):
        """Clear progress state."""
        with self.lock:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            self.start_time = time.time()


def mkdir(path):
    """Create directory, deleting if it already exists."""
    if os.path.exists(path):
        # delete directory and everything in the directory
        shutil.rmtree(path)
    os.makedirs(path)


def projection_profile_method(image, delta=0.1, limit=15):
    """
    Detects the skew angle of a preprocessed (binary) image using the
    Projection Profile Method.

    Args:
        image: The binary input image (text should be white, background black).
        delta: The step size for angle search (in degrees).
        limit: The range of angles to search (e.g., limit=15 searches from -15 to +15).

    Returns:
        The estimated skew angle in degrees.
    """
    best_angle = 0
    max_score = -1
    
    # Iterate through a range of angles
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        # Rotate the image
        rotated = rotate(image, angle, reshape=False, order=0)
        
        # Compute the horizontal projection profile (sum of pixels per row)
        projection = np.sum(rotated, axis=1)
        
        # Calculate the criterion function (score)
        # Here, we use the sum of squared differences of adjacent profile values
        score = np.sum((projection[1:] - projection[:-1]) ** 2)
        
        # Update the best angle if the current score is higher
        if score > max_score:
            max_score = score
            best_angle = angle
            
    return best_angle


def float_img_to_top(img, target_height):
    """Float image to top by adding bottom border."""
    delta_height = target_height - img.height
    return ImageOps.expand(img, border=(0, 0, 0, delta_height), fill="white")


def float_img_to_center(img, target_height):
    """Float image to center by adding top and bottom borders."""
    delta_height = target_height - img.height
    top_border = delta_height // 2
    bottom_border = delta_height - top_border
    return ImageOps.expand(img, border=(0, top_border, 0, bottom_border), fill="white")


def float_img_to_bottom(img, target_height):
    """Float image to bottom by adding top border."""
    delta_height = target_height - img.height
    return ImageOps.expand(img, border=(0, delta_height, 0, 0), fill="white")


def process_images_batch(input_pages_dir, output_dir, chapters, desired_aspect_ratio=9/5.5, progress_state=None):
    """
    Batch process images according to chapter definitions.
    
    Args:
        input_pages_dir: Directory containing input JPG files
        output_dir: Directory where output will be saved
        chapters: List of chapter dictionaries with keys:
            - title: Chapter name
            - start: Start page filename (e.g., "image00001.jpg")
            - end: End page filename (e.g., "image00010.jpg")
            - is_text: Boolean indicating if chapter contains text
            - vertical_float: "Top", "Center", or "Bottom" for vertical alignment
            - horizontal_margin: Horizontal margin percentage
            - vertical_margin: Vertical margin percentage
            - aspect_ratio: Target aspect ratio for this chapter (width/height)
        desired_aspect_ratio: Default aspect ratio if not specified in chapter (width/height)
        progress_state: Optional ProgressState object for tracking progress
    
    Returns:
        Dictionary with processing results
    """
    
    # Convert chapters format from UI to processing format
    processed_chapters = []
    for chapter in chapters:
        # Extract page number from filename (e.g., "image00001.jpg" -> 1)
        start_filename = chapter["start"]
        end_filename = chapter["end"]
        
        # Use the filenames directly instead of extracting numbers
        processed_chapter = {
            "title": chapter["title"],
            "start": start_filename,
            "end": end_filename,
            "is_text": chapter["is_text"],
            "align": chapter.get("vertical_float", "Center").lower(),
            "aspect_ratio": chapter.get("aspect_ratio", desired_aspect_ratio),
            "horizontal_margin": chapter.get("horizontal_margin", 5),
            "vertical_margin": chapter.get("vertical_margin", 5),
        }
        processed_chapters.append(processed_chapter)
    
    # Create a list of text pages for processing
    pages = []
    page_is_first_or_last_in_chapter = []
    
    # Get all JPG files from input directory
    all_jpg_files = sorted([f for f in os.listdir(input_pages_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
    
    # Calculate total steps for progress tracking
    total_pages = 0
    for chapter in processed_chapters:
        if chapter["is_text"]:
            start_idx = all_jpg_files.index(chapter["start"]) if chapter["start"] in all_jpg_files else 0
            end_idx = all_jpg_files.index(chapter["end"]) if chapter["end"] in all_jpg_files else len(all_jpg_files) - 1
            chapter_pages = all_jpg_files[start_idx:end_idx + 1]
            total_pages += len(chapter_pages)
        else:
            start_idx = all_jpg_files.index(chapter["start"]) if chapter["start"] in all_jpg_files else 0
            end_idx = all_jpg_files.index(chapter["end"]) if chapter["end"] in all_jpg_files else len(all_jpg_files) - 1
            chapter_pages = all_jpg_files[start_idx:end_idx + 1]
            total_pages += len(chapter_pages)
    
    # Calculate total pages for analysis and processing
    text_analysis_pages = 0  # Pages that will be analyzed for text width
    total_processing_pages = 0  # All pages that will be processed
    
    for chapter in processed_chapters:
        if chapter["is_text"]:
            # Get the range of files between start and end
            start_idx = all_jpg_files.index(chapter["start"]) if chapter["start"] in all_jpg_files else 0
            end_idx = all_jpg_files.index(chapter["end"]) if chapter["end"] in all_jpg_files else len(all_jpg_files) - 1
            
            chapter_pages = all_jpg_files[start_idx:end_idx + 1]
            if chapter_pages:
                page_is_first_or_last_in_chapter.extend([chapter_pages[0], chapter_pages[-1]])
                pages.extend(chapter_pages)
                # Count pages that will be analyzed (excluding first/last of each chapter)
                text_analysis_pages += len([p for p in chapter_pages if p not in [chapter_pages[0], chapter_pages[-1]]])
        
        # Count all pages for processing
        start_idx = all_jpg_files.index(chapter["start"]) if chapter["start"] in all_jpg_files else 0
        end_idx = all_jpg_files.index(chapter["end"]) if chapter["end"] in all_jpg_files else len(all_jpg_files) - 1
        chapter_pages = all_jpg_files[start_idx:end_idx + 1]
        total_processing_pages += len(chapter_pages)
    
    df = pd.DataFrame()
    df["page"] = pages
    df = df.set_index("page")
    
    # Step 1: Initialize EasyOCR reader
    if progress_state:
        progress_state.add_step("ocr_init", "Initializing OCR reader...", "text")
    
    reader = easyocr.Reader(["en"], gpu=False)  # Use GPU if available
    
    if progress_state:
        progress_state.complete_step("ocr_init")
    
    # Step 2: Create output directories
    if progress_state:
        progress_state.add_step("create_dirs", "Creating output directories...", "text")
    
    # For text pages, rotate images based on the orientation of the lines of text on
    # the page, crop, add margins and resize to the minimum height and width.
    df["angle"] = np.nan
    df = df.assign(
        para_bboxes=pd.Series(dtype=object),
        all_text_bbox=pd.Series(dtype=object))
    
    rotated_subdir = os.path.join(output_dir, "2_rotate")
    mkdir(rotated_subdir)
    
    crop_subdir = os.path.join(output_dir, "3_crop")
    mkdir(crop_subdir)
    
    margin_subdir = os.path.join(output_dir, "4_margin")
    mkdir(margin_subdir)
    
    resize_subdir = os.path.join(output_dir, "5_resize")
    mkdir(resize_subdir)
    
    final_subdir = os.path.join(output_dir, "6_final")
    mkdir(final_subdir)
    
    if progress_state:
        progress_state.complete_step("create_dirs")
    
    # Step 3: Analyze text width (if there are text pages to analyze)
    if text_analysis_pages > 0 and progress_state:
        progress_state.add_step("text_analysis", f"Analyzing text width across {text_analysis_pages} pages...", "progress")
        progress_state.update_step("text_analysis", current=0, total=text_analysis_pages)
    
    min_width = float('inf')
    min_width_page = ""
    analyzed_count = 0
    
    for page in df.index:
        if page in page_is_first_or_last_in_chapter:
            continue
        
        # Quick text detection just for sizing
        result = reader.readtext(os.path.join(input_pages_dir, page))
        
        analyzed_count += 1
        if progress_state and text_analysis_pages > 0:
            progress_state.update_step("text_analysis", current=analyzed_count, total=text_analysis_pages)
        
        if len(result) == 0:
            continue
        
        # Get bounding boxes and calculate text width
        x_coords = [point[0] for result_item in result for point in result_item[0]]
        if len(x_coords) > 0:
            min_x, max_x = min(x_coords), max(x_coords)
            width = max_x - min_x
            if width < min_width:
                min_width = width
                min_width_page = page
    
    if progress_state and text_analysis_pages > 0:
        progress_state.complete_step("text_analysis")
    
    # If no text was found, use a default width based on typical page dimensions
    if min_width == float('inf'):
        min_width = 2000  # Default width for standardization
    
    final_height = int(min_width * desired_aspect_ratio)
    final_width = min_width
    
    # Step 4: Process all images
    if progress_state:
        progress_state.add_step("image_processing", f"Processing {total_processing_pages} images...", "progress")
        progress_state.update_step("image_processing", current=0, total=total_processing_pages)
    
    processed_count = 0
    total_chapters = len(processed_chapters)
    
    for chapter_idx, chapter in enumerate(processed_chapters):
        # Get chapter pages
        start_idx = all_jpg_files.index(chapter["start"]) if chapter["start"] in all_jpg_files else 0
        end_idx = all_jpg_files.index(chapter["end"]) if chapter["end"] in all_jpg_files else len(all_jpg_files) - 1
        chapter_pages = all_jpg_files[start_idx:end_idx + 1]
        
        # Get chapter-specific dimensions
        chapter_aspect_ratio = chapter.get("aspect_ratio", desired_aspect_ratio)
        chapter_final_height = int(final_width / chapter_aspect_ratio)
        
        if chapter["is_text"]:
            for page_idx, page in enumerate(chapter_pages):
                # Update progress
                processed_count += 1
                if progress_state:
                    progress_state.update_step("image_processing", current=processed_count, total=total_processing_pages)
                
                img = Image.open(os.path.join(input_pages_dir, page))
                
                # convert image to grayscale and create binary image using Otsu's threshold
                image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                _, binary_image = cv2.threshold(image_cv, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                
                # determine the skew angle using projection profile method
                vertical_angle = projection_profile_method(binary_image)
                if page in df.index:
                    df.at[page, "angle"] = vertical_angle
                
                # correct the image orientation
                rotated_img = img.rotate(
                    angle=vertical_angle,
                    expand=True,
                    fillcolor='white',
                    resample=Image.BICUBIC
                )
                
                # save the rotated image
                rotated_img.save(os.path.join(rotated_subdir, page))
                
                # find the paragraphs on the rotated image
                image_cv = cv2.cvtColor(np.array(rotated_img), cv2.IMREAD_GRAYSCALE)
                result = reader.readtext(image_cv, paragraph=True)
                
                # consolidate the paragraph bounding boxes into one all-containing bounding box
                x_min, x_max = rotated_img.width, 0
                y_min, y_max = rotated_img.height, 0
                bboxes = []
                for result_item in result:
                    bbox = result_item[0]
                    bboxes.append(bbox)
                    for pt in bbox:
                        if pt[0] < x_min:
                            x_min = pt[0]
                        if pt[0] > x_max:
                            x_max = pt[0]
                        if pt[1] < y_min:
                            y_min = pt[1]
                        if pt[1] > y_max:
                            y_max = pt[1]
                
                # Handle case where no text was detected - use full image bounds
                if x_min >= x_max or y_min >= y_max:
                    x_min, y_min = 0, 0
                    x_max, y_max = rotated_img.width, rotated_img.height
                
                if page in df.index:
                    df.at[page, "para_bboxes"] = bboxes
                    df.at[page, "all_text_bbox"] = [x_min, y_min, x_max, y_max]
                
                # crop the image
                cropped_img = rotated_img.crop((x_min, y_min, x_max, y_max))
                cropped_img.save(os.path.join(crop_subdir, page))
                
                # add margins to the image using chapter-specific margin settings
                h_margin_pct = chapter.get("horizontal_margin", 5) / 100
                v_margin_pct = chapter.get("vertical_margin", 5) / 100
                
                bordered_img = ImageOps.expand(
                    cropped_img,
                    border=(int(h_margin_pct * cropped_img.width / 2), int(v_margin_pct * cropped_img.height / 2)),
                    fill='white'
                )
                bordered_img.save(os.path.join(margin_subdir, page))
                
                # resize the image to the final width while maintaining the aspect ratio
                height = int(final_width / bordered_img.width * bordered_img.height)
                std_img = bordered_img.resize((final_width, height), resample=Image.LANCZOS)
                std_img.save(os.path.join(resize_subdir, page))
                
                if height < chapter_final_height:
                    # float the page vertically based on the chosen alignment strategy
                    if chapter["align"] == "top":
                        std_img = float_img_to_top(std_img, chapter_final_height)
                    
                    elif chapter["align"] == "center":
                        std_img = float_img_to_center(std_img, chapter_final_height)
                    
                    elif chapter["align"] == "bottom":
                        std_img = float_img_to_bottom(std_img, chapter_final_height)
                    
                    elif chapter["align"] == "chapter":
                        if page == chapter_pages[0]:
                            std_img = float_img_to_bottom(std_img, chapter_final_height)
                        elif page == chapter_pages[-1]:
                            std_img = float_img_to_top(std_img, chapter_final_height)
                        else:
                            std_img = float_img_to_center(std_img, chapter_final_height)
                
                else:
                    std_img = std_img.resize((final_width, chapter_final_height), resample=Image.LANCZOS)
                
                std_img.save(os.path.join(final_subdir, page))
        
        else:
            # For pages that are largely pictorial, just resize them
            for page_idx, page in enumerate(chapter_pages):
                # Update progress
                processed_count += 1
                if progress_state:
                    progress_state.update_step("image_processing", current=processed_count, total=total_processing_pages)
                
                with Image.open(os.path.join(input_pages_dir, page)) as img:
                    std_img = img.resize((final_width, chapter_final_height), resample=Image.LANCZOS)
                    std_img.save(os.path.join(final_subdir, page))
    
    # Complete image processing step
    if progress_state:
        progress_state.complete_step("image_processing")
    
    # Save the dataframe
    df.to_pickle(os.path.join(output_dir, "df.pkl"))
    
    # Final step - mark as complete with total elapsed time
    if progress_state:
        progress_state.set_complete("Processing complete!")
    
    return {
        "success": True,
        "output_dir": output_dir,
        "final_width": final_width,
        "final_height": final_height,
        "pages_processed": len(pages)
    }
