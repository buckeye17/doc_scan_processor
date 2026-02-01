"""
Export page - allows users to configure export settings for the final PDF document.
"""
import dash
from dash import callback, Input, Output, State, no_update
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import os
from PIL import Image
from components import get_pages_list, data_dir

dash.register_page(__name__, path="/export", title="Export")


def get_final_images():
    """Get sorted list of final processed images."""
    final_dir = os.path.join(data_dir, "6_final")
    if not os.path.exists(final_dir):
        return []

    pages = get_pages_list()
    # Filter out empty string and get only pages that have final images
    final_images = []
    for page in pages:
        if page:  # Skip empty string
            img_path = os.path.join(final_dir, page)
            if os.path.exists(img_path):
                final_images.append(img_path)

    return sorted(final_images)


def create_pdf_from_images(image_paths, output_path):
    """
    Create a PDF file from a list of image paths.

    Args:
        image_paths: List of paths to JPG images
        output_path: Path where the PDF will be saved

    Returns:
        dict with 'success' boolean and 'message' or 'error' string
    """
    if not image_paths:
        return {"success": False, "error": "No images found to export"}

    try:
        # Open all images and convert to RGB (required for PDF)
        images = []
        for img_path in image_paths:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)

        if not images:
            return {"success": False, "error": "Failed to load any images"}

        # Save first image as PDF with remaining images appended
        first_image = images[0]
        remaining_images = images[1:] if len(images) > 1 else []

        first_image.save(
            output_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=remaining_images
        )

        return {
            "success": True,
            "message": f"Successfully exported {len(images)} pages to PDF"
        }

    except Exception as e:
        return {"success": False, "error": f"Error creating PDF: {str(e)}"}

def validate_export_folder(folder_path):
    """Validate the export folder path."""
    if not folder_path or not folder_path.strip():
        return {"valid": False, "error": "Please enter a folder path"}

    folder_path = folder_path.strip()

    if not os.path.exists(folder_path):
        return {"valid": False, "error": "Folder does not exist"}

    if not os.path.isdir(folder_path):
        return {"valid": False, "error": "Path is not a folder"}

    try:
        # Check if we have write permission
        test_file = os.path.join(folder_path, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        return {"valid": True, "error": None}
    except PermissionError:
        return {"valid": False, "error": "No write permission for this folder"}
    except Exception as e:
        return {"valid": False, "error": f"Error accessing folder: {str(e)}"}

def validate_filename(filename):
    """Validate the PDF filename."""
    if not filename or not filename.strip():
        return {"valid": False, "error": "Please enter a filename"}

    filename = filename.strip()

    # Check for invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        if char in filename:
            return {"valid": False, "error": f"Filename cannot contain '{char}'"}

    # Ensure .pdf extension
    if not filename.lower().endswith('.pdf'):
        filename = filename + '.pdf'

    return {"valid": True, "error": None, "filename": filename}

layout = dmc.Container([
    dmc.Stack([
        dmc.Title("Export Document", order=1, mb="lg"),

        dmc.Paper([
            dmc.Title("PDF Export Settings", order=3, mb="md"),

            dmc.Stack([
                dmc.Text(
                    "Configure the output settings for your final PDF document.",
                    size="sm",
                    c="dimmed"
                ),

                dmc.TextInput(
                    label="PDF Filename",
                    placeholder="my_document.pdf",
                    description="Enter the name for your exported PDF file",
                    id="export-filename-input",
                    rightSection=dmc.Text(".pdf", size="sm", c="dimmed"),
                ),

                dmc.TextInput(
                    label="Export Folder",
                    placeholder="C:\\Users\\YourName\\Documents",
                    description="Select the folder where the PDF will be saved",
                    id="export-folder-input",
                    rightSection=dmc.ActionIcon(
                        DashIconify(icon="tabler:folder-open", width=16),
                        variant="subtle",
                        id="export-browse-button"
                    ),
                ),

                # Validation message
                dmc.Box(
                    id="export-validation-message",
                    children="",
                ),

                dmc.Group([
                    dmc.Button(
                        "Export PDF",
                        leftSection=DashIconify(icon="fluent:document-pdf-24-filled", width=20),
                        id="export-button",
                        size="lg",
                        disabled=True,
                    ),
                ], mt="md"),

                # Export result message
                dmc.Box(
                    id="export-result-message",
                    children="",
                ),

            ], gap="md")
        ], p="lg", withBorder=True, shadow="sm"),

    ], gap="lg")
], size="md", py="xl")


@callback(
    [Output("export-validation-message", "children"),
     Output("export-button", "disabled"),
     Output("export-filename-input", "error"),
     Output("export-folder-input", "error")],
    [Input("export-filename-input", "value"),
     Input("export-folder-input", "value")]
)
def validate_export_settings(filename, folder):
    """Validate export settings and update UI."""
    filename_result = validate_filename(filename)
    folder_result = validate_export_folder(folder)

    filename_error = filename_result.get("error", "") if not filename_result["valid"] else ""
    folder_error = folder_result.get("error", "") if not folder_result["valid"] else ""

    # Both must be valid to enable the export button
    is_valid = filename_result["valid"] and folder_result["valid"]

    if is_valid:
        full_path = os.path.join(folder.strip(), filename_result["filename"])
        message = dmc.Alert(
            f"Ready to export to: {full_path}",
            color="green",
            icon=DashIconify(icon="fluent:checkmark-circle-24-filled"),
        )
    elif not filename and not folder:
        message = dmc.Alert(
            "Enter a filename and folder to export your document.",
            color="blue",
            icon=DashIconify(icon="fluent:info-24-filled"),
        )
    else:
        message = ""

    return message, not is_valid, filename_error, folder_error


@callback(
    Output("export-result-message", "children"),
    Input("export-button", "n_clicks"),
    [State("export-filename-input", "value"),
     State("export-folder-input", "value")],
    prevent_initial_call=True
)
def export_pdf(n_clicks, filename, folder):
    """Handle the export button click and create the PDF."""
    if not n_clicks:
        return no_update

    # Validate inputs again
    filename_result = validate_filename(filename)
    folder_result = validate_export_folder(folder)

    if not filename_result["valid"] or not folder_result["valid"]:
        return dmc.Alert(
            "Invalid filename or folder. Please check your inputs.",
            color="red",
            icon=DashIconify(icon="fluent:error-circle-24-filled"),
        )

    # Get the final images
    image_paths = get_final_images()

    if not image_paths:
        return dmc.Alert(
            "No processed images found. Please run the image processor first.",
            color="red",
            icon=DashIconify(icon="fluent:error-circle-24-filled"),
        )

    # Build full output path
    output_path = os.path.join(folder.strip(), filename_result["filename"])

    # Check if file already exists
    if os.path.exists(output_path):
        # Overwrite existing file
        pass

    # Create the PDF
    result = create_pdf_from_images(image_paths, output_path)

    if result["success"]:
        return dmc.Alert(
            [
                dmc.Text(result["message"], fw=500),
                dmc.Text(f"Saved to: {output_path}", size="sm", c="dimmed"),
            ],
            color="green",
            icon=DashIconify(icon="fluent:checkmark-circle-24-filled"),
            title="Export Complete",
        )
    else:
        return dmc.Alert(
            result["error"],
            color="red",
            icon=DashIconify(icon="fluent:error-circle-24-filled"),
            title="Export Failed",
        )
