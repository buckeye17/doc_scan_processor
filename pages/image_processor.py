"""
Setup page - allows users to configure the image directory and other application settings.
"""
import dash
from dash import callback, dcc, html, Input, Output, State, no_update
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import dash_ag_grid as dag
import os
import json
from pathlib import Path

# Register this page
dash.register_page(__name__, path="/setup", title="Setup")

def get_config_file_path():
    """Get the path to the configuration file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "config.json")
    return config_path

def get_jpg_files_from_directory(directory_path):
    """Get sorted list of JPG files from directory."""
    if not directory_path or not os.path.exists(directory_path):
        return []
    try:
        jpg_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        return sorted(jpg_files)
    except Exception:
        return []

def load_config():
    """Load configuration from file, return defaults if file doesn't exist."""
    config_path = get_config_file_path()
    
    # Try to get existing images directory to set default filenames
    existing_config = {}
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                existing_config = json.load(f)
    except Exception:
        pass
    
    images_dir = existing_config.get("images_directory", "")
    jpg_files = get_jpg_files_from_directory(images_dir)
    
    # Set default start/end to first and last filenames if available
    default_start = jpg_files[0] if jpg_files else "image00001.jpg"
    default_end = jpg_files[-1] if jpg_files else "image99999.jpg"
    
    # Default chapters - start with one row encompassing all page images
    default_chapters = [
        {"id": 1, "title": "All Pages", "start": default_start, "end": default_end, "is_text": True, "aspect_ratio": 9/5.5, "vertical_float": "Center", "horizontal_margin": 5, "vertical_margin": 5},
    ]
    
    default_config = {
        "images_directory": "",
        "app_name": "Scan Post Processing",
        "default_aspect_ratio": 9/5.5,
        "default_vertical_float": "Center",
        "default_horizontal_margin": 5,
        "default_vertical_margin": 5,
        "chapters": default_chapters
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_config = {**default_config, **config}
                # Ensure chapters have all required fields
                if "chapters" not in merged_config or not merged_config["chapters"]:
                    merged_config["chapters"] = default_chapters
                return merged_config
    except Exception:
        pass
    
    return default_config

def save_config(config):
    """Save configuration to file."""
    config_path = get_config_file_path()
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False

def validate_directory_path(directory_path):
    """Validate directory path and return validation result."""
    if not directory_path or not directory_path.strip():
        return {"valid": False, "error": "Please enter a directory path", "jpg_count": 0}
    
    directory_path = directory_path.strip()
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        return {"valid": False, "error": "Directory does not exist", "jpg_count": 0}
    
    if not os.path.isdir(directory_path):
        return {"valid": False, "error": "Path is not a directory", "jpg_count": 0}
    
    # Check for JPG files
    try:
        all_files = os.listdir(directory_path)
        jpg_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg'))]
        
        if not jpg_files:
            return {"valid": False, "error": "No JPG files found in directory", "jpg_count": 0}
        
        return {"valid": True, "error": None, "jpg_count": len(jpg_files), "files": jpg_files}
        
    except PermissionError:
        return {"valid": False, "error": "Permission denied accessing directory", "jpg_count": 0}
    except Exception as e:
        return {"valid": False, "error": f"Error accessing directory: {str(e)}", "jpg_count": 0}

# Load current configuration
current_config = load_config()

def layout():
    return dmc.Container([
        dmc.Stack([
            dmc.Title("Image Processor", order=1, mb="lg"),
            
            # Page Import Section
            dmc.Paper([
                dmc.Title("Page Import", order=3, mb="md"),
                
                dmc.Stack([
                    dmc.Text(
                        "Select the directory containing your scanned document images (JPG files).",
                        size="sm",
                        c="dimmed"
                    ),
                    
                    dmc.TextInput(
                        label="Images Directory Path",
                        placeholder="C:\\Users\\YourName\\Documents\\Images",
                        value=current_config.get("images_directory", ""),
                        id="setup-directory-input",
                        rightSection=dmc.ActionIcon(
                            DashIconify(icon="tabler:folder-open", width=16),
                            variant="subtle",
                            id="setup-browse-button"
                        ),
                        debounce=500,  # Add debounce for automatic validation
                        error="",  # Will be updated by callback
                    ),
                    
                    # JPG file count message
                    dmc.Box(
                        id="setup-jpg-count-message",
                        children="",  # Will be populated by callback
                    )
                ], gap="sm")
            ], p="lg", withBorder=True, shadow="sm"),
            
            # Default Adjustments Section
            dmc.Paper([
                dmc.Title("Default Adjustments", order=3, mb="md"),
                
                dmc.Stack([
                    dmc.Text(
                        "Configure default settings for page processing and layout. These values will be applied to new chapters in the Section Definition table.",
                        size="sm",
                        c="dimmed",
                        mb="xs"
                    ),
                    
                    dmc.SimpleGrid(
                        cols=2,
                        spacing="md",
                        children=[
                            dmc.NumberInput(
                                label="Page Aspect Ratio",
                                description="Width / Height ratio of final pages (e.g., 0.75 for portrait)",
                                placeholder="0.75",
                                value=current_config.get("default_aspect_ratio", 11/8.5),
                                min=0.1,
                                max=10,
                                step=0.01,
                                decimalScale=2,
                                id="setup-aspect-ratio-input",
                            ),
                            
                            dmc.Select(
                                label="Vertical Float",
                                description="Vertical alignment of content when it doesn't span full page height",
                                placeholder="Select alignment",
                                value=current_config.get("default_vertical_float", "Center"),
                                data=[
                                    {"value": "Top", "label": "Top"},
                                    {"value": "Center", "label": "Center"},
                                    {"value": "Bottom", "label": "Bottom"}
                                ],
                                id="setup-vertical-float-select",
                            ),
                            
                            dmc.NumberInput(
                                label="Left & Right Margin Size",
                                description="Total horizontal margin as percentage of page width (split equally)",
                                placeholder="5",
                                value=current_config.get("default_horizontal_margin", 5),
                                min=0,
                                max=50,
                                step=0.5,
                                decimalScale=1,
                                suffix="%",
                                id="setup-horizontal-margin-input",
                            ),
                            
                            dmc.NumberInput(
                                label="Top & Bottom Margin Size",
                                description="Total vertical margin as percentage of page height (split equally)",
                                placeholder="5",
                                value=current_config.get("default_vertical_margin", 5),
                                min=0,
                                max=50,
                                step=0.5,
                                decimalScale=1,
                                suffix="%",
                                id="setup-vertical-margin-input",
                            ),
                        ]
                    )
                ], gap="md")
            ], p="lg", withBorder=True, shadow="sm", mt="xl"),
            
            # Section Definition Section
            dmc.Paper([
                dmc.Title("Section Definition", order=3, mb="md"),
                
                dmc.Stack([
                    dmc.Text(
                        """Define the chapters/sections of your document. Each row represents a chapter that 
                        spans a range of page images.  Clicking once on a cell will select that row for 
                        deletion.  Double clicking on a cell will allow you to edit its value.""",
                        size="sm",
                        c="dimmed",
                        mb="xs"
                    ),
                    
                    # AG Grid for chapter definitions
                    dag.AgGrid(
                        id="setup-chapters-grid",
                        rowData=current_config.get("chapters", []),
                        columnDefs=[
                            {
                                "field": "id",
                                "headerName": "ID",
                                "editable": True,
                                "width": 80,
                                "hide": True,  # Hide the ID column from display
                            },
                            {
                                "headerName": "",
                                "checkboxSelection": True,
                                "headerCheckboxSelection": True,
                                "width": 50,
                                "pinned": "left",
                                "lockPosition": True,
                                "suppressMenu": True,
                                "sortable": False,
                                "filter": False,
                            },
                            {
                                "field": "title",
                                "headerName": "Chapter Title",
                                "editable": True,
                                "width": 200,
                            },
                            {
                                "field": "start",
                                "headerName": "Start Page",
                                "editable": True,
                                "width": 150,
                                "cellEditor": "agSelectCellEditor",
                                "cellEditorParams": {
                                    "values": get_jpg_files_from_directory(current_config.get("images_directory", ""))
                                },
                            },
                            {
                                "field": "end",
                                "headerName": "End Page",
                                "editable": True,
                                "width": 150,
                                "cellEditor": "agSelectCellEditor",
                                "cellEditorParams": {
                                    "values": get_jpg_files_from_directory(current_config.get("images_directory", ""))
                                },
                            },
                            {
                                "field": "is_text",
                                "headerName": "Has Text",
                                "editable": True,
                                "cellDataType": "boolean",
                                "width": 110,
                            },
                            {
                                "field": "aspect_ratio",
                                "headerName": "Aspect Ratio",
                                "editable": True,
                                "type": "numericColumn",
                                "width": 130,
                                "valueFormatter": {"function": "d3.format('.2f')(params.value)"},
                            },
                            {
                                "field": "vertical_float",
                                "headerName": "Vertical Float",
                                "editable": True,
                                "cellEditor": "agSelectCellEditor",
                                "cellEditorParams": {
                                    "values": ["Top", "Center", "Bottom"]
                                },
                                "width": 140,
                            },
                            {
                                "field": "horizontal_margin",
                                "headerName": "H. Margin (%)",
                                "editable": True,
                                "type": "numericColumn",
                                "width": 140,
                                "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
                            },
                            {
                                "field": "vertical_margin",
                                "headerName": "V. Margin (%)",
                                "editable": True,
                                "type": "numericColumn",
                                "width": 140,
                                "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
                            },
                        ],
                        defaultColDef={
                            "resizable": True,
                            "sortable": True,
                            "filter": True,
                        },
                        dashGridOptions={
                            "rowSelection": "multiple",
                            "animateRows": True,
                            "pagination": False,
                            "domLayout": "autoHeight",
                        },
                        style={"height": None},
                        className="ag-theme-alpine",
                    ),
                    
                    # Action buttons for the grid
                    dmc.Group([
                        dmc.Button(
                            "Add Chapter",
                            id="setup-add-chapter-button",
                            leftSection=DashIconify(icon="tabler:plus", width=16),
                            variant="light",
                            color="blue",
                        ),
                        dmc.Button(
                            "Delete Selected",
                            id="setup-delete-chapters-button",
                            leftSection=DashIconify(icon="tabler:trash", width=16),
                            variant="light",
                            color="red",
                        ),
                        dmc.Button(
                            "Apply Default Adjustments",
                            id="setup-apply-defaults-button",
                            leftSection=DashIconify(icon="tabler:refresh", width=16),
                            variant="light",
                            color="green",
                        ),
                    ], gap="sm", mt="md"),
                ], gap="md")
            ], p="lg", withBorder=True, shadow="sm", mt="xl"),
        ], gap="md"),
        
        # Hidden stores for state management
        dcc.Store(id="setup-config-store", data=current_config)
        
    ], size="xl", py="lg")

# Layout will be set dynamically via callback

@callback(
    [Output("setup-directory-input", "error"),
     Output("setup-directory-input", "style"),
     Output("setup-jpg-count-message", "children"),
     Output("setup-config-store", "data"),
     Output("shared-page-state", "data", allow_duplicate=True)],
    [Input("setup-directory-input", "value")],
    [State("setup-config-store", "data")],
    prevent_initial_call='initial_duplicate'
)
def validate_and_save_directory(directory_path, current_config):
    """Automatically validate directory and save configuration when valid."""
    if not directory_path:
        return (
            "",  # No error for empty input
            {"width": "100%"},  # Default style
            "",  # No message to show
            current_config,
            ""  # No page selected
        )
    
    validation_result = validate_directory_path(directory_path)
    
    if not validation_result["valid"]:
        return (
            validation_result["error"],  # Show error in TextInput
            {"width": "100%"},  # Default style
            "",  # No success message to show when invalid
            current_config,
            ""  # No page selected
        )
    
    # Directory is valid, create JPG count message
    jpg_count = validation_result["jpg_count"]
    
    # Create success message with JPG count as standard text
    message = dmc.Text(
        f"Found {jpg_count} JPG files ready for processing.",
        size="sm",
        c="dark",
        style={"marginTop": "8px"}
    )
    
    # Automatically save valid configuration
    new_config = current_config.copy()
    new_config["images_directory"] = directory_path.strip()
    save_config(new_config)  # Save automatically when valid
    
    # Set first JPG file as the selected page for viewer/editor
    jpg_files = validation_result["files"]
    first_jpg = sorted(jpg_files)[0] if jpg_files else ""
    
    return (
        "",  # No error
        {"width": "100%", "borderColor": "#40c057", "borderWidth": "2px"},  # Green border for valid input
        message,  # JPG count success message as standard text
        new_config,
        first_jpg  # Set first JPG as selected page
    )




# Initialize shared page state on app load (only once)
@callback(
    Output("shared-page-state", "data", allow_duplicate=True),
    [Input("url", "pathname")],
    prevent_initial_call=True
)
def initialize_shared_page_state_on_app_load(pathname):
    """Initialize shared page state with first JPG file from current config when app loads."""
    # Only run this initialization once when the app first loads (any page)
    if pathname in ["/", "/editor", "/setup"]:
        config = load_config()
        directory_path = config.get("images_directory", "")
        
        if directory_path:
            validation_result = validate_directory_path(directory_path)
            if validation_result["valid"] and validation_result["files"]:
                first_jpg = sorted(validation_result["files"])[0]
                return first_jpg
    
    return no_update

@callback(
    Output("setup-directory-input", "value", allow_duplicate=True),
    [Input("setup-browse-button", "n_clicks")],
    [State("setup-directory-input", "value")],
    prevent_initial_call=True
)
def browse_directory(n_clicks, current_value):
    """Handle directory browsing (placeholder - would need additional implementation for file dialog)."""
    if not n_clicks:
        return current_value or ""
    
    # Note: In a real implementation, you might want to use a file dialog
    # For now, this is a placeholder that could be enhanced with additional JavaScript
    return current_value or ""


# Callback to add a new chapter row
@callback(
    [Output("setup-chapters-grid", "rowData", allow_duplicate=True),
     Output("setup-config-store", "data", allow_duplicate=True)],
    [Input("setup-add-chapter-button", "n_clicks")],
    [State("setup-chapters-grid", "rowData"),
     State("setup-config-store", "data"),
     State("setup-aspect-ratio-input", "value"),
     State("setup-vertical-float-select", "value"),
     State("setup-horizontal-margin-input", "value"),
     State("setup-vertical-margin-input", "value")],
    prevent_initial_call=True
)
def add_chapter(n_clicks, current_rows, current_config, aspect_ratio, vertical_float, h_margin, v_margin):
    """Add a new chapter row to the grid with default values from Default Adjustments."""
    if not n_clicks:
        return no_update, no_update
    
    # Get the next available ID
    max_id = max([row.get("id", 0) for row in current_rows], default=0)
    new_id = max_id + 1
    
    # Get the images directory to determine available filenames
    images_dir = current_config.get("images_directory", "")
    jpg_files = get_jpg_files_from_directory(images_dir)
    
    # Determine the next start page filename
    # Try to find the next filename after the last chapter's end
    new_start = "image00001.jpg"
    new_end = "image00001.jpg"
    
    if current_rows and jpg_files:
        last_end = current_rows[-1].get("end", "")
        try:
            # Find the index of the last end filename
            if last_end in jpg_files:
                last_index = jpg_files.index(last_end)
                if last_index + 1 < len(jpg_files):
                    new_start = jpg_files[last_index + 1]
                    new_end = jpg_files[last_index + 1]
                else:
                    # Use last file if we're at the end
                    new_start = jpg_files[-1]
                    new_end = jpg_files[-1]
            else:
                # If last_end not found, use first available file
                new_start = jpg_files[0] if jpg_files else "image00001.jpg"
                new_end = jpg_files[0] if jpg_files else "image00001.jpg"
        except Exception:
            new_start = jpg_files[0] if jpg_files else "image00001.jpg"
            new_end = jpg_files[0] if jpg_files else "image00001.jpg"
    elif jpg_files:
        new_start = jpg_files[0]
        new_end = jpg_files[0]
    
    # Create new chapter with default adjustments
    new_chapter = {
        "id": new_id,
        "title": f"New Chapter {new_id}",
        "start": new_start,
        "end": new_end,
        "is_text": True,
        "aspect_ratio": aspect_ratio or 9/5.5,
        "vertical_float": vertical_float or "Center",
        "horizontal_margin": h_margin or 5,
        "vertical_margin": v_margin or 5,
    }
    
    # Add to rows
    updated_rows = current_rows + [new_chapter]
    
    # Update config
    updated_config = current_config.copy()
    updated_config["chapters"] = updated_rows
    save_config(updated_config)
    
    return updated_rows, updated_config


# Callback to delete selected chapter rows
@callback(
    [Output("setup-chapters-grid", "rowData", allow_duplicate=True),
     Output("setup-config-store", "data", allow_duplicate=True)],
    [Input("setup-delete-chapters-button", "n_clicks")],
    [State("setup-chapters-grid", "selectedRows"),
     State("setup-chapters-grid", "rowData"),
     State("setup-config-store", "data")],
    prevent_initial_call=True
)
def delete_chapters(n_clicks, selected_rows, current_rows, current_config):
    """Delete selected chapter rows from the grid."""
    if not n_clicks or not selected_rows:
        return no_update, no_update
    
    # Get IDs of selected rows
    selected_ids = [row.get("id") for row in selected_rows]
    
    # Filter out selected rows
    updated_rows = [row for row in current_rows if row.get("id") not in selected_ids]
    
    # Update config
    updated_config = current_config.copy()
    updated_config["chapters"] = updated_rows
    save_config(updated_config)
    
    return updated_rows, updated_config


# Callback to apply default adjustments to all chapters
@callback(
    [Output("setup-chapters-grid", "rowData", allow_duplicate=True),
     Output("setup-config-store", "data", allow_duplicate=True)],
    [Input("setup-apply-defaults-button", "n_clicks")],
    [State("setup-chapters-grid", "rowData"),
     State("setup-config-store", "data"),
     State("setup-aspect-ratio-input", "value"),
     State("setup-vertical-float-select", "value"),
     State("setup-horizontal-margin-input", "value"),
     State("setup-vertical-margin-input", "value")],
    prevent_initial_call=True
)
def apply_default_adjustments(n_clicks, current_rows, current_config, aspect_ratio, vertical_float, h_margin, v_margin):
    """Apply default adjustments to all chapter rows."""
    if not n_clicks:
        return no_update, no_update
    
    # Update all rows with default adjustments
    updated_rows = []
    for row in current_rows:
        updated_row = row.copy()
        updated_row["aspect_ratio"] = aspect_ratio or 9/5.5
        updated_row["vertical_float"] = vertical_float or "Center"
        updated_row["horizontal_margin"] = h_margin or 5
        updated_row["vertical_margin"] = v_margin or 5
        updated_rows.append(updated_row)
    
    # Update config
    updated_config = current_config.copy()
    updated_config["chapters"] = updated_rows
    updated_config["default_aspect_ratio"] = aspect_ratio or 9/5.5
    updated_config["default_vertical_float"] = vertical_float or "Center"
    updated_config["default_horizontal_margin"] = h_margin or 5
    updated_config["default_vertical_margin"] = v_margin or 5
    save_config(updated_config)
    
    return updated_rows, updated_config


# Callback to save grid edits to config
@callback(
    Output("setup-config-store", "data", allow_duplicate=True),
    [Input("setup-chapters-grid", "cellValueChanged")],
    [State("setup-chapters-grid", "rowData"),
     State("setup-config-store", "data")],
    prevent_initial_call=True
)
def save_grid_edits(cell_changed, current_rows, current_config):
    """Save grid edits to configuration."""
    if not cell_changed:
        return no_update
    
    # Update config with current grid data
    updated_config = current_config.copy()
    updated_config["chapters"] = current_rows
    save_config(updated_config)
    
    return updated_config


# Callback to update chapters grid when directory changes
@callback(
    [Output("setup-chapters-grid", "rowData", allow_duplicate=True),
     Output("setup-chapters-grid", "columnDefs", allow_duplicate=True),
     Output("setup-config-store", "data", allow_duplicate=True)],
    [Input("setup-directory-input", "value")],
    [State("setup-config-store", "data")],
    prevent_initial_call=True
)
def update_chapters_on_directory_change(directory_path, current_config):
    """Update chapters grid with first and last filenames when directory is set, and update dropdown options."""
    if not directory_path:
        return no_update, no_update, no_update
    
    validation_result = validate_directory_path(directory_path)
    if not validation_result["valid"]:
        return no_update, no_update, no_update
    
    jpg_files = validation_result.get("files", [])
    if not jpg_files:
        return no_update, no_update, no_update
    
    sorted_files = sorted(jpg_files)
    first_file = sorted_files[0]
    last_file = sorted_files[-1]
    
    # Update column definitions with new dropdown values
    updated_column_defs = [
        {
            "field": "id",
            "headerName": "ID",
            "editable": True,
            "width": 80,
            "hide": True,
        },
        {
            "headerName": "",
            "checkboxSelection": True,
            "headerCheckboxSelection": True,
            "width": 50,
            "pinned": "left",
            "lockPosition": True,
            "suppressMenu": True,
            "sortable": False,
            "filter": False,
        },
        {
            "field": "title",
            "headerName": "Chapter Title",
            "editable": True,
            "width": 200,
        },
        {
            "field": "start",
            "headerName": "Start Page",
            "editable": True,
            "width": 150,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {
                "values": sorted_files
            },
        },
        {
            "field": "end",
            "headerName": "End Page",
            "editable": True,
            "width": 150,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {
                "values": sorted_files
            },
        },
        {
            "field": "is_text",
            "headerName": "Has Text",
            "editable": True,
            "cellDataType": "boolean",
            "width": 110,
        },
        {
            "field": "aspect_ratio",
            "headerName": "Aspect Ratio",
            "editable": True,
            "type": "numericColumn",
            "width": 130,
            "valueFormatter": {"function": "d3.format('.2f')(params.value)"},
        },
        {
            "field": "vertical_float",
            "headerName": "Vertical Float",
            "editable": True,
            "cellEditor": "agSelectCellEditor",
            "cellEditorParams": {
                "values": ["Top", "Center", "Bottom"]
            },
            "width": 140,
        },
        {
            "field": "horizontal_margin",
            "headerName": "H. Margin (%)",
            "editable": True,
            "type": "numericColumn",
            "width": 140,
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
        },
        {
            "field": "vertical_margin",
            "headerName": "V. Margin (%)",
            "editable": True,
            "type": "numericColumn",
            "width": 140,
            "valueFormatter": {"function": "d3.format('.1f')(params.value)"},
        },
    ]
    
    # Only update if chapters is empty or has default placeholder values
    current_chapters = current_config.get("chapters", [])
    
    # Check if we need to update (if there's only one chapter with placeholder values)
    if len(current_chapters) == 1:
        chapter = current_chapters[0]
        # Update if start/end are numeric or placeholder filenames
        if (isinstance(chapter.get("start"), (int, float)) or 
            chapter.get("start") in ["image00001.jpg", "image99999.jpg"] or
            isinstance(chapter.get("end"), (int, float)) or
            chapter.get("end") in ["image00001.jpg", "image99999.jpg"]):
            
            updated_chapter = chapter.copy()
            updated_chapter["start"] = first_file
            updated_chapter["end"] = last_file
            
            updated_config = current_config.copy()
            updated_config["chapters"] = [updated_chapter]
            save_config(updated_config)
            
            return [updated_chapter], updated_column_defs, updated_config
    
    return no_update, updated_column_defs, no_update