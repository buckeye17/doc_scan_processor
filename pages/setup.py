"""
Setup page - allows users to configure the image directory and other application settings.
"""
import dash
from dash import callback, dcc, html, Input, Output, State, no_update
from dash_iconify import DashIconify
import dash_mantine_components as dmc
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

def load_config():
    """Load configuration from file, return defaults if file doesn't exist."""
    config_path = get_config_file_path()
    default_config = {
        "images_directory": "",
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
            dmc.Title("Book Setup", order=1, mb="lg"),
            
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
            
            # Section Definition Section
            dmc.Paper([
                dmc.Title("Section Definition", order=3, mb="md"),
            ], p="lg", withBorder=True, shadow="sm", mt="xl"),
        ], gap="md"),
        
        # Hidden stores for state management
        dcc.Store(id="setup-config-store", data=current_config)
        
    ], size="md", py="lg")

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