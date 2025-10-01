"""
Manual Result Editor page - allows manual override of automated processing on a per-page basis.
"""
import dash
from dash import callback, ctx, dcc, html, Input, Output, State
from dash_iconify import DashIconify
import dash_mantine_components as dmc
import numpy as np
import plotly.express as px

from components import create_empty_figure, create_page_select, create_navigation_buttons, \
    create_rotated_page_figure_with_manual_crop, page_viewer_height

# Register this page
dash.register_page(__name__, path="/editor", title="Manual Result Editor")

controls = dmc.Stack([
    dmc.Title("Manual Editor", order=4, mb="md"),
    dmc.Group([
        create_page_select("editor"),
        create_navigation_buttons("editor"),
    ]),
    dmc.Divider(my="md"),
    dmc.Title("Manual Adjustments", order=5, mb="md"),
    dmc.Switch(
        label="Preview Mode",
        description="Show processed preview instead of original with bounding box",
        checked=False,
        mt=0,
        mb="md",
        id="editor-preview-switch",
    ),
    dmc.NumberInput(
        label="Step 1: Rotation Angle (°)",
        value=0.0,
        decimalScale=1,
        min=-180,
        step=0.1,
        max=180,
        w=200,
        id="editor-rotation-angle",
    ),
    dmc.Stack([
        dmc.Text("Step 2: Crop", size="sm", fw=500),
        dmc.Switch(
            label="Manual crop mode",
            description="Draw your own bounding box (only available in original mode)",
            checked=False,
            mt=0,
            id="editor-crop-switch",
        ),
    ], gap=2, mt=10),
    dmc.NumberInput(
        label="Step 3: Top-Bottom Margin",
        value=0.0,
        suffix="%",
        decimalScale=1,
        min=0,
        step=1,
        max=99,
        w=200,
        mt=10,
        id="editor-top-bottom-margin",
    ),
    dmc.NumberInput(
        label="Step 3: Left-Right Margin",
        value=0.0,
        suffix="%",
        decimalScale=1,
        min=0,
        step=1,
        max=99,
        w=200,
        id="editor-left-right-margin",
    ),
    dmc.Select(
        label="Step 4: Float Position",
        value="center",
        data=["top", "center", "bottom"],
        w=200,
        mt=10,
        id="editor-float",
    ),
    dmc.Button(
        "Reset to Auto",
        mt=20,
        w=200,
        variant="outline",
        color="red",
        id="editor-reset-btn",
        leftSection=DashIconify(icon="material-symbols:refresh", width=16),
    ),
    # Store components for state management
    dmc.Box([
        dcc.Store(id="editor-manual-adjustments", data={}),
        dcc.Store(id="editor-manual-bbox", data={})
    ], style={"display": "none"})
])

editor = dmc.Box([
    dmc.LoadingOverlay(
        visible=False,
        id="editor-loading-overlay",
        overlayProps={"radius": "sm", "blur": 2},
        zIndex=10,
    ),
    dcc.Graph(
        figure=create_empty_figure(), 
        id="editor-page-viewer",
        style={"height": "100%", "width": "100%"}
    )
], style={"height": "100%", "width": "100%"})

info = dmc.Stack([
    dmc.Title("Processing Status", order=4, mb="md"),
    dmc.Box(id="editor-info-container", style={"minHeight": "200px"}),
    dmc.Divider(my="md"),
    dmc.Title("Help", order=5, mb="sm"),
    dmc.Text([
        "Use the controls on the left to manually adjust the processing steps. ",
        "Changes will be applied to the currently selected page only."
    ], size="sm", c="dimmed")
])

layout = dmc.Box([
    dmc.NotificationContainer(id="editor-notification-container"),
    dmc.Flex([
        # Left column - Controls (fixed width)
        dmc.Box(
            controls,
            style={
                "width": "280px",
                "minWidth": "280px",
                "height": "calc(100vh - 60px)",  # Account for header height
                "overflow": "auto",
                "backgroundColor": "#f8f9fa",
                "borderRight": "1px solid #dee2e6",
                "padding": "20px"
            }
        ),
        # Center column - Figure viewer (flexible)
        dmc.Box(
            editor,
            style={
                "flex": "1",
                "height": "calc(100vh - 60px)",  # Account for header height
                "overflow": "auto",
                "padding": "20px",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center"
            }
        ),
        # Right column - Information panel (fixed width)
        dmc.Box(
            info,
            style={
                "width": "300px",
                "minWidth": "300px",
                "height": "calc(100vh - 60px)",  # Account for header height
                "overflow": "auto",
                "backgroundColor": "#f8f9fa",
                "borderLeft": "1px solid #dee2e6",
                "padding": "20px"
            }
        )
    ], direction="row", style={"height": "calc(100vh - 60px)"})
])


# Editor page callbacks
@callback(
    Output("editor-loading-overlay", "visible", allow_duplicate=True),
    Output("editor-page-select", "value"),
    Output("editor-prev-page-btn", "disabled"),
    Output("editor-next-page-btn", "disabled"),
    Output("editor-manual-adjustments", "data"),
    Output("editor-notification-container", "sendNotifications"),
    Output("shared-page-state", "data", allow_duplicate=True),
    Input("editor-page-select", "value"),
    Input("editor-prev-page-btn", "n_clicks"),
    Input("editor-next-page-btn", "n_clicks"),
    State("shared-page-state", "data"),
    State("editor-rotation-angle", "value"),
    State("editor-top-bottom-margin", "value"),
    State("editor-left-right-margin", "value"),
    State("editor-float", "value"),
    State("editor-crop-switch", "checked"),
    State("editor-manual-bbox", "data"),
    State("editor-manual-adjustments", "data"),
    prevent_initial_call=True,
)
def page_navigation(page, prev_clicks, next_clicks, shared_page, rotation, tb_margin, lr_margin, float_pos, crop_switch, manual_bbox, manual_state):
    """Handle page navigation in the editor."""
    from components import pages, save_manual_adjustments, load_manual_adjustments, clear_manual_adjustments

    triggered_id = ctx.triggered_id
    prev_disabled = True
    next_disabled = True
    manual_state = manual_state or {}
    manual_bbox = manual_bbox or {}
    notifications = []

    current_page = manual_state.get("current_page")
    if not isinstance(current_page, str) or current_page not in pages:
        current_page = page if isinstance(page, str) else None
    
    # Use shared page state if current page is not set
    if not current_page or current_page == "":
        if shared_page and shared_page != "":
            current_page = shared_page
            page = shared_page

    leaving_page = current_page

    if triggered_id in ["editor-prev-page-btn", "editor-next-page-btn", "editor-page-select"]:
        if isinstance(leaving_page, str) and leaving_page:
            has_adjustments = (
                rotation != 0 or 
                tb_margin != 0 or 
                lr_margin != 0 or 
                float_pos != "center" or
                crop_switch or
                manual_bbox
            )

            if has_adjustments:
                crop_bbox = manual_bbox if crop_switch and manual_bbox else None
                save_manual_adjustments(
                    page=leaving_page,
                    rotation=rotation if rotation != 0 else None,
                    crop_bbox=crop_bbox,
                    tb_margin=tb_margin if tb_margin != 0 else None,
                    lr_margin=lr_margin if lr_margin != 0 else None,
                    float_pos=float_pos if float_pos != "center" else None
                )
                notifications = [{
                    "title": "Manual Adjustments Saved",
                    "id": f"manual-save-{leaving_page}",
                    "action": "show",
                    "message": f"Saved manual adjustments to {leaving_page}",
                    "icon": DashIconify(icon="material-symbols:save-rounded"),
                    "position": "top-center"
                }]
                manual_state["last_loaded"] = {
                    "page": leaving_page,
                    "rotation": float(rotation or 0.0),
                    "tb_margin": float(tb_margin or 0.0),
                    "lr_margin": float(lr_margin or 0.0),
                    "float_pos": float_pos or "center",
                    "crop_switch": bool(crop_bbox),
                    "manual_bbox": crop_bbox or {}
                }
            else:
                clear_manual_adjustments(leaving_page)
                manual_state["last_loaded"] = {
                    "page": leaving_page,
                    "rotation": 0.0,
                    "tb_margin": 0.0,
                    "lr_margin": 0.0,
                    "float_pos": "center",
                    "crop_switch": False,
                    "manual_bbox": {}
                }

    new_page = current_page

    if triggered_id == "editor-prev-page-btn" and isinstance(current_page, str) and current_page in pages:
        current_index = pages.index(current_page)
        new_index = (current_index - 1) % len(pages)
        new_page = pages[new_index]
    elif triggered_id == "editor-next-page-btn" and isinstance(current_page, str) and current_page in pages:
        current_index = pages.index(current_page)
        new_index = (current_index + 1) % len(pages)
        new_page = pages[new_index]
    elif triggered_id == "editor-page-select" and isinstance(page, str):
        new_page = page
    elif isinstance(page, str):
        new_page = page

    if isinstance(new_page, str) and new_page in pages:
        current_index = pages.index(new_page)
        prev_disabled = current_index == 0
        next_disabled = current_index == len(pages) - 1
        manual_values = load_manual_adjustments(new_page) or {}
        manual_state["last_loaded"] = {
            "page": new_page,
            "rotation": float(manual_values.get("rotation", 0.0) or 0.0),
            "tb_margin": float(manual_values.get("tb_margin", 0.0) or 0.0),
            "lr_margin": float(manual_values.get("lr_margin", 0.0) or 0.0),
            "float_pos": manual_values.get("float_pos", "center") or "center",
            "crop_switch": bool(manual_values.get("crop_bbox")),
            "manual_bbox": manual_values.get("crop_bbox") or {}
        }

    else:
        new_page = ""
        prev_disabled = True
        next_disabled = True

    manual_state["current_page"] = new_page

    return True, new_page, prev_disabled, next_disabled, manual_state, notifications, new_page

@callback(
    Output("editor-rotation-angle", "value"),
    Output("editor-top-bottom-margin", "value"),  
    Output("editor-left-right-margin", "value"),
    Output("editor-float", "value"),
    Output("editor-crop-switch", "checked"),
    Output("editor-manual-bbox", "data", allow_duplicate=True),
    Input("editor-manual-adjustments", "data"),
    State("editor-page-select", "value"),
    prevent_initial_call=True,
)
def load_manual_adjustments_for_page(manual_state, page_value):
    """Load previously saved manual adjustments when a page is selected or navigation buttons are used."""
    from components import load_manual_adjustments
    manual_state = manual_state or {}

    target_page = manual_state.get("current_page")
    if not isinstance(target_page, str) or not target_page:
        target_page = page_value if isinstance(page_value, str) else None

    if not isinstance(target_page, str) or not target_page:
        return 0.0, 0.0, 0.0, "center", False, {}

    snapshot = manual_state.get("last_loaded")
    if isinstance(snapshot, dict) and snapshot.get("page") == target_page:
        manual_values = snapshot
    else:
        manual_values = load_manual_adjustments(target_page) or {}

    rotation = float(manual_values.get('rotation', 0.0) or 0.0)
    tb_margin = float(manual_values.get('tb_margin', 0.0) or 0.0)
    lr_margin = float(manual_values.get('lr_margin', 0.0) or 0.0)
    float_pos = manual_values.get('float_pos', "center") or "center"
    crop_bbox = manual_values.get('manual_bbox') or manual_values.get('crop_bbox') or {}
    crop_switch = bool(crop_bbox)

    return rotation, tb_margin, lr_margin, float_pos, crop_switch, crop_bbox


@callback(
    Output("editor-page-viewer", "figure"),
    Output("editor-info-container", "children"),
    Output("editor-loading-overlay", "visible", allow_duplicate=True),
    Output("editor-manual-bbox", "data"),
    Input("editor-page-select", "value"),
    Input("editor-rotation-angle", "value"),
    Input("editor-top-bottom-margin", "value"),
    Input("editor-left-right-margin", "value"),
    Input("editor-float", "value"),
    Input("editor-crop-switch", "checked"),
    Input("editor-preview-switch", "checked"),
    Input("editor-page-viewer", "relayoutData"),
    State("editor-manual-bbox", "data"),
    prevent_initial_call=True,
)
def update_editor_page(page, rotation, top_bottom_margin, 
                       left_right_margin, float_position, crop_switch_checked, preview_switch_checked, relayout_data,
                       manual_bbox):
    """Update the editor page viewer with the selected page and any manual adjustments."""

    manual_bbox = {} if manual_bbox is None else manual_bbox    
    loading = False
    if not isinstance(page, str):
        info = [
            dmc.Alert(
                "Select a page to start editing. Use the controls on the left to manually adjust the processing parameters.",
                title="Getting Started",
                color="blue",
                icon=DashIconify(icon="mdi:information"),
            )
        ]
        return create_empty_figure(), info, loading, manual_bbox

    # Validate rotation angle - must be numeric
    try:
        rotation = float(rotation)
    except (ValueError, TypeError):
        # If rotation is not numeric, use 0 as default
        rotation = 0
    

    if crop_switch_checked and relayout_data and 'shapes' in relayout_data:
        # Check if a rectangle was drawn
        shapes = relayout_data.get('shapes', [])
        if shapes:
            # Get the last drawn shape (rectangle)
            last_shape = shapes[-1]
            if last_shape.get('type') == 'rect':
                manual_bbox = {
                    'x0': last_shape.get('x0'),
                    'y0': last_shape.get('y0'),
                    'x1': last_shape.get('x1'),
                    'y1': last_shape.get('y1')
                }
    
    # Create figure with appropriate settings
    if preview_switch_checked:
        # Preview mode - show processed image with manual adjustments if available
        current_adjustments = {
            "rotation": rotation,
            "top_bottom_margin": top_bottom_margin,
            "left_right_margin": left_right_margin,
            "float_position": float_position,
            "manual_crop": crop_switch_checked,
        }
        
        # Include manual bbox if available (regardless of crop switch state in preview mode)
        # The user may have drawn a bbox before switching to preview mode
        if manual_bbox:
            current_adjustments["manual_bbox"] = manual_bbox
            current_adjustments["manual_crop"] = True  # Ensure manual crop is enabled when bbox exists
        
        # Check if any manual adjustments are being applied
        # In preview mode, we should use manual processing to apply all current settings
        has_manual_adjustments = (
            rotation != 0 or 
            top_bottom_margin != 0 or 
            left_right_margin != 0 or 
            crop_switch_checked or
            manual_bbox  # Include manual bbox
        )
        
        # Always use manual processing in preview mode to ensure float position is respected
        # even when it's set to the default "center" value
        
        # Always use manual processing in preview mode to ensure float position is respected
        # even when it's set to the default "center" value
        
        if True:  # Always use manual processing in preview mode
            # Show manually processed image
            from components import process_image_with_manual_adjustments
            try:
                processed_img = process_image_with_manual_adjustments(page, current_adjustments)
                fig = px.imshow(np.asarray(processed_img))
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
                )
            except Exception as e:
                # Fallback to automatic processing if manual processing fails
                from components import create_page_figure
                fig = create_page_figure(page, step=6, show_bboxes=False)
        else:
            # This branch is no longer used in preview mode
            # Show final processed image from automatic pipeline
            from components import create_page_figure
            fig = create_page_figure(page, step=6, show_bboxes=False)
        # Override the drag mode for preview
        fig.update_layout(dragmode="zoom")
    else:
        # Original mode - show original image with bounding boxes
        dragmode = "drawrect" if crop_switch_checked else "zoom"
        show_auto_bbox = not crop_switch_checked
        show_manual_bbox = crop_switch_checked and manual_bbox
        
        fig = create_rotated_page_figure_with_manual_crop(
            page, 
            rotation_angle=rotation, 
            step=0, 
            show_auto_bboxes=show_auto_bbox,
            show_manual_bbox=show_manual_bbox,
            manual_bbox=manual_bbox,
            dragmode=dragmode
        )
    
    # Create info panel with current settings
    if preview_switch_checked:
        # Show manual bbox info in preview mode if it exists
        if manual_bbox:
            bbox = manual_bbox
            bbox_info = f"Using manual bbox: ({bbox['x0']:.0f}, {bbox['y0']:.0f}) to ({bbox['x1']:.0f}, {bbox['y1']:.0f})"
            alert_message = "Preview mode shows the final processed image using your manual crop area. Toggle off preview mode to modify the crop area."
        else:
            bbox_info = "Using automatic text boundary detection"
            alert_message = "Preview mode shows the final processed image. Manual crop mode is disabled in preview mode. Toggle off preview mode to enable manual cropping."
        alert_title = "Preview Mode"
        alert_color = "green"
        alert_icon = "mdi:eye"
    else:
        if crop_switch_checked:
            if manual_bbox:
                bbox = manual_bbox
                bbox_info = f"Manual bbox: ({bbox['x0']:.0f}, {bbox['y0']:.0f}) to ({bbox['x1']:.0f}, {bbox['y1']:.0f})"
            else:
                bbox_info = "Draw a rectangle to define crop area"
        else:
            bbox_info = "Using automatic text boundary detection"
        alert_message = "Rotation changes are applied in real-time. Toggle the crop switch to enable manual bounding box drawing. Manual adjustments will be automatically saved when you navigate to another page."
        alert_title = "Manual Mode"
        alert_color = "yellow"
        alert_icon = "mdi:pencil"
    
    info_content = [
        dmc.Title("Current Settings", order=6, mb="md"),
        dmc.Group([dmc.Text("Page:", fw=500), dmc.Text(page)], gap="xs"),
        dmc.Divider(my="xs"),
        dmc.Group([dmc.Text("View Mode:", fw=500), dmc.Text("Preview" if preview_switch_checked else "Original")], gap="xs"),
        dmc.Group([dmc.Text("Rotation:", fw=500), dmc.Text(f"{rotation}°")], gap="xs"),
        dmc.Group([dmc.Text("TB Margin:", fw=500), dmc.Text(f"{top_bottom_margin or 0}%")], gap="xs"),
        dmc.Group([dmc.Text("LR Margin:", fw=500), dmc.Text(f"{left_right_margin or 0}%")], gap="xs"),
        dmc.Group([dmc.Text("Float:", fw=500), dmc.Text(str(float_position or "center").title())], gap="xs"),
        dmc.Group([dmc.Text("Crop Mode:", fw=500), dmc.Text("Manual" if (crop_switch_checked or manual_bbox) else "Auto")], gap="xs"),
        dmc.Text(bbox_info, size="sm", c="dimmed"),
        dmc.Divider(my="md"),
        dmc.Alert(
            alert_message,
            title=alert_title,
            color=alert_color,
            icon=DashIconify(icon=alert_icon),
        )
    ]
    
    info_card = dmc.Card(dmc.Group(info_content, gap="xs"))
    
    return fig, info_card, False, manual_bbox


@callback(
    Output("editor-rotation-angle", "value", allow_duplicate=True),
    Output("editor-top-bottom-margin", "value", allow_duplicate=True),  
    Output("editor-left-right-margin", "value", allow_duplicate=True),
    Output("editor-float", "value", allow_duplicate=True),
    Output("editor-crop-switch", "checked", allow_duplicate=True),
    Output("editor-manual-bbox", "data", allow_duplicate=True),
    Input("editor-reset-btn", "n_clicks"),
    State("editor-page-select", "value"),
    prevent_initial_call=True,
)
def reset_manual_adjustments(reset_clicks, page):
    """Reset manual adjustments for the current page to auto values."""
    from components import clear_manual_adjustments
    
    if not isinstance(page, str) or not page:
        return 0.0, 0.0, 0.0, "center", False, {}
    
    triggered_id = ctx.triggered_id
    
    if triggered_id == "editor-reset-btn":
        # Clear manual adjustments from dataframe
        clear_manual_adjustments(page)
        
        # Reset UI controls to default values
        return 0.0, 0.0, 0.0, "center", False, {}
    
    # This shouldn't happen due to prevent_initial_call=True
    return 0.0, 0.0, 0.0, "center", False, {}


@callback(
    Output("editor-crop-switch", "disabled"),
    Input("editor-preview-switch", "checked"),
    prevent_initial_call=True,
)
def manage_crop_switch(preview_mode):
    """Disable crop switch in preview mode."""
    # Disable crop switch in preview mode, enable in original mode
    if preview_mode:
        return True  # Disable crop switch in preview mode
    else:
        return False  # Enable crop switch in original mode


@callback(
    [Output("editor-page-select", "data", allow_duplicate=True),
     Output("editor-page-select", "value", allow_duplicate=True)],
    [Input("shared-page-state", "data"),
     Input("url", "pathname")],
    prevent_initial_call=True
)
def update_editor_page_selector(shared_page, pathname):
    """Update page selector options when configuration changes."""
    from dash import no_update
    from components import get_pages_list
    
    # Only update if we're on the editor page
    if pathname != "/editor":
        return no_update, no_update
    
    # Refresh the pages list from current configuration
    current_pages = get_pages_list()
    
    if len(current_pages) <= 1:  # Only empty option or no options
        return [{"value": "", "label": "No images found"}], ""
    
    page_data = [{"value": page, "label": page} for page in current_pages]
    
    # Set value to shared page if it exists in the list, otherwise use first non-empty page
    selected_page = ""
    if shared_page and shared_page in current_pages:
        selected_page = shared_page
    elif len(current_pages) > 1:  # Has pages beyond empty option
        selected_page = current_pages[1]  # First actual page (index 0 is empty)
    
    return page_data, selected_page


@callback(
    [Output("editor-page-select", "data"),
     Output("editor-page-select", "value", allow_duplicate=True)],
    [Input("shared-page-state", "data")],
    prevent_initial_call=True
)
def update_page_selector(shared_page):
    """Update page selector options when configuration changes."""
    from components import get_pages_list
    
    # Refresh the pages list from current configuration
    current_pages = get_pages_list()
    
    if len(current_pages) <= 1:  # Only empty option or no options
        return [{"value": "", "label": "No images found"}], ""
    
    page_data = [{"value": page, "label": page} for page in current_pages]
    
    # Set value to shared page if it exists in the list, otherwise use first non-empty page
    selected_page = ""
    if shared_page and shared_page in current_pages:
        selected_page = shared_page
    elif len(current_pages) > 1:  # Has pages beyond empty option
        selected_page = current_pages[1]  # First actual page (index 0 is empty)
    
    return page_data, selected_page
