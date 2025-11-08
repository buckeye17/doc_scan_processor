"""
Automated Result Viewer page - shows results from each automated processing step.

This page is registered with Dash to enable routing, but the actual layout 
is handled by the main app.py with a centralized three-column design.
"""
import dash
from dash import callback, ctx, dcc, html, Input, Output, State
from dash_iconify import DashIconify
import dash_mantine_components as dmc

from components import create_empty_figure, create_page_select, create_navigation_buttons

# Register this page
dash.register_page(__name__, path="/", title="Automated Result Viewer")

controls = dmc.Stack([
    dmc.Title("Processing Steps", order=4, mb="md"),
    dmc.Group([
        create_page_select("viewer"),
        create_navigation_buttons("viewer"),
    ]),
    dmc.Divider(my="md"),
    dmc.Stepper(
        children=[
            dmc.StepperStep(label="Original Image"),
            dmc.StepperStep(label="Find Text"),
            dmc.StepperStep(label="Crop to Text & Rotate"),
            dmc.StepperStep(label="Add Margins"),
            dmc.StepperStep(label="Resize Page"),
            dmc.StepperStep(label="Final Result"),
        ],
        active=0,
        orientation="vertical",
        id="viewer-process-steps"
    ),
    # Store components for state management
    dmc.Box([
        dcc.Store(id="viewer-page-select-value"),
        dcc.Store(id="viewer-active-step-value"),
        dcc.Store(id="viewer-trigger-value")
    ], style={"display": "none"})
])

viewer = dmc.Box([
    dmc.LoadingOverlay(
        visible=False,
        id="viewer-loading-overlay",
        overlayProps={"radius": "sm", "blur": 2},
        zIndex=10,
    ),
    dcc.Graph(
        figure=create_empty_figure(), 
        id="viewer-page-viewer",
        style={"height": "100%", "width": "100%"}
    )
], style={"height": "100%", "width": "100%"})

info = dmc.Stack([
    dmc.Title("Page Information", order=4, mb="md"),
    dmc.Box(id="viewer-info-container", style={"minHeight": "200px"})
])

layout = dmc.Flex([
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
        viewer,
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


# Viewer page callbacks
@callback(
    Output("viewer-loading-overlay", "visible", allow_duplicate=True),
    Output("viewer-page-select", "value"),
    Output("viewer-prev-page-btn", "disabled"),
    Output("viewer-next-page-btn", "disabled"),
    Output("viewer-page-select-value", "data"),
    Output("viewer-active-step-value", "data"),
    Output("viewer-trigger-value", "data"),
    Output("shared-page-state", "data", allow_duplicate=True),
    Input("viewer-page-select", "value"),
    Input("viewer-process-steps", "active"),
    Input("viewer-prev-page-btn", "n_clicks"),
    Input("viewer-next-page-btn", "n_clicks"),
    State("shared-page-state", "data"),
    prevent_initial_call=True,
)
def page_navigation(page, step, prev_clicks, next_clicks, shared_page):
    """Handle page navigation and step changes for viewer."""
    from components import pages, get_shared_page_state_image, get_shared_page_state_last_active, create_shared_page_state

    print(f"Viewer: {ctx.triggered_id}")
    
    triggered_id = ctx.triggered_id
    prev_disabled = True
    next_disabled = True
    
    # Extract shared page state info
    shared_image = get_shared_page_state_image(shared_page)
    last_active_page = get_shared_page_state_last_active(shared_page)
    
    # Check if we're switching from editor to viewer
    # If triggered by viewer-page-select and viewer was NOT the last active page,
    # load the image from shared state
    if triggered_id == "viewer-page-select" and last_active_page != "viewer":
        if shared_image and shared_image in pages:
            page = shared_image
    
    # Use shared page state if current page is empty/not set
    if not page or page == "":
        if shared_image and shared_image != "":
            page = shared_image
    
    if isinstance(page, str):
        # update selected page if navigation button was clicked
        if triggered_id == "viewer-prev-page-btn":
            current_index = pages.index(page)
            new_index = (current_index - 1) % len(pages)
            page = pages[new_index]

        elif triggered_id == "viewer-next-page-btn":
            current_index = pages.index(page)
            new_index = (current_index + 1) % len(pages)
            page = pages[new_index]
        
        current_index = pages.index(page)
        prev_disabled = current_index == 0
        next_disabled = current_index == len(pages) - 1

    # Update shared page state with current image and mark viewer as last active
    new_shared_state = create_shared_page_state(page, "viewer")

    return True, page, prev_disabled, next_disabled, page, step, triggered_id, new_shared_state


@callback(
    Output("viewer-page-viewer", "figure"),
    Output("viewer-info-container", "children"),
    Output("viewer-loading-overlay", "visible", allow_duplicate=True),
    Input("viewer-page-select-value", "data"),
    Input("viewer-active-step-value", "data"),
    Input("viewer-trigger-value", "data"),
    prevent_initial_call=True,
)
def update_viewer_page(page, step, triggered_id):
    """Update the page viewer with the selected page and processing step."""
    from components import create_page_figure, create_page_info_card, create_empty_figure
    
    info_card = None
    if isinstance(page, str) and page:
        # Create figure with bounding boxes for specific steps
        show_bboxes = step in [1]
        fig = create_page_figure(page, step, show_bboxes)
        
        # Create info card
        info_card = create_page_info_card(page, step)
        
        return fig, info_card, False

    # return empty figure if no valid page is selected
    return create_empty_figure(), info_card, False


@callback(
    [Output("viewer-page-select", "data", allow_duplicate=True),
     Output("viewer-page-select", "value", allow_duplicate=True)],
    [Input("shared-page-state", "data"),
     Input("url", "pathname")],
    prevent_initial_call=True
)
def update_page_selector(shared_page, pathname):
    """Update page selector options when configuration changes."""
    from dash import no_update
    from components import get_pages_list, get_shared_page_state_image
    
    # Only update if we're on the viewer page
    if pathname != "/":
        return no_update, no_update
    
    # Refresh the pages list from current configuration
    current_pages = get_pages_list()
    
    if len(current_pages) <= 1:  # Only empty option or no options
        return [{"value": "", "label": "No images found"}], ""
    
    page_data = [{"value": page, "label": page} for page in current_pages]
    
    # Extract the current image from shared page state
    shared_image = get_shared_page_state_image(shared_page)
    
    # Set value to shared page if it exists in the list, otherwise use first non-empty page
    selected_page = ""
    if shared_image and shared_image in current_pages:
        selected_page = shared_image
    elif len(current_pages) > 1:  # Has pages beyond empty option
        selected_page = current_pages[1]  # First actual page (index 0 is empty)
    
    return page_data, selected_page
