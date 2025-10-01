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
            dmc.StepperStep(label="Rotate Page"),
            dmc.StepperStep(label="Crop to Text"),
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
    from components import pages
    
    triggered_id = ctx.triggered_id
    prev_disabled = True
    next_disabled = True
    
    # Use shared page state if current page is empty/not set
    if not page or page == "":
        if shared_page and shared_page != "":
            page = shared_page
    
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

    return True, page, prev_disabled, next_disabled, page, step, triggered_id, page


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
        show_bboxes = step in [1, 2]
        fig = create_page_figure(page, step, show_bboxes)
        
        # Create info card
        info_card = create_page_info_card(page, step)
        
        return fig, info_card, False

    # return empty figure if no valid page is selected
    return create_empty_figure(), info_card, False
