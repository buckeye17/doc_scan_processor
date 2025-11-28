import dash
from dash import html, dcc, callback, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import math
from components import get_pages_list, get_image_path, encode_image_base64

dash.register_page(__name__, path="/gallery", title="Gallery")

layout = dmc.Stack([
    dmc.Group([
        dmc.Title("Gallery View", order=2),
        dmc.Group([
            dmc.Group([
                dmc.Text("Cols:", size="sm"),
                dmc.NumberInput(
                    value=3,
                    min=1,
                    max=6,
                    step=1,
                    id="gallery-cols-input",
                    w=60,
                    size="sm"
                )
            ], gap="xs"),
            dmc.Group([
                dmc.Text("Rows:", size="sm"),
                dmc.NumberInput(
                    value=2,
                    min=1,
                    max=5,
                    step=1,
                    id="gallery-rows-input",
                    w=60,
                    size="sm"
                )
            ], gap="xs"),
            dmc.Button(
                "Previous",
                leftSection=DashIconify(icon="fluent:chevron-left-24-regular"),
                id="gallery-prev-btn",
                disabled=True,
                variant="outline"
            ),
            dmc.Text(id="gallery-page-info"),
            dmc.Button(
                "Next",
                rightSection=DashIconify(icon="fluent:chevron-right-24-regular"),
                id="gallery-next-btn",
                disabled=False,
                variant="outline"
            )
        ])
    ], justify="space-between", mb="lg"),

    dmc.SimpleGrid(
        cols=3,
        spacing="lg",
        verticalSpacing="lg",
        id="gallery-grid",
        children=[]
    ),
    
    dcc.Store(id="gallery-page-index", data=0)
], p="md")

@callback(
    [Output("gallery-grid", "children"),
     Output("gallery-grid", "cols"),
     Output("gallery-page-info", "children"),
     Output("gallery-prev-btn", "disabled"),
     Output("gallery-next-btn", "disabled"),
     Output("gallery-page-index", "data")],
    [Input("gallery-prev-btn", "n_clicks"),
     Input("gallery-next-btn", "n_clicks"),
     Input("url", "pathname"),
     Input("gallery-cols-input", "value"),
     Input("gallery-rows-input", "value")],
    [State("gallery-page-index", "data")]
)
def update_gallery(prev_clicks, next_clicks, pathname, cols, rows, current_page_index):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    pages = get_pages_list()
    # Filter out empty strings if any
    pages = [p for p in pages if p]
    
    # Default to 3 cols, 2 rows if None
    cols = cols or 3
    rows = rows or 2
    
    if not pages:
        return [], cols, "No images found", True, True, 0
    
    items_per_page = cols * rows
    total_pages = math.ceil(len(pages) / items_per_page)
    
    # Handle initial load or navigation
    if current_page_index is None:
        current_page_index = 0
        
    if triggered_id == "gallery-prev-btn":
        current_page_index = max(0, current_page_index - 1)
    elif triggered_id == "gallery-next-btn":
        current_page_index = min(total_pages - 1, current_page_index + 1)
    elif triggered_id == "url":
        # Reset to 0 if we just navigated to the gallery, or keep state?
        # If we want to keep state, we rely on the State passed in.
        # But if we are coming from another page, maybe we want to start at 0?
        # For now, let's trust the store.
        pass
            
    # Ensure index is within bounds
    if current_page_index >= total_pages:
        current_page_index = total_pages - 1
    if current_page_index < 0:
        current_page_index = 0
            
    start_idx = current_page_index * items_per_page
    end_idx = start_idx + items_per_page
    current_batch = pages[start_idx:end_idx]
    
    cards = []
    for page_name in current_batch:
        # Get path for step 5 (Final Result)
        img_path = get_image_path(page_name, step=5)
        img_src = encode_image_base64(img_path)
        
        # Create image component or placeholder
        if img_src:
            image_component = dmc.Image(
                src=img_src,
                h="25vh",
                w="auto",
                fit="contain",
                style={"border": "1px solid #228be6"}
            )
        else:
            image_component = dmc.Center(
                dmc.Text("No Image", c="dimmed"),
                h="30vh",
                w=200,
                bg="gray.1"
            )
        
        card = dmc.Card([
            dmc.Stack([
                image_component,
                dmc.Text(page_name, fw=500, ta="center")
            ], align="center", gap="sm")
        ], withBorder=True, shadow="sm", radius="md", w="fit-content", p="sm")
        cards.append(dmc.Center(card))
        
    page_info = f"Page {current_page_index + 1} of {total_pages}"
    prev_disabled = current_page_index == 0
    next_disabled = current_page_index >= total_pages - 1
    
    return cards, cols, page_info, prev_disabled, next_disabled, current_page_index
