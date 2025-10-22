"""
Scan Post Processing App - Multi-page application for viewing and editing document processing results.

This is the main application file that sets up the Dash app with pages support.
Individual pages are defined in the pages/ directory.
"""
import dash
from dash import Dash, html, dcc, page_container, callback, Input, Output, State
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# Initialize the Dash app with pages support
app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

nav_bar_content = dmc.Stack([
    dmc.Tooltip(
        label="Setup & Process Images",
        children=dmc.Anchor(
            DashIconify(icon="fluent:image-copy-24-filled", width=24),
            href="/setup",
            underline=False,
            style={
                "padding": "12px",
                "borderRadius": "8px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "color": "#666",
                "width": "48px",
                "height": "48px"
            },
            id="nav-setup"
        ),
        position="right"
    ),
    dmc.Tooltip(
        label="Automated Result Viewer",
        children=dmc.Anchor(
            DashIconify(icon="ix:eye-filled", width=24),
            href="/",
            underline=False,
            style={
                "padding": "12px",
                "backgroundColor": "#e3f2fd",
                "borderRadius": "8px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "color": "#1976d2",
                "width": "48px",
                "height": "48px"
            },
            id="nav-viewer"
        ),
        position="right"
    ),
    dmc.Tooltip(
        label="Manual Result Editor",
        children=dmc.Anchor(
            DashIconify(icon="fluent:image-edit-16-filled", width=24),
            href="/editor",
            underline=False,
            style={
                "padding": "12px",
                "borderRadius": "8px",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "color": "#666",
                "width": "48px",
                "height": "48px"
            },
            id="nav-editor"
        ),
        position="right"
    ),
], gap="sm")

# Main app layout with AppShell containing navbar and three-column layout
app.layout = dmc.MantineProvider([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="shared-page-state", storage_type="session", data={"current_image": "", "last_active_page": ""}),
    dmc.AppShell([
        dmc.AppShellHeader([
            dmc.Group([
                dmc.Text("Scan Post Processing", size="lg", fw=500, c="blue")
            ], h="100%", px="md")
        ]),
        dmc.AppShellNavbar([
            dmc.Box(
                nav_bar_content,
                p="md"
            )
        ], id="navbar"),
        dmc.AppShellMain([
            page_container
        ])
    ], 
    navbar={"width": 80},
    header={"height": 60},
    padding="md",
    id="appshell"
    )
])

# Callbacks for handling navigation and layout updates
@callback(
    [Output("nav-viewer", "style"),
     Output("nav-editor", "style"),
     Output("nav-setup", "style"),
     Output("nav-viewer", "href"),
     Output("nav-editor", "href"),
     Output("nav-setup", "href")],
    [Input("url", "pathname"),
     Input("shared-page-state", "data")]
)
def update_navigation_styles(pathname, current_page):
    """Update navigation link styles and hrefs based on current page."""
    base_style = {
        "padding": "12px",
        "borderRadius": "8px",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
        "width": "48px",
        "height": "48px"
    }
    
    active_style = {
        "backgroundColor": "#e3f2fd",
        "color": "#1976d2"
    }
    
    inactive_style = {
        "backgroundColor": "transparent",
        "color": "#666"
    }
    
    # Initialize all styles as inactive
    viewer_style = {**base_style, **inactive_style}
    editor_style = {**base_style, **inactive_style}
    setup_style = {**base_style, **inactive_style}
    
    # Set active style based on pathname
    if pathname == "/editor":
        editor_style.update(active_style)
    elif pathname == "/setup":
        setup_style.update(active_style)
    else:
        # Viewer is active (default)
        viewer_style.update(active_style)
    
    # Create hrefs that preserve the current page
    viewer_href = "/"
    editor_href = "/editor"
    setup_href = "/setup"
    
    return viewer_style, editor_style, setup_style, viewer_href, editor_href, setup_href

if __name__ == "__main__":
    app.run(debug=True)
