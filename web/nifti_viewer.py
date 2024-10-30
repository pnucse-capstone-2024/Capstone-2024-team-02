import os
import nibabel as nib
import numpy as np
import plotly.graph_objects as go

def create_nifti_visualization(filepath):
    try:
        nii_img = nib.load(filepath) 
    except Exception as e:
        return f"Error loading NIfTI file: {str(e)}", None

    volume = np.asanyarray(nii_img.dataobj)

    # Check volume shape
    if volume.ndim != 4:
        return "Error: The NIfTI file must have a 4D shape (x, y, z, time).", None

    # Extract dimensions
    height, width, slices, volumes = volume.shape
    chosen_volume_index = 0

    frames = [
        go.Frame(
            data=go.Surface(
                z=np.full((height, width), k),
                surfacecolor=np.flipud(volume[:, :, k, chosen_volume_index]),
                cmin=0,
                cmax=np.max(volume),
                colorscale="Gray",
            ),
            name=str(k),
        )
        for k in range(slices)
    ]

    initial_surface = go.Surface(
        z=np.full((height, width), 0),
        surfacecolor=np.flipud(volume[:, :, 0, chosen_volume_index]),
        colorscale="Gray",
        cmin=0,
        cmax=np.max(volume),
        colorbar=dict(thickness=20, ticklen=4),
    )

    fig = go.Figure(data=[initial_surface], frames=frames)

    # Create the slider
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f.name],
                        {
                            "frame": {"duration": 0},
                            "mode": "immediate",
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Set up the layout with controls
    fig.update_layout(
        title="3D Volumetric Data Visualization",
        width=600,
        height=600,
        scene=dict(
            xaxis=dict(range=[0, height], title="Height"),
            yaxis=dict(range=[0, width], title="Width"),
            zaxis=dict(range=[0, slices - 1], title="Slices"),
            aspectmode="cube",
        ),
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 50},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {"duration": 50},
                            },
                        ],
                        "label": "&#9654;",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "&#9724;",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders,
    )

    graph_html = fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})
    
    return None, graph_html
