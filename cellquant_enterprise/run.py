#!/usr/bin/env python
"""
CellQuant - Entry Point

Launch the cell quantification application.
"""

import argparse
import sys
from pathlib import Path


def main():
    from cellquant_enterprise import __version__

    parser = argparse.ArgumentParser(
        description=f"CellQuant v{__version__} - High-throughput cell quantification"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the web server (default: 7860)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the web server (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )

    args = parser.parse_args()

    # Set GPU environment if requested
    if args.no_gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Import and launch the app
    from cellquant_enterprise.ui.app import create_app, get_theme, get_css, get_js

    print(f"""
    ================================================================

       CELLQUANT v{__version__}

       High-throughput cell quantification
       Vectorized CTCF | Batch GPU Segmentation | Napari ROI

       Themes: Histology Atlas (light) / Deep Focus (dark)

    ================================================================
    """)

    app = create_app()

    print(f"\n  Starting server at http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop\n")

    # Gradio 6.x: theme, css, js passed to launch()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        theme=get_theme(),
        css=get_css(),
        js=get_js()
    )


if __name__ == "__main__":
    main()
