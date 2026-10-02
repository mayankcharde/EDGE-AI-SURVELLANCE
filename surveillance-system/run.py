import logging
import threading
import time
import webbrowser
import uvicorn

from config import HOST, PORT
from app.pipeline import SurveillancePipeline
from app import main as api_module
from app.main import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/surveillance.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("run")


def start_pipeline():
    """Start the pipeline with NO camera source.
    The pipeline loop will idle until the user selects a source via the UI."""
    try:
        # source=None means no camera opened yet — pipeline idles
        pipeline = SurveillancePipeline(source=None, is_file=False)
        api_module.pipeline_ref["instance"] = pipeline
        api_module.pipeline_ref["current_source"] = None
        pipeline.run(show_window=False)
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")


def open_browser():
    time.sleep(3)  # give the server time to bind
    url = f"http://127.0.0.1:{PORT}"
    logger.info(f"Opening browser → {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")


def start_server():
    logger.info(f"Dashboard running at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="error")


if __name__ == "__main__":
    # Start FastAPI server in a background thread
    threading.Thread(target=start_server, daemon=True).start()

    # Open browser after a short delay
    threading.Thread(target=open_browser, daemon=True).start()

    logger.info(f"Dashboard will be available at http://localhost:{PORT}")
    logger.info(f"Manual fallback:  http://127.0.0.1:{PORT}/dashboard")
    logger.info("Camera will NOT start until you select a source in the UI.")

    # Start surveillance pipeline in the MAIN thread (no camera yet)
    # cv2.imshow MUST run in the main thread on Windows/macOS to avoid freezing
    try:
        pipeline = SurveillancePipeline(source=None, is_file=False)
        api_module.pipeline_ref["instance"] = pipeline
        api_module.pipeline_ref["current_source"] = None
        pipeline.run(show_window=False)
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")