"""
Application entry point for `python -m uq_desktop_processor` (logging setup and GUI launch).
"""

import logging
import sys

from uq_desktop_processor.gui import main as start_gui
from uq_desktop_processor.logging_config import configure_logging


def main() -> int:
    """
    Main entry point for the City-Lens pipeline.

    Configures logging, starts the application, and returns the exit code.
    """
    # 1. Configure logging system
    configure_logging(level=logging.DEBUG)

    # 2. Get the main logger
    log = logging.getLogger(__name__)
    log.debug("Logging configured.")

    # 3. Log application launch
    log.info("Launching UrbanQuality-AI application pipeline...")

    # 4. Run the main pipeline function and get its exit code
    try:
        exit_code = start_gui()

        if exit_code == 0:
            log.info("Application pipeline finished successfully (Exit Code 0).")
        else:
            log.warning("Application pipeline finished with errors (Exit Code %d).", exit_code)

        return exit_code

    except Exception as e:
        # Catch any critical, unhandled exceptions
        log.critical("An unhandled exception occurred: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    # Call main() and exit with its proper return code
    sys.exit(main())
