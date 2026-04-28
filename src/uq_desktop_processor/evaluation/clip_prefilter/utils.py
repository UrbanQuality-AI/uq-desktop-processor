"""
Image discovery, directory checks, and file moves for the prefilter workflow.
"""

import logging
import os
import shutil
import time

log = logging.getLogger(__name__)


def _list_images(folder: str) -> list[str]:
    """
    Return list of image filenames (png/jpg/jpeg) from a folder.

    :param folder: Path to folder.
    :return: Sorted list of image filenames.

    Example::
        In: _list_images(folder)
        Out: function result returned for provided inputs
    """
    log.debug("Scanning folder for images: %s", folder)

    return sorted(filename for filename in os.listdir(folder) if filename.lower().endswith((".png", ".jpg", ".jpeg")))


def _unique_path(destination_directory: str, filename: str) -> str:
    """
    Generate a unique destination path to avoid overwriting existing files.
    Appends timestamp and a counter until a free filename is found.

    :param destination_directory: Target folder.
    :param filename: Original filename.
    :return: Unique full file path.

    Example::
        In: _unique_path(destination_directory, filename)
        Out: function result returned for provided inputs
    """
    file_base, file_extension = os.path.splitext(filename)
    candidate_path = os.path.join(destination_directory, filename)
    counter = 1

    while os.path.exists(candidate_path):
        candidate_path = os.path.join(
            destination_directory, f"{file_base}__{int(time.time())}_{counter}{file_extension}"
        )
        counter += 1

    if counter > 1:
        log.debug(
            "File collision detected. Renamed to: %s",
            os.path.basename(candidate_path),
        )

    return candidate_path


def _ensure_dir(directory_path: str) -> None:
    """
    Ensure that a directory exists, creating it if needed.

    :param directory_path: Directory path to ensure.

    Example::
        In: _ensure_dir(directory_path)
        Out: function result returned for provided inputs
    """
    if not os.path.exists(directory_path):
        log.debug("Creating directory: %s", directory_path)
    os.makedirs(directory_path, exist_ok=True)


def _move(source_path: str, destination_directory: str) -> None:
    """
    Move a file to another directory using a unique destination filename.

    :param source_path: Source file path.
    :param destination_directory: Target folder.

    Example::
        In: _move(source_path, destination_directory)
        Out: function result returned for provided inputs
    """
    _ensure_dir(destination_directory)
    unique_destination = _unique_path(destination_directory, os.path.basename(source_path))

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Moving file '%s' to '%s'", source_path, unique_destination)

    shutil.move(source_path, unique_destination)


def resolve_rejected_folder_path(image_folder: str, rejected_folder: str) -> str:
    """
    Resolve where rejected images are stored.

    If ``rejected_folder`` is an absolute path, it is used as-is (expanded).
    Otherwise it is resolved relative to the parent directory of ``image_folder``
    (same behaviour as a sibling folder name such as ``rejected``).

    :param image_folder: Folder being filtered (absolute or relative).
    :param rejected_folder: Absolute path, or relative name/path under the image folder's parent.
    :return: Absolute path to the rejected-images directory.

    Example::
        In: resolve_rejected_folder_path(image_folder, rejected_folder)
        Out: function result returned for provided inputs
    """
    image_abs = os.path.abspath(os.path.expanduser(image_folder))
    rejected_folder_input = (rejected_folder or "").strip() or "rejected"
    rejected_folder_expanded = os.path.expanduser(rejected_folder_input)
    if os.path.isabs(rejected_folder_expanded):
        return os.path.abspath(rejected_folder_expanded)
    return os.path.abspath(os.path.join(os.path.dirname(image_abs), rejected_folder_expanded))


def _validate_dirs(image_folder: str, rejected_absolute_path: str) -> None:
    """
    Ensure the source image folder and the destination rejected folder are distinct.

    This prevents operations where files might be moved into the same directory
    they currently reside in.

    :param image_folder: Path to the source directory containing images.
    :param rejected_absolute_path: Absolute path to the directory for rejected items.
    :raises ValueError: If both paths resolve to the same directory.

    Example::
        In: _validate_dirs(image_folder, rejected_absolute_path)
        Out: function result returned for provided inputs
    """
    if os.path.abspath(image_folder) == os.path.abspath(rejected_absolute_path):
        error_message = "rejected_folder must be different from image_folder."
        log.error(error_message)
        raise ValueError(error_message)
