
import logging
from logging import LogRecord
from logging.handlers import HTTPHandler
from typing import Any, Dict, Optional, Tuple

# Create logger
LOGGER_NAME = "flwr"
FLOWER_LOGGER = logging.getLogger(LOGGER_NAME)
FLOWER_LOGGER.setLevel(logging.DEBUG)

DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
)

# Configure console logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(DEFAULT_FORMATTER)
FLOWER_LOGGER.addHandler(console_handler)


class CustomHTTPHandler(HTTPHandler):
    """Custom HTTPHandler which overrides the mapLogRecords method."""

    # pylint: disable=too-many-arguments,bad-option-value,R1725
    def __init__(
        self,
        identifier: str,
        host: str,
        url: str,
        method: str = "GET",
        secure: bool = False,
        credentials: Optional[Tuple[str, str]] = None,
    ) -> None:
        super(CustomHTTPHandler, self).__init__(host, url, method, secure, credentials)
        self.identifier = identifier

    def mapLogRecord(self, record: LogRecord) -> Dict[str, Any]:
        """Filter for the properties to be send to the logserver."""
        record_dict = record.__dict__
        return {
            "identifier": self.identifier,
            "levelname": record_dict["levelname"],
            "name": record_dict["name"],
            "asctime": record_dict["asctime"],
            "filename": record_dict["filename"],
            "lineno": record_dict["lineno"],
            "message": record_dict["message"],
        }


def configure(
    identifier: str, filename: Optional[str] = None, host: Optional[str] = None
) -> None:
    """Configure logging to file and/or remote log server."""

    # Create formatter
    string_to_input = f"{identifier} | %(levelname)s %(name)s %(asctime)s "
    string_to_input += "| %(filename)s:%(lineno)d | %(message)s"
    formatter = logging.Formatter(string_to_input)

    if filename:
        # Create file handler and log to disk
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        FLOWER_LOGGER.addHandler(file_handler)

    if host:
        # Create http handler which logs even debug messages
        http_handler = CustomHTTPHandler(
            identifier,
            host,
            "/log",
            method="POST",
        )
        http_handler.setLevel(logging.DEBUG)
        # Override mapLogRecords as setFormatter has no effect on what is send via http
        FLOWER_LOGGER.addHandler(http_handler)


logger = logging.getLogger(LOGGER_NAME)  # pylint: disable=invalid-name
log = logger.log  # pylint: disable=invalid-name
