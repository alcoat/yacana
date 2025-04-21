import logging


class ColoredFormatter(logging.Formatter):
    """
    A custom formatter that adds color to log messages based on their level and content.

    This formatter uses ANSI escape codes to colorize log messages. Special formatting is applied
    for different types of messages like prompts, AI responses, and tool responses.

    Attributes
    ----------
    COLORS : dict
        Dictionary mapping log levels to their corresponding ANSI color codes.
    RESET : str
        ANSI code to reset text color.
    """
    # ANSI escape codes for colors
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[96m'  # CYAN
    }
    RESET = '\033[0m'

    # Magenta '\033[95m'

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with appropriate colors.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to format.

        Returns
        -------
        str
            The formatted log message with appropriate color codes.
        """
        if record.levelname == "INFO" and record.msg.startswith("[PROMPT]"):
            log_color: str = '\033[92m'  # Green
        elif record.levelname == "INFO" and record.msg.startswith("[AI_RESPONSE]"):
            log_color: str = '\033[95m'  # Magenta
        elif record.levelname == "INFO" and record.msg.startswith("[TOOL_RESPONSE]"):
            log_color: str = '\033[96m'  # Magenta
        else:
            log_color: str = self.COLORS.get(record.levelname, self.RESET)
        log_message: str = super().format(record)
        return f"\n{log_color}{log_message}{self.RESET}"


class LoggerManager:
    """
    Manages logging configuration for the application.

    This class provides static methods to configure logging levels and handlers.
    It allows setting different log levels for the main application and specific libraries.
    Defaults to INFO level and includes colored output for different types of messages.

    Attributes
    ----------
    LOG_LEVEL : str
        Default log level for the application.
    AVAILABLE_LOG_LEVELS : list[str]
        List of valid log levels that can be set.
    """
    LOG_LEVEL = "INFO"
    AVAILABLE_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", None]

    @staticmethod
    def setup_logging(log_level: str | None = None) -> None:
        """
        Initialize logging configuration for the application.

        This method sets up the root logger with a colored formatter and configures
        the logging level. It should not be called directly unless you know what you
        are doing. Use set_log_level() instead for normal usage.

        Parameters
        ----------
        log_level : str | None, optional
            The desired log level. Must be one of: "DEBUG", "INFO", "WARNING",
            "ERROR", "CRITICAL", or None. If None, uses the default LOG_LEVEL.
            Defaults to None.

        Notes
        -----
        This method is automatically called when the module is imported.
        """
        if log_level is None:
            log_level = LoggerManager.LOG_LEVEL
        else:
            log_level = log_level.upper()

        if log_level not in LoggerManager.AVAILABLE_LOG_LEVELS:
            log_level = "INFO"
            print("WARNING:root:Invalid value given to LOG_LEVEL. Defaulting to level: INFO")

        # Create and configure the handler with the ColoredFormatter
        handler = logging.StreamHandler()
        formatter = ColoredFormatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)

        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        root_logger.handlers = [handler]  # Replace default handlers with our colored handler

    @staticmethod
    def set_log_level(log_level: str | None) -> None:
        """
        Set the logging level for the application.

        This method allows changing the logging level at runtime. All log levels have
        specific colors, and for INFO level, the color depends on whether it's a user
        prompt or an AI answer. Setting log_level to None will disable logging.

        Parameters
        ----------
        log_level : str | None
            The desired log level. Must be one of: "DEBUG", "INFO", "WARNING",
            "ERROR", "CRITICAL", or None. If None, logging will be disabled.

        Notes
        -----
        If an invalid log level is provided, the current level will be maintained.
        """
        if log_level is None:
            # Disable logging if the log level is None
            logging.disable(logging.CRITICAL)
        elif log_level in LoggerManager.AVAILABLE_LOG_LEVELS:
            logging.getLogger().setLevel(getattr(logging, log_level))
            logging.disable(logging.NOTSET)  # Re-enable logging
        else:
            print(
                f"WARNING:root:Invalid value given to LOG_LEVEL. Keeping current level: {logging.getLevelName(logging.getLogger().level)}")

    @staticmethod
    def set_library_log_level(library_name: str, log_level: str) -> None:
        """
        Set the logging level for a specific Python library.

        This is useful to control logging from external libraries independently
        from the main application logging level.

        Parameters
        ----------
        library_name : str
            The name of the target library.
        log_level : str
            The desired log level. Must be one of: "DEBUG", "INFO", "WARNING",
            "ERROR", "CRITICAL", or None.

        Notes
        -----
        If an invalid log level is provided, a warning will be printed but no changes
        will be made to the library's logging configuration.
        """
        if log_level in LoggerManager.AVAILABLE_LOG_LEVELS:
            library_logger = logging.getLogger(library_name)
            library_logger.setLevel(getattr(logging, log_level))
        else:
            print(f"WARNING:root:Invalid log level {log_level} for library {library_name}.")


# Automatically set up logging when the class is first imported
LoggerManager.setup_logging()
