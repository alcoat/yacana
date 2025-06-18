from typing import Callable

from src.yacana import Tool


class YacanaTool(Tool):

    def __init__(self, tool_name: str, function_description: str, max_custom_error: int = 5, max_call_error: int = 5):
        super().__init__(tool_name, function_description, function_ref=None, max_custom_error=max_custom_error, max_call_error=max_call_error, optional=True)

    @staticmethod
    def _extract_prototype(func: Callable) -> str:
        """
        Extract the function prototype as a string.

        Parameters
        ----------
        func : Callable
            The function to extract the prototype from.

        Returns
        -------
        str
            The function prototype as a string, including the function name and signature.
        """
        # Get the function's signature
        sig = inspect.signature(func)
        # Format the signature as a string and returns it
        return f"{func.__name__}{sig}"

    @staticmethod
    def _extract_parameters(func: Callable) -> List[str]:
        """
        Extract the parameter names from a function's signature.

        Parameters
        ----------
        func : Callable
            The function to extract parameters from.

        Returns
        -------
        List[str]
            A list of parameter names from the function's signature.
        """
        signature = inspect.signature(func)
        # Access the parameters
        parameters = signature.parameters
        # Extract the parameter names into a list
        return [param_name for param_name in parameters]