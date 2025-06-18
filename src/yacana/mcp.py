import copy
import json
import requests
import uuid
from typing import Dict, List, Any, Optional

from .tool import Tool, ToolType


#@dataclass
#class Tool:
#    name: str
#    description: str
#    input_schema: Dict[str, Any]

class Mcp:

    def __init__(self, server_url: str, headers: dict = None) -> None:
        self.server_url = server_url.rstrip('/')
        self.headers = headers if headers is not None else {}
        self.session = requests.Session()
        self.tools: List[Tool] = []
        self.initialized = False
        self.session_id: Optional[str] = None

    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a JSON-RPC request to the MCP server using Streamable HTTP transport"""
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        if params:
            payload["params"] = params

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        # Add session ID header if we have one
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        try:
            response = self.session.post(
                self.server_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()

            # Check if server returned a session ID during initialization
            if method == "initialize" and "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
                print(f"Server assigned session ID: {self.session_id}")

            # Handle different response types
            content_type = response.headers.get('content-type', '').lower()

            if 'application/json' in content_type:
                result = response.json()
                print("result en application/json", result)
                if "error" in result:
                    raise Exception(f"MCP Error: {result['error']}")
                return result.get("result", {})

            elif 'text/event-stream' in content_type:
                # Handle SSE response - for simplicity, we'll read the first JSON response
                # In a production client, you'd want to properly handle the SSE stream
                print("Received SSE response", response)
                return self._handle_sse_response(response)

            else:
                # If no specific content type, try to parse as JSON
                try:
                    result = response.json()
                    print("On sait pas c'est quoi alors on parse en json", result)
                    if "error" in result:
                        raise Exception(f"MCP Error: {result['error']}")
                    return result.get("result", {})
                except:
                    raise Exception(f"Unexpected response format: {response.text}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [404, 405]:
                # Try fallback to old HTTP+SSE transport
                print("Attempting fallback to legacy HTTP+SSE transport...")
                return self._try_legacy_transport(method, params)
            raise Exception(f"HTTP Error {e.response.status_code}: {e.response.text}")

    def _handle_sse_response(self, response) -> Dict[str, Any]:
        """Handle Server-Sent Events response (simplified implementation)"""
        # This is a simplified SSE parser - in production you'd want a proper SSE client
        lines = response.text.split('\n')
        for line in lines:
            print("line = ", line)
            print("----")
            line = line.strip()
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    if "result" in data:
                        return data["result"]
                    elif "error" in data:
                        raise Exception(f"MCP Error: {data['error']}")
                except json.JSONDecodeError:
                    continue

        raise Exception("No valid JSON-RPC response found in SSE stream")

    def _try_legacy_transport(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback to legacy HTTP+SSE transport for older servers"""
        print("Note: This server appears to use the legacy HTTP+SSE transport")
        print("For full compatibility with legacy servers, consider implementing proper SSE handling")

        # For now, just attempt a GET request to see if we get an SSE endpoint event
        try:
            headers = {"Accept": "text/event-stream"}
            response = self.session.get(self.server_url, headers=headers, timeout=10)

            if response.headers.get('content-type', '').startswith('text/event-stream'):
                print("Server supports SSE - but full legacy support not implemented in this example")
                # You would need to implement the full legacy HTTP+SSE protocol here
                raise Exception("Legacy HTTP+SSE transport detected but not fully supported in this example")

        except Exception as e:
            print(f"Legacy transport attempt failed: {e}")

        raise Exception("Could not connect using either new or legacy MCP transport")

    def connect(self) -> bool:
        """Initialize connection with the MCP server"""
        # Initialize the protocol
        init_result = self._make_request("initialize", {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "yacana-mcp-client",
                "version": "0.1.0"
            }
        })

        server_info = init_result.get('serverInfo', {})
        print(f"Connected to MCP server: {server_info.get('name', 'Unknown')} v{server_info.get('version', 'Unknown')}")

        # List available tools
        tools_result = self._make_request("tools/list")
        tools = tools_result.get("tools", [])

        for tool_info in tools:
            tool = Tool(tool_info.get("name"),
                        tool_info.get("description"),
                        function_ref=self.call_tool,
                        mcp_input_schema=tool_info["inputSchema"])
            self.tools.append(tool)
            print(f"Available tool: {tool.tool_name} - {tool.function_description}")

        if not tools:
            print("No tools available on this server")

        self.initialized = True
        return True

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if not self.initialized:
            raise Exception("Client not initialized")
        print("calling tool:", tool_name, "with arguments: ", arguments)

        #if tool_name not in [tool.tool_name for tool in self.tools]:
        #    raise Exception(f"Tool '{tool_name}' not available. Available tools: {[tool.tool_name for tool in self.tools]}")

        result = self._make_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        return result

    def get_tools_as(self, tools_type: ToolType = ToolType.YACANA):
        tools_copy = copy.deepcopy(self.tools)
        for tool in tools_copy:
            tool.tool_type = tools_type
        return tools_copy

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [tool.tool_name for tool in self.tools]

    def get_tool_info(self, tool_name: str) -> Optional[Tool]:
        """Get information about a specific tool"""
        for tool in self.tools:
            if tool.tool_name == tool_name:
                return tool
        return None

    def disconnect(self):
        """Explicitly disconnect from the server"""
        if self.session_id:
            try:
                headers = {"Mcp-Session-Id": self.session_id}
                self.session.delete(self.server_url, headers=headers, timeout=5)
            except:
                pass  # Ignore errors during cleanup
        self.session.close()