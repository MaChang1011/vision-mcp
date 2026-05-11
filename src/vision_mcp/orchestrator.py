class VisionOrchestrator:
    def process(self, tool_name: str, params: dict):
        return {
            "ok": False,
            "engine": "none",
            "data": None,
            "meta": {},
            "error": f"tool not implemented: {tool_name}",
        }
