class SymbolNotRegisteredError(Exception):
    symbol: str
    operation: str

    def __init__(
        self, message: str, *, symbol: str, operation: str = ""
    ) -> None: ...
