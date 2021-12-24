class EnvironmentError(Exception):
    def __init__(self, *args):
        super().__init__(args)

class AgentError(Exception):
    def __init__(self, *args):
        super().__init__(args)