import abc
from myExceptions import AgentError, EnvironmentError

class Statement(abc.ABC):
    def __init__(self):
        self.parent = None
    @abc.abstractmethod
    def _toString(self, indent):
        pass
    def __str__(self):
        return self._toString(0)
    
    @abc.abstractmethod
    def evaluate(self, arguments):
        pass
    def add(self, statement):
        statement.parent = self
    def encode(self):
        return ''

class FunctionStatement(Statement):
    def __init__(self, name, eval_func = None):
        super().__init__()
        self.name = name
        self.eval_func = eval_func

    def _toString(self, indent):
        return ('\t' * indent) + self.name
    
    def evaluate(self, arguments):
        if self.eval_func is None:
            raise EnvironmentError("Evaluation function is undefined")

        retVal = self.eval_func(arguments)
        if retVal is None:
            raise EnvironmentError("Evaluation function returned None !!!")
        return retVal

class ConditionStatement(FunctionStatement):
    def __init__(self, name, eval_func=None):
        super().__init__(name, eval_func)
        self.ifBlock = None
        self.elseBlock = None

    def _toString(self, indent):
        if self.ifBlock is None:
            return ''
        res = ''
        indentation = '\t' * indent
        if self.ifBlock is not None:
            res += indentation + "Si " + self.name + ' alors\n'
            res += self.ifBlock._toString(indent + 1) + '\n'
        if self.elseBlock is not None:
            res += indentation  + 'Sinon\n'
            res += self.elseBlock._toString(indent + 1) + '\n'
        return res + indentation + 'FinSi'

    def evaluate(self, arguments):
        if self.ifBlock is None:
            raise AgentError("If block is undefined")
        retVal = super().evaluate(arguments)

        if type(retVal) != bool:
            raise EnvironmentError("Error condition evaluation does not return boolean !!!")

        if retVal:
            return self.ifBlock.evaluate(arguments)
        elif self.elseBlock is not None:
            return self.elseBlock.evaluate(arguments)

        return arguments
    
    def add(self, isIf):
        block = StatementsBlock()
        super().add(block)
        if isIf:
            self.ifBlock = block
        else:
            self.elseBlock = block

    def encode(self):
        codage = ('1' + self.ifBlock.encode()) if self.ifBlock is not None else ''
        codage += ('0' + self.elseBlock.encode()) if self.elseBlock is not None else ''

        return codage

class StatementsBlock(Statement):
    def __init__(self):
        super().__init__()
        self.statement = None
        self.statementsBlock = None
    
    def _toString(self, indent):
        res = ""
        res += self.statement._toString(indent) if self.statement is not None else ""
        res += ('\n'  + self.statementsBlock._toString(indent)) if self.statementsBlock is not None else ""
        return res

    def evaluate(self, arguments):
        if self.statement is None:
            raise AgentError("block is missing statement")

        args = self.statement.evaluate(arguments)

        if self.statementsBlock is not None:
            args = self.statementsBlock.evaluate(args)
        return args

    def add(self, statement_s):
        super().add(statement_s)
        if isinstance(statement_s, FunctionStatement):
            self.statement = statement_s
        elif isinstance(statement_s, StatementsBlock):
            self.statementsBlock = statement_s

    def encode(self):
        codage = ('1' + self.statement.encode()) if self.statement is not None else ''
        codage += ('0' + self.statementsBlock.encode()) if self.statementsBlock is not None else ''

        return codage





