import abc
from myExceptions import AgentError, EnvironmentError

class Statement(abc.ABC):
    def __init__(self):
        self.parent = None
    @abc.abstractmethod
    def __str__(self):
        pass
    
    @abc.abstractmethod
    def evaluate(self, arguments):
        pass
    def add(self, statement):
        statement.parent = self

class FunctionStatement(Statement):
    def __init__(self, name, eval_func = None):
        super().__init__()
        self.name = name
        self.eval_func = eval_func

    def __str__(self):
        return self.name
    
    def evaluate(self, arguments):
        if self.eval_func is None:
            raise EnvironmentError("Evaluation function is undefined")

        retVal = self.eval_func(arguments)
        if retVal is None:
            raise EnvironmentError("Evaluation function returned None !!!")
        return retVal

class ConditionStatement(FunctionStatement):
    def __init__(self, name, eval_func=None, isNeg = False):
        super().__init__(name, eval_func)
        self.ifBlock = None
        self.elseBlock = None
        if isNeg:
            self.name = 'not ' + self.name

    def __str__(self):
        res = ""
        if self.ifBlock is not None:
            res += "Si " + self.name + ' alors\n'
            res += '\t' + str(self.ifBlock) + '\n'
        if self.elseBlock is not None:
            res += 'else\n\t'
            res += str(self.elseBlock) + '\n'
        return res + 'FinSi'

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
    
    def add(self, conditionBlock, isIf = True):
        super().add(conditionBlock)
        if isIf:
            self.ifBlock = conditionBlock
        else:
            self.elseBlock = conditionBlock

class StatementsBlock(Statement):
    def __init__(self):
        super().__init__()
        self.statement = None
        self.statementsBlock = None
    
    def __str__(self):
        res = ""
        res += str(self.statement) if self.statement is not None else ""
        res += '\n'  + (self.statementsBlock if self.statementsBlock is not None else "")
        return res

    def evaluate(self, arguments):
        if self.statement is not None:
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





