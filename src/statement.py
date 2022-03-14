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
    
    @abc.abstractmethod
    def encode_str(self):
        pass

    def encode(self):
        return int(self.encode_str(), 5)

    def setLeft(self, statement):
        pass
    def setRight(self, statement):
        pass
    def getLeft(self):
        return None
    def getRight(self):
        return None

    def removeLeft(self):
        left = self.getLeft()
        if left is not None:
            left.parent = None
        self.setLeft(None)
    def removeRight(self):
        right = self.getRight()
        if right is not None:
            right.parent = None
        self.setRight(None)


class FunctionStatement(Statement):
    def __init__(self, name, coding, eval_func = None):
        super().__init__()
        self.name = name
        self.eval_func = eval_func
        self.coding = coding

    def _toString(self, indent):
        return ('\t' * indent) + self.name
    
    def evaluate(self, arguments):
        if self.eval_func is None:
            raise EnvironmentError("Evaluation function is undefined")

        retVal = self.eval_func(arguments)
        if retVal is None:
            raise EnvironmentError("Evaluation function returned None !!!")
        return retVal

    def encode_str(self):
        return str(self.coding)
    
    def removeLeft(self):
        pass
    def removeRight(self):
        pass

class ConditionStatement(FunctionStatement):
    def __init__(self, name, codage, eval_func=None):
        super().__init__(name, codage, eval_func)
        self.ifBlock = StatementsBlock()
        self.ifBlock.parent = self
        self.elseBlock = None

        self.eval_func_copy = eval_func
        self.isNeg = False

    def negation(self):
        if self.isNeg:
            self.eval_func = self.eval_func_copy
        else:
            self.eval_func = lambda x: not self.eval_func_copy(x)
        self.name = 'non ' + self.name


    def _toString(self, indent):
        indentation = '\t' * indent
        res = indentation + "Si " + self.name + ' alors\n'
        if self.ifBlock is not None:
            res += self.ifBlock._toString(indent + 1) + '\n'
        if self.elseBlock is not None:
            res += indentation  + 'Sinon\n'
            res += self.elseBlock._toString(indent + 1) + '\n'
        return res + indentation + 'FinSi'

    def evaluate(self, arguments):
        retVal = super().evaluate(arguments)

        if type(retVal) != bool:
            raise EnvironmentError("Error condition evaluation does not return boolean !!!")

        if retVal and self.ifBlock is not None:
            return self.ifBlock.evaluate(arguments)
        elif not retVal and self.elseBlock is not None:
            return self.elseBlock.evaluate(arguments)

        return arguments
    
    def add(self, block):
        if type(block) != StatementsBlock:
            raise EnvironmentError()
        super().add(block)
        self.elseBlock = block

    def encode_str(self):
        codage = super().encode_str()
        codage += self.ifBlock.encode_str()
        codage += self.elseBlock.encode_str() if self.elseBlock is not None else ''

        return codage

    def setLeft(self, statement):
        self.ifBlock = statement
    def setRight(self, statement):
        self.elseBlock = statement

    def getLeft(self):
        return self.ifBlock
    def getRight(self):
        return self.elseBlock

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
        args = arguments
        if self.statement is not None:
            args = self.statement.evaluate(args)

        if self.statementsBlock is not None:
            args = self.statementsBlock.evaluate(args)
        return args

    def add(self, statement_s):
        if self.statement is None:
            super().add(statement_s)
            self.statement = statement_s
        else:
            block = StatementsBlock()
            block.add(statement_s)
            super().add(block)
            if self.statementsBlock is not None:
                block.statementsBlock = self.statementsBlock
                self.statementsBlock.parent = block
            self.statementsBlock = block     

    def encode_str(self):
        codage = '1'
        codage += self.statement.encode_str() if self.statement is not None else ''
        codage += self.statementsBlock.encode_str() if self.statementsBlock is not None else ''
        codage += '0'
        return codage

    def setLeft(self, statement):
        self.statement = statement
    def setRight(self, statement):
        self.statementsBlock = statement

    def getLeft(self):
        return self.statement
    def getRight(self):
        return self.statementsBlock