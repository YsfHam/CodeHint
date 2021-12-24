import statement

algo = statement.StatementsBlock()
condition = statement.ConditionStatement("est_vide", lambda x: x > 0, True)
algo.add(condition)

condition.add(statement.FunctionStatement("retirer", lambda x: x - 1))
condition.add(statement.FunctionStatement("poser", lambda x: x + 1), False)

print(algo)

print(algo.evaluate(0))