from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI


def sql_module(user_input: str, db: SQLDatabase, sql_llm: ChatOpenAI, ba_llm: ChatOpenAI) -> tuple[str, str]:

    sql_system_prompt = """Double check the user's {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query.
If there are no mistakes, just reproduce the original query with no further commentary.
Don't use LIMIT if it's unnecessary

Output the final SQL query only without markdown syntax."""
    chain = create_sql_query_chain(sql_llm, db)

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", sql_system_prompt),
            #    MessagesPlaceholder("chat_history"),
            ("human", "{query}"),
        ]
    ).partial(dialect=db.dialect)
    validation_chain = sql_prompt | sql_llm | StrOutputParser()

    full_chain = {"query": chain} | validation_chain

    query = full_chain.invoke({"question": user_input})
    ba_system_prompt = """You are product analyst. Give your business analysis or insigths based on manager question and DB query result:

Manager question: {question}
DB query result: {query}
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ba_system_prompt),
            # MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    category_chain = prompt | ba_llm | StrOutputParser()
    result = category_chain.invoke({"question": user_input, "query": db.run(query)})
    return query, result


def ml_train_module(file_path: str, caption: str, llm: ChatOpenAI):
    if caption is None:
        caption = ""
    tools = [PythonREPLTool()]
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    instructions = """Вы — агент, предназначенный для написания и выполнения кода Python для ответа на вопросы по файлу. 
Не переспрашивай, а сразу готовь ответ.
У вас есть доступ к REPL Python, который вы можете использовать для выполнения кода Python.
Если вы получили ошибку, отладьте код и повторите попытку.
Вам дается файл формата csv.

Вам необходимо построить модель машинного обучения для предсказания целевой переменной.
Подбери гиперпараметры по валидационной выборке при помощи библиотеки optuna. Для optuna отключи логгирование при помощи optuna.logging.set_verbosity(optuna.logging.WARNING).
Также необходимо посчитать метрики качества на валидационной выборке с гиперпараметрами, подобраннными optuna.

Используйте только данные из отправленного тебе файла и выходные данные вашего кода для ответа на вопрос. 
Вы можете знать ответ, не запуская никакого кода, но вам все равно придется запустить код, чтобы получить ответ.
Если кажется, что вы не можете написать код для ответа на вопрос, просто верните «Я не знаю» в качестве ответа.
Весь сгенерированный код и выходы оберни в markdown для удобства копирования.
"""
    prompt = base_prompt.partial(instructions=instructions)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    executor = agent_executor.stream({"input": f"Название файла {file_path}" + caption})

    for chunk in executor:
        # Agent Action
        if "actions" in chunk:
            for action in chunk["actions"]:
                yield "action", f"{action.tool_input}"

        elif "steps" in chunk:
            for step in chunk["steps"]:
                yield "step", f"{step.observation}"
        # Final result
        elif "output" in chunk:
            yield "output", f'{chunk["output"]}'


def eda_module(file_path: str, caption: str, llm: ChatOpenAI):
    if caption is None:
        caption = ""
    tools = [PythonREPLTool()]
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    instructions = """Вы — агент, предназначенный для написания и выполнения кода Python для ответа на вопросы по файлу. 
Не переспрашивай, а сразу готовь ответь.
У вас есть доступ к REPL Python, который вы можете использовать для выполнения кода Python.
Если вы получили ошибку, отладьте код и повторите попытку.
Вам дается файл формата csv.

Вам необходимо провести разведочный анализ данных (EDA) для следующего набора данных. 
Если появляется KeyError проверь все названия столбцов и используй только их
Напечатай статистики, которые описывали бы данные.
Сделай выводы о данных по статистикам 
Поищи зависимости в данных
Все статистики и выводы предоставь в ответе.

Используйте только данные из отправленного тебе файла и выходные данные вашего кода для ответа на вопрос. 
Вы можете знать ответ, не запуская никакого кода, но вам все равно придется запустить код, чтобы получить ответ.
Если кажется, что вы не можете написать код для ответа на вопрос, просто верните «Я не знаю» в качестве ответа.
Весь сгенерированный код и выходы оберни в markdown для удобства копирования.
"""
    prompt = base_prompt.partial(instructions=instructions)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    executor = agent_executor.stream({"input": f"Название файла {file_path}" + caption})

    for chunk in executor:
        # Agent Action
        if "actions" in chunk:
            for action in chunk["actions"]:
                yield "action", f"```python{action.tool_input}"

        elif "steps" in chunk:
            for step in chunk["steps"]:
                yield "step", f"```python{step.observation}"
        # Final result
        elif "output" in chunk:
            yield "output", f'{chunk["output"]}'
