import os
import queue
import threading  # Import threading module

import gradio as gr
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate

os.environ["DASHSCOPE_API_KEY"] = [you api key]

llm = Tongyi(model="qwen-plus") 

template = ''' 
你是一个业务分析师，而我是领域专家.

尽你所能地回答下面的问题，你可以使用如下的工具:

{tools}

这里是上下文:

===CONTEXT
{context}
===END OF CONTEXT 

这里是用户故事:

===USER STORY
{story}
===END OF USER STORY

使用 场景 解释用户故事，并遵循如下格式:

Thought：你应该考虑用户故事中不清晰的部分。但忽略技术细节.
Action：要采取的动作，应该是以下工具之一 [{tool_names}] (问我需要澄清的问题)
Action Input：动作的输入内容。（这个应该是用来澄清用户故事而提出的问题）
Observation：动作的结果. (我给出答案)
... （这个 Thought/Action/Action Input/Observation 过程重复至少 3 次而不多于 10 次）

当你已经了解用户故事，不需要使用工具或者已经有了响应，你必须使用一下的格式回答：

思考: 我已经对这个用户故事了解了足够多的内容.
最终回答: [场景：尽可能列出所有场景。使用 Given/When/Then 的格式表述.]


开始!

之前的沟通历史:
{chat_history}

新的输入: {input}

思考: {agent_scratchpad}
'''

#When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
# This queue will be used to pass the answer back to the ask_expert function
answers_queue = queue.Queue()
questions_queue = queue.Queue()
starting_llm = True


@tool("Ask The Domain Expert")
def ask_expert(question: str) -> str:
    """当你需要提出问题来澄清用户故事时非常有用."""
    print(question)
    questions_queue.put(question, block=False)
    answer = answers_queue.get()  # Wait for an answer to be put into the queue
    return answer


tools = [ask_expert]

prompt = PromptTemplate.from_template(template)

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create Gradio interface
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Clarification (Type 'hi' to start)",
                             placeholder="Give Clarification...")
        with gr.Column():
            sty = ('作为学校的教职员工（As a faculty），\n'
                   '我希望学生可以根据录取通知将学籍注册到教学计划上'
                   '（I want the student to be able to enroll in an academic program with given offer），\n'
                   '从而我可以跟踪他们的获取学位的进度（So that I can track their progress）')
            user_story = gr.Textbox(label="用户故事 (User Story)", placeholder="Write user story here...", lines=4,
                                    value=sty)
            ctx = ('整个学籍管理系统是一个 Web 应用；\n\n'
                   '当教职员工发放录取通知时，会同步建立学生的账号；\n'
                   '学生可以根据身份信息，查询自己的账号；\n'
                   '在报道注册时，学生登录账号，按照录取通知书完成学年的注册；')
            biz_context = gr.Code(label="业务上下文 (Business Context)",
                                  value=ctx)


    def start_llm(context, story):
        resp = agent_executor.invoke(
            {"context": context, "story": story, "input": "", "chat_history": "", })
        print("LLM RESP:\n", resp)
        # 这里是最终的回答，即验收条件
        questions_queue.put(resp["output"], block=False)


    def respond(message, chat_history, context, story):
        global starting_llm
        if starting_llm:
            # Start start_llm in a separate thread to avoid blocking
            threading.Thread(target=(lambda: start_llm(context, story))).start()
            bot_message = questions_queue.get()
            chat_history.append((message, bot_message))
            starting_llm = False
            return "", chat_history
        answers_queue.put(message)
        bot_message = questions_queue.get()
        chat_history.append((message, bot_message))
        return "", chat_history


    msg.submit(respond, [msg, chatbot, biz_context, user_story], [msg, chatbot])

app.launch()
