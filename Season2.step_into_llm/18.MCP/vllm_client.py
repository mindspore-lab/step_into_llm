import asyncio
import json
from openai import OpenAI
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# 配置 vLLM 的 OpenAI 兼容 API
OPENAI_API_KEY = "EMPTY"  # vLLM 不需要真实密钥
OPENAI_API_BASE = "http://localhost:8000/v1"  # vLLM 的 OpenAI 兼容接口地址

class MCPClientDemo:
    def __init__(self, server_path: str):
        self.server_path = server_path
        # ✅ 使用 vLLM 的 OpenAI 兼容接口
        self.llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    async def run(self, user_query: str):
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # 获取服务端注册的工具
                tools = (await session.list_tools()).tools

                # 转换为 OpenAI 格式的 tools
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {
                                "type": "object",
                                "properties": {
                                    "city_name": {"type": "string", "description": "城市名称"}
                                },
                                "required": ["city_name"]
                            }
                        }
                    })

                # -------------------------------
                # 方法 1: vLLM + MCP 工具调用
                # -------------------------------
                # 第一步：让模型判断是否需要调用工具
                system_message = f"""
                你是一个智能助手。你可以使用以下工具：
                {json.dumps(openai_tools, ensure_ascii=False, indent=2)}

                如果用户的问题需要调用工具，请回复 "TOOL_CALL: <工具名> <JSON参数>"。
                如果不需要调用工具，请直接回答。
                """

                messages_for_tool_decision = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}
                ]
                
                # ✅ 使用 vLLM（通过 OpenAI 客户端）获取决策
                try:
                    decision_response = self.llm.chat.completions.create(
                        model="/home/ma-user/work/Qwen2.5-1.5B-Instruct/",  # 模型名需与启动 vLLM 时一致
                        messages=messages_for_tool_decision,
                        max_tokens=512,
                    )
                    decision_text = decision_response.choices[0].message.content.strip()
                except Exception as e:
                    decision_text = ""
                    print(f"调用 vLLM 失败: {e}")

                result_with_tool = {"model_reply": "", "tool_called": None, "tool_result": None}

                # 检查是否需要调用工具
                if decision_text.startswith("TOOL_CALL:"):
                    try:
                        # 解析工具调用指令
                        _, tool_name, args_json_str = decision_text.split(" ", 2)
                        arguments = json.loads(args_json_str)

                        # ✅ 通过 MCP 会话调用实际工具
                        tool_result = await session.call_tool(tool_name, arguments)

                        # 第二步：将工具结果返回给模型，生成最终回复
                        messages_with_result = messages_for_tool_decision + [
                            {"role": "assistant", "content": decision_text},
                            {"role": "tool", "content": json.dumps(tool_result.model_dump(), ensure_ascii=False),
                             "name": tool_name},
                            {"role": "user", "content": "请根据以上工具调用结果，回答用户的问题。"}
                        ]

                        final_response = self.llm.chat.completions.create(
                            model="/home/ma-user/work/Qwen2.5-1.5B-Instruct/",
                            messages=messages_with_result,
                            max_tokens=512,
                        )
                        result_with_tool["model_reply"] = final_response.choices[0].message.content
                        result_with_tool["tool_called"] = tool_name
                        result_with_tool["tool_arguments"] = arguments
                        result_with_tool["tool_result"] = tool_result
                    except Exception as e:
                        result_with_tool["model_reply"] = f"工具调用解析错误: {e}。原始回复: {decision_text}"
                else:
                    result_with_tool["model_reply"] = decision_text

                # -------------------------------
                # 方法 2: 仅模型回复（无工具）
                # -------------------------------
                try:
                    response_no_tool = self.llm.chat.completions.create(
                        model="/home/ma-user/work/Qwen2.5-1.5B-Instruct/",
                        messages=[{"role": "user", "content": user_query}],
                        max_tokens=512,
                    )
                    message_no_tool = response_no_tool.choices[0].message
                    result_no_tool = {
                        "model_reply": message_no_tool.content
                    }
                except Exception as e:
                    result_no_tool = {
                        "model_reply": f"调用 vLLM 失败（无工具）: {e}"
                    }

                return {
                    "user_query": user_query,
                    "with_mcp_tool": result_with_tool,
                    "without_tool": result_no_tool
                }


async def main():
    client = MCPClientDemo(server_path="/home/ma-user/work/mcp/stdio_mcp.py")
    result = await client.run("nanjing的天气怎么样")  

    print(">>> 用户提问：", result["user_query"])
    print("\n【使用 MCP 工具】")
    print("模型回复：", result["with_mcp_tool"]["model_reply"])
    if result["with_mcp_tool"]["tool_called"]:
        print("调用工具：", result["with_mcp_tool"]["tool_called"])
        print("工具参数：", result["with_mcp_tool"]["tool_arguments"])
        print("工具结果：", result["with_mcp_tool"]["tool_result"])
    else:
        print("未调用任何工具")

    print("\n【不使用工具】")
    print("模型回复：", result["without_tool"]["model_reply"])


if __name__ == "__main__":
    asyncio.run(main())
