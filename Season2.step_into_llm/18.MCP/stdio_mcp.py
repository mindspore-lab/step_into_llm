#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import pdb
import logging
import json
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

proxies = {
    "http://": httpx.Proxy("http://proxy-notebook.modelarts.com:8083"),
    "https://": httpx.Proxy("http://proxy-notebook.modelarts.com:8083"),
}

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")
 
# OpenWeather API 配置
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = " "# "请替换为你自己的 OpenWeather API Key"
USER_AGENT = "weather-app/1.0"
 
async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}
    print(OPENWEATHER_API_BASE, params, headers)
    async with httpx.AsyncClient(proxies=proxies,timeout=30.0) as client:
        try:
            response = await client.get(OPENWEATHER_API_BASE, params=params, headers=headers)
            response.raise_for_status()
            return response.json()  # 返回字典类型
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误: {e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}
 
def format_weather(data: dict[str, Any] | str) -> str:
    """
    将天气数据格式化为易读文本。
    :param data: 天气数据（可以是字典或 JSON 字符串）
    :return: 格式化后的天气信息字符串
    """
    # 如果传入的是字符串，则先转换为字典
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"无法解析天气数据: {e}"
 
    # 如果数据中包含错误信息，直接返回错误提示
    if "error" in data:
        return f"⚠️ {data['error']}"
 
    # 提取数据时做容错处理
    city = data.get("name", "未知")
    country = data.get("sys", {}).get("country", "未知")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    # weather 可能为空列表，因此用 [0] 前先提供默认字典
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "未知")
 
    return (
        f"🌍 {city}, {country}\n"
        f"🌡 温度: {temp}°C\n"
        f"💧 湿度: {humidity}%\n"
        f"🌬 风速: {wind_speed} m/s\n"
        f"🌤 天气: {description}\n"
    )
 
@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果。
    :param city: 城市名称（需使用英文）
    :return: 格式化后的天气信息
    """
    data = await fetch_weather(city)
    print(data)
    return format_weather(data)


async def manual_get_weather():
    await query_weather("beijing")

import asyncio
if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    mcp.run(transport='stdio')









