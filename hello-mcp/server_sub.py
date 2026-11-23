import os
import math
import time
import asyncio
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Initialize the FastMCP Server
mcp = FastMCP("math-sub-server")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@mcp.tool()
async def sub_numbers(num1: float, num2: float) -> str:
    """Strictly subtracts num2 from num1. DO NOT USE FOR ANYTHING ELSE. Example input: num1=10, num2=5."""
    # Enforce 15-digit limit and inf/nan
    if math.isinf(num1) or math.isnan(num1) or math.isinf(num2) or math.isnan(num2):
        return "ERROR: Invalid number (inf/nan)."
    if abs(num1) > 999999999999999 or abs(num2) > 999999999999999:
        return "ERROR: Number exceeds 15 digit limit."
        
    try:
        # Fast, deterministic, zero-cost exact calculation
        result = num1 - num2
        if math.isclose(result, round(result), abs_tol=1e-9):
            return str(int(round(result)))
        return str(round(result, 6))
    except Exception:
        pass

    try:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You are a strict calculator. Return only the exact numerical result of subtracting num2 from num1. Round any decimal results to a maximum of 6 decimal places. If inputs are non-mathematical, return 'I'm a calculator Bot, please ask mathematical questions'. Never return fractions, only decimals."},
                        {"role": "user", "content": f"Subtract {num2} from {num1}"}
                    ]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == 2:
                    return f"ERROR: The API encountered an error after 3 retries. ({str(e)})"
                await asyncio.sleep(1)
    except Exception as e:
        return "ERROR: API encountered an error."

if __name__ == "__main__":
    mcp.run()
