import io
import sys
import traceback
from typing import Dict, Any

def execute_python_code(code: str) -> Dict[str, Any]:
    """
    安全执行 Python 代码的沙箱工具。
    
    Args:
        code: 要执行的 Python 代码字符串
        
    Returns:
        包含执行结果的字典，格式为 {
            "output": str,  # 标准输出
            "error": str,   # 错误信息（如果有）
            "success": bool  # 是否成功执行
        }
        
    Raises:
        Exception: 代码执行过程中抛出的异常
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    
    try:
        sys.stdout = redirected_output
        sys.stderr = redirected_error
        
        exec_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
            }
        }
        
        exec(code, exec_globals)
        
        output = redirected_output.getvalue()
        error = redirected_error.getvalue()
        
        return {
            "output": output,
            "error": error,
            "success": True
        }
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return {
            "output": "",
            "error": error_msg,
            "success": False
        }
        
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
