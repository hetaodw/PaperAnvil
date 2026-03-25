import os
import subprocess
import sys
import shutil

def build():
    print("--- 正在准备极致稳定版打包 PaperAnvil GUI ---")
    
    # 1. 获取当前正在执行的 Python 信息
    python_exe = sys.executable
    python_dir = os.path.dirname(python_exe)
    print(f"当前运行 Python: {python_exe}")
    print(f"Python 版本: {sys.version}")

    # 2. 定位 DLL 路径 (针对 Anaconda)
    # 优先查找环境下的 Library/bin
    library_bin = os.path.join(python_dir, "Library", "bin")
    if not os.path.exists(library_bin):
        # 兜底查找上级目录 (如果 python.exe 在 envs/xxx/ 下而非其子目录)
        library_bin = os.path.join(os.path.dirname(python_dir), "Library", "bin")

    print(f"预期 Library/bin: {library_bin}")

    # 3. 准备打包命令 (强制使用 onedir 提高成功率，避免中文路径解压失败)
    # 0. 尝试关闭正在运行的程序实例 (防止 PermissionError: [WinError 5])
    print("\n检查并清理运行中的程序实例...")
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/IM", "PaperAnvil_ResearchAssist.exe", "/T"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(1) # 等待操作系统释放文件句柄
            print("  已清理旧进程。")
        except:
            pass

    cmd = [
        python_exe, "-m", "PyInstaller",
        "--noconsole",
        "--hide-console", "hide-early",
        "--onedir", 
        "--noconfirm", # 自动覆盖
        "--name", "PaperAnvil_ResearchAssist",
        "--icon", "logo.ico",
        "--add-data", "src;src",
        "--add-data", "data/assets;data/assets",
        "--add-data", "logo.png;.",
        "--add-data", "logo.ico;.",
        "--clean",
        "--noconfirm",
        "--collect-all", "langgraph",
        "--hidden-import", "pyexpat",
    ]

    # 4. 手动补全核心 DLL (libexpat.dll)
    expat_dll = os.path.join(library_bin, "libexpat.dll")
    if os.path.exists(expat_dll):
        print(f"✅ 找到 libexpat.dll: {expat_dll}")
        cmd.extend(["--add-binary", f"{expat_dll};."])
    else:
        print("⚠️ 未能在常规位置找到 libexpat.dll，尝试系统映射搜索...")
        # 搜索更广的范围
        for root, dirs, files in os.walk(python_dir):
            if "libexpat.dll" in files:
                p = os.path.join(root, "libexpat.dll")
                print(f"✅ 在深层目录找到: {p}")
                cmd.extend(["--add-binary", f"{p};."])
                break

    # 5. 增加库搜索路径
    if os.path.exists(library_bin):
        cmd.extend(["--paths", library_bin])

    cmd.append("gui_main.py")

    # 6. 执行清理并打包
    print("\n🚀 启动打包命令...")
    try:
        if os.path.exists("build"): shutil.rmtree("build", ignore_errors=True)
        if os.path.exists("dist"): shutil.rmtree("dist", ignore_errors=True)
        
        subprocess.run(cmd, check=True)
        
        print("\n" + "="*50)
        print("✨✨ 打包流程执行完毕！ ✨✨")
        print(f"由于您的路径包含中文/特殊字符，强烈建议运行文件夹版本。")
        print(f"主程序位于: dist/PaperAnvil_ResearchAssist/PaperAnvil_ResearchAssist.exe")
        print(f"注意：分发时必须带上整个 PaperAnvil_ResearchAssist 文件夹。")
        print("="*50)
    except Exception as e:
        print(f"\n❌ 打包过程出错: {e}")

if __name__ == "__main__":
    build()
