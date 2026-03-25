# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('src', 'src'), ('data/assets', 'data/assets'), ('logo.png', '.'), ('logo.ico', '.')]
binaries = [('D:\\Anaconda\\envs\\agent\\Library\\bin\\libexpat.dll', '.')]
hiddenimports = ['pyexpat']
tmp_ret = collect_all('langgraph')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['gui_main.py'],
    pathex=['D:\\Anaconda\\envs\\agent\\Library\\bin'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PaperAnvil_ResearchAssist',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['logo.ico'],
    hide_console='hide-early',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PaperAnvil_ResearchAssist',
)
