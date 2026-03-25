import sys
import os
import json
import time
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                               QTextEdit, QProgressBar, QSpinBox, QFrame, QMessageBox,
                               QTabWidget, QScrollArea, QGraphicsOpacityEffect, QRadioButton, 
                               QButtonGroup, QFileDialog, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, QRect, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QFont, QIcon, QPixmap, QColor, QPalette

# 导入我们的核心工作流逻辑和默认提示词
from app import run_workflow, DEFAULT_PROMPTS


class AnimatedButton(QPushButton):
    """带动画效果的按钮"""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._opacity = 1.0
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        
    def get_opacity(self):
        return self._opacity
    
    def set_opacity(self, value):
        self._opacity = value
        self.opacity_effect.setOpacity(value)
    
    opacity = Property(float, get_opacity, set_opacity)
    
    def animate_click(self):
        """点击动画：缩放 + 透明度变化"""
        self.animate_scale(0.95, 100)
        
    def animate_scale(self, scale, duration=150):
        """缩放动画"""
        anim = QPropertyAnimation(self, b"geometry")
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        
        rect = self.geometry()
        center = rect.center()
        new_width = int(rect.width() * scale)
        new_height = int(rect.height() * scale)
        new_rect = QRect(center.x() - new_width // 2, center.y() - new_height // 2, new_width, new_height)
        
        anim.setStartValue(rect)
        anim.setKeyValueAt(0.5, new_rect)
        anim.setEndValue(rect)
        anim.start()
        
        self._anim = anim  # 保持引用防止被回收
        
    def enterEvent(self, event):
        """鼠标进入动画"""
        super().enterEvent(event)
        anim = QPropertyAnimation(self, b"opacity")
        anim.setDuration(100)
        anim.setStartValue(self._opacity)
        anim.setEndValue(0.9)
        anim.start()
        self._opacity_anim = anim
        
    def leaveEvent(self, event):
        """鼠标离开动画"""
        super().leaveEvent(event)
        anim = QPropertyAnimation(self, b"opacity")
        anim.setDuration(100)
        anim.setStartValue(self._opacity)
        anim.setEndValue(1.0)
        anim.start()
        self._opacity_anim = anim

def resource_path(relative_path):
    """ 获取资源的绝对路径，兼容 PyInstaller 打包后的路径 """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

CONFIG_FILE = "config.json"

class WorkflowThread(QThread):
    """
    后台线程，用于运行 LangGraph 工作流而不会导致 GUI 冻结。
    """
    log_signal = Signal(str)
    finished_signal = Signal(dict)
    
    def __init__(self, topic, persona_count, api_key, custom_prompts, input_text, resume_checkpoint=False, use_existing_csv=False, existing_csv_path=""):
        super().__init__()
        self.topic = topic
        self.persona_count = persona_count
        self.api_key = api_key
        self.custom_prompts = custom_prompts
        self.input_text = input_text
        self.resume_checkpoint = resume_checkpoint
        self.use_existing_csv = use_existing_csv
        self.existing_csv_path = existing_csv_path
        self._is_running = True
        
    def run(self):
        # 设置环境变量
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        
        def callback(msg):
            if not self._is_running:
                return
            self.log_signal.emit(msg)
            
        try:
            final_state = run_workflow(
                topic=self.topic, 
                persona_count=self.persona_count, 
                log_callback=callback,
                use_stdout_redirect=False,
                custom_prompts=self.custom_prompts,
                input_text=self.input_text,
                resume_persona_checkpoint=self.resume_checkpoint,
                use_existing_csv=self.use_existing_csv,
                existing_csv_path=self.existing_csv_path
            )
            if self._is_running:
                self.finished_signal.emit(final_state)
        except Exception as e:
            if self._is_running:
                self.log_signal.emit(f"\n❌ 线程运行发生致命错误: {str(e)}")
                self.finished_signal.emit({"current_step": "error"})

    def stop(self):
        self._is_running = False

class PaperAnvilGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PaperAnvil: AI Research Dashboard")
        self.setMinimumSize(900, 700)
        
        self.load_config()
        
        # 窗口样式 (深色模式)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Microsoft YaHei';
            }
            QLineEdit, QSpinBox {
                background-color: #2A2A2A;
                color: #FFFFFF;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3D5AFE;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #536DFE;
            }
            QPushButton:pressed {
                background-color: #304FFE;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QPushButton#StartBtn {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3D5AFE, stop:1 #536DFE);
            }
            QPushButton#StartBtn:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #536DFE, stop:1 #7C4DFF);
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: #A5D6A7;
                border: 1px solid #333;
                border-radius: 5px;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                padding: 10px;
            }
            QProgressBar {
                background-color: #2A2A2A;
                border: 1px solid #444;
                border-radius: 10px;
                text-align: center;
                color: white;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3D5AFE, stop:1 #00E5FF);
                border-radius: 10px;
            }
            QFrame#Sidebar {
                background-color: #1E1E1E;
                border-right: 1px solid #333;
            }
            QTabWidget::pane {
                border: 1px solid #333;
                background: #121212;
            }
            QTabBar::tab {
                background: #1E1E1E;
                color: #BBB;
                padding: 10px 20px;
                border: 1px solid #333;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background: #3D5AFE;
                color: white;
            }
            QTableWidget {
                background-color: #1E1E1E;
                color: #EEE;
                gridline-color: #333;
                border: none;
            }
            QHeaderView::section {
                background-color: #2A2A2A;
                color: #BBB;
                padding: 5px;
                border: 1px solid #333;
            }
        """)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # --- 侧边栏 ---
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)
        
        # Logo
        logo_label = QLabel()
        logo_img_path = resource_path("logo.png")
        if os.path.exists(logo_img_path):
            pixmap = QPixmap(logo_img_path).scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
        else:
            logo_label.setText("🛠️ PaperAnvil")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #448AFF;")
        sidebar_layout.addWidget(logo_label)
        
        sidebar_layout.addWidget(self._create_section_label("分析配置"))
        
        # API Key 输入
        sidebar_layout.addWidget(QLabel("DashScope API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("请输入您的 API Key (sk-...)")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        
        # 优先级：1. config.json, 2. Env
        current_key = self.config.get("api_key", os.getenv("DASHSCOPE_API_KEY", ""))
        self.api_key_input.setText(current_key)
        sidebar_layout.addWidget(self.api_key_input)

        # 主题输入
        sidebar_layout.addWidget(QLabel("调研主题:"))
        self.topic_input = QTextEdit()
        self.topic_input.setPlaceholderText("请输入调研报告的主题...")
        default_topic = self.config.get("topic", "")
        self.topic_input.setText(default_topic)
        self.topic_input.setFixedHeight(100)
        self.topic_input.setStyleSheet("QTextEdit { background-color: #2A2A2A; color: white; border: 1px solid #444; }")
        sidebar_layout.addWidget(self.topic_input)
        
        # 样本数量
        sidebar_layout.addWidget(QLabel("画像样本数量:"))
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 4999)
        self.count_input.setValue(self.config.get("persona_count", 21))
        sidebar_layout.addWidget(self.count_input)
        
        # 断点恢复选项
        self.resume_checkpoint = QCheckBox("从断点恢复")
        self.resume_checkpoint.setStyleSheet("color: #AAA; padding: 5px;")
        sidebar_layout.addWidget(self.resume_checkpoint)
        
        # 从已有 CSV 继续分析选项
        self.use_existing_csv = QCheckBox("使用已有数据文件")
        self.use_existing_csv.setStyleSheet("color: #AAA; padding: 5px;")
        self.use_existing_csv.toggled.connect(self.toggle_csv_input)
        sidebar_layout.addWidget(self.use_existing_csv)
        
        # CSV 文件选择
        self.csv_file_input = QLineEdit()
        self.csv_file_input.setPlaceholderText("选择 CSV 文件...")
        self.csv_file_input.setStyleSheet("QLineEdit { background-color: #2A2A2A; color: white; border: 1px solid #444; }")
        self.csv_file_input.hide()
        sidebar_layout.addWidget(self.csv_file_input)
        
        self.csv_browse_btn = QPushButton("浏览...")
        self.csv_browse_btn.setStyleSheet("QPushButton { background-color: #444; color: white; }")
        self.csv_browse_btn.clicked.connect(self.browse_csv_file)
        self.csv_browse_btn.hide()
        sidebar_layout.addWidget(self.csv_browse_btn)
        
        # 模式切换
        sidebar_layout.addWidget(QLabel("问卷生成方式:"))
        self.mode_group = QButtonGroup(self)
        self.radio_ai = QRadioButton("🤖 AI 全自动生成")
        self.radio_ai.setChecked(True)
        self.radio_text = QRadioButton("📋 转换已有问卷")
        self.radio_ai.setStyleSheet("color: white; padding: 5px;")
        self.radio_text.setStyleSheet("color: white; padding: 5px;")
        self.mode_group.addButton(self.radio_ai)
        self.mode_group.addButton(self.radio_text)
        
        sidebar_layout.addWidget(self.radio_ai)
        sidebar_layout.addWidget(self.radio_text)

        # 文本问卷输入框
        self.survey_text_input = QTextEdit()
        self.survey_text_input.setPlaceholderText("请在此粘贴您的问卷文本内容...")
        self.survey_text_input.setStyleSheet("QTextEdit { background-color: #2A2A2A; color: white; border: 1px solid #444; }")
        self.survey_text_input.hide()
        sidebar_layout.addWidget(self.survey_text_input)

        self.radio_text.toggled.connect(self.toggle_survey_text_input)
        
        sidebar_layout.addStretch()
        
        # 启动按钮
        self.start_btn = AnimatedButton("🚀 启动全自动分析")
        self.start_btn.setObjectName("StartBtn")
        self.start_btn.setFixedHeight(50)
        self.start_btn.clicked.connect(self.start_analysis)
        sidebar_layout.addWidget(self.start_btn)
        
        # 作者与开源信息 (放在启动按钮下方)
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: transparent; border: none; margin-top: 10px;")
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(2)
        info_layout.setContentsMargins(0, 10, 0, 0)
        
        author_label = QLabel("PaperAnvil Research Assistant")
        author_label.setStyleSheet("font-size: 11px; color: #666; font-weight: bold;")
        author_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(author_label)
        
        repo_label = QLabel("GitHub: hetaodw/PaperAnvil")
        repo_label.setStyleSheet("font-size: 9px; color: #555;")
        repo_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(repo_label)
        
        license_label = QLabel("MIT License | Author: hetaodw")
        license_label.setStyleSheet("font-size: 9px; color: #444;")
        license_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(license_label)
        
        sidebar_layout.addWidget(info_frame)
        
        # --- 主内容区 (选项卡) ---
        self.tabs = QTabWidget()
        
        # Tab 1: 运行看板
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_tab)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        dashboard_layout.setSpacing(15)
        
        dashboard_layout.addWidget(QLabel("🛠️ 运行控制面板"))
        self.status_label = QLabel("等待启动...")
        self.status_label.setStyleSheet("font-size: 16px; color: #BBB;")
        dashboard_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        dashboard_layout.addWidget(self.progress_bar)
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("系统运行日志将实时显示在这里...")
        dashboard_layout.addWidget(self.log_output)
        
        self.footer_layout = QHBoxLayout()
        self.open_report_btn = QPushButton("📄 查看 Markdown 报告")
        self.open_report_btn.setEnabled(False)
        self.open_report_btn.setStyleSheet("background-color: #4CAF50;")
        self.open_report_btn.clicked.connect(self.open_report)
        
        self.open_survey_btn = QPushButton("🌐 预览调查网页 (HTML)")
        self.open_survey_btn.setEnabled(False)
        self.open_survey_btn.setStyleSheet("background-color: #FF9800;")
        self.open_survey_btn.clicked.connect(self.open_survey)
        
        self.footer_layout.addWidget(self.open_report_btn)
        self.footer_layout.addWidget(self.open_survey_btn)
        
        self.open_xlsx_btn = QPushButton("📊 打开调查数据 (Excel)")
        self.open_xlsx_btn.setEnabled(False)
        self.open_xlsx_btn.setStyleSheet("background-color: #2E7D32;")
        self.open_xlsx_btn.clicked.connect(self.open_xlsx)
        self.footer_layout.addWidget(self.open_xlsx_btn)
        
        dashboard_layout.addLayout(self.footer_layout)
        
        # Tab 2: 画像预览
        self.persona_tab = QWidget()
        persona_layout = QVBoxLayout(self.persona_tab)
        self.persona_text = QTextEdit()
        self.persona_text.setReadOnly(True)
        self.persona_text.setPlaceholderText("运行完成后，此处将展示生成的虚拟画像详细信息...")
        persona_layout.addWidget(self.persona_text)
        
        # Tab 3: 数据分析结果
        self.analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(self.analysis_tab)
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText("运行完成后，此处将展示多维度统计与分析结论...")
        analysis_layout.addWidget(self.analysis_text)
        
        # Tab 4: 提示词配置 (原 Tab 2)
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.prompt_editors_layout = QVBoxLayout(scroll_content)
        self.prompt_editors_layout.setSpacing(20)
        
        self.prompt_widgets = {} # {key: QTextEdit}
        
        # 从 DEFAULT_PROMPTS 生成编辑器
        for key, value in DEFAULT_PROMPTS.items():
            section = QFrame()
            section.setStyleSheet("background-color: #1E1E1E; border-radius: 5px; border: 1px solid #333; padding: 10px;")
            sec_lay = QVBoxLayout(section)
            
            title = QLabel(f"Prompt: {key}")
            title.setStyleSheet("font-weight: bold; color: #448AFF; border: none;")
            sec_lay.addWidget(title)
            
            editor = QTextEdit()
            editor.setAcceptRichText(False)
            editor.setPlainText(value)
            editor.setStyleSheet("background-color: #121212; color: #CCC; border: 1px solid #444;")
            editor.setMinimumHeight(150)
            sec_lay.addWidget(editor)
            
            self.prompt_widgets[key] = editor
            self.prompt_editors_layout.addWidget(section)
            
        scroll.setWidget(scroll_content)
        prompt_layout.addWidget(scroll)
        
        # 按钮栏
        btn_bar = QHBoxLayout()
        reset_btn = QPushButton("恢复默认提示词")
        reset_btn.setStyleSheet("background-color: #757575;")
        reset_btn.clicked.connect(self.reset_prompts)
        btn_bar.addStretch()
        btn_bar.addWidget(reset_btn)
        prompt_layout.addLayout(btn_bar)
        
        self.tabs.addTab(dashboard_tab, "运行看板")
        self.tabs.addTab(self.persona_tab, "画像预览")
        self.tabs.addTab(self.analysis_tab, "数据分析")
        self.tabs.addTab(prompt_tab, "提示词配置")
        
        # Tab 5: 帮助说明
        help_tab = QWidget()
        help_layout = QVBoxLayout(help_tab)
        self.help_text = QTextEdit()
        self.help_text.setReadOnly(True)
        # 加载帮助文档
        help_md_path = resource_path("data/assets/help.md")
        if os.path.exists(help_md_path):
            with open(help_md_path, 'r', encoding='utf-8') as f:
                self.help_text.setMarkdown(f.read())
        else:
            self.help_text.setText("帮助文档尚未生成。")
        help_layout.addWidget(self.help_text)
        self.tabs.addTab(help_tab, "帮助说明")
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.tabs)
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def toggle_survey_text_input(self, checked):
        if checked:
            self.survey_text_input.show()
        else:
            self.survey_text_input.hide()
    
    def toggle_csv_input(self, checked):
        if checked:
            self.csv_file_input.show()
            self.csv_browse_btn.show()
        else:
            self.csv_file_input.hide()
            self.csv_browse_btn.hide()
    
    def browse_csv_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 CSV 数据文件",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.csv_file_input.setText(file_path)

    def _create_section_label(self, text):
        label = QLabel(text)
        label.setStyleSheet("font-weight: bold; color: #448AFF; border-bottom: 2px solid #448AFF; padding-bottom: 5px;")
        return label
        
    def start_analysis(self):
        # 触发按钮动画
        self.start_btn.animate_click()
        
        topic = self.topic_input.toPlainText().strip()
        count = self.count_input.value()
        api_key = self.api_key_input.text().strip()
        
        if not topic:
            self.log_output.append("⚠️ 请先输入调研主题！")
            return
            
        if not api_key:
            self.log_output.append("⚠️ 请先输入并保存 DashScope API Key！")
            return
            
        input_text = ""
        if self.radio_text.isChecked():
            input_text = self.survey_text_input.toPlainText().strip()
            if not input_text:
                self.log_output.append("⚠️ 请先粘贴问卷文本！")
                return

        # 实时更新环境变量，这样后续 Agent 都能拿到最新的
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        # 自动保存配置
        self.config["api_key"] = api_key
        self.config["topic"] = topic
        self.config["persona_count"] = count
        self.save_config()
        
        self.status_label.setText("🚀 正在运行...")
        # 收集自定义提示词
        custom_prompts = {}
        for key, widget in self.prompt_widgets.items():
            custom_prompts[key] = widget.toPlainText()
            
        self.log_output.clear()
        self.log_output.append(">>> 系统初始化中...")
        self.start_btn.setEnabled(False)
        self.open_report_btn.setEnabled(False)
        self.open_survey_btn.setEnabled(False)
        self.progress_bar.setValue(5)
        
        # 临时切换到运行看板
        self.tabs.setCurrentIndex(0)
        
        resume_checkpoint = self.resume_checkpoint.isChecked()
        use_existing_csv = self.use_existing_csv.isChecked()
        existing_csv_path = ""
        
        if use_existing_csv:
            existing_csv_path = self.csv_file_input.text().strip()
            if not existing_csv_path:
                self.log_output.append("⚠️ 请先选择 CSV 文件！")
                return
            if not os.path.exists(existing_csv_path):
                self.log_output.append("⚠️ CSV 文件不存在！")
                return
        
        self.workflow_thread = WorkflowThread(topic, count, api_key, custom_prompts, input_text, resume_checkpoint, use_existing_csv, existing_csv_path)
        self.workflow_thread.log_signal.connect(self.update_log)
        self.workflow_thread.finished_signal.connect(self.workflow_finished)
        self.workflow_thread.start()
        
    def update_log(self, text):
        if "DEBUG" in text:
            # 高亮 Debug 信息
            self.log_output.append(f"<span style='color: #76FF03;'>{text}</span>")
        else:
            self.log_output.append(text)
            
        # 简易进度条模拟
        if "Execution started" in text: self.progress_bar.setValue(10)
        elif ("survey_agent" in text) or ("Text-to-Survey" in text): self.progress_bar.setValue(20)
        elif "persona_agent" in text: self.progress_bar.setValue(35)
        elif "respondent_agent" in text: self.progress_bar.setValue(50)
        elif "analysis_agent" in text: self.progress_bar.setValue(70)
        elif "writer_agent" in text: self.progress_bar.setValue(90)
        
        # 滚动到底部
        self.log_output.ensureCursorVisible()
        
    def workflow_finished(self, final_state):
        self.start_btn.setEnabled(True)
        self.progress_bar.setValue(100)
        self.workflow_thread = None # 清理引用
        
        if final_state.get("current_step") == "error":
            self.status_label.setText("❌ 运行因错误中断")
            self.status_label.setStyleSheet("color: #FF5252;")
        else:
            self.status_label.setText("✅ 全流程生产完成！")
            self.status_label.setStyleSheet("color: #69F0AE; font-weight: bold;")
            self.open_report_btn.setEnabled(True)
            self.open_survey_btn.setEnabled(True)
            self.open_xlsx_btn.setEnabled(True)
            
            # 填充画像预览
            self.populate_persona_preview()
            # 填充数据分析
            self.populate_analysis_preview()
            
    def populate_persona_preview(self):
        try:
            path = "data/intermediate/personas.json"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.persona_text.setText(json.dumps(data, indent=2, ensure_ascii=False))
        except:
            pass

    def populate_analysis_preview(self):
        try:
            # 汇总多个分析文件的关键信息
            report = "### 📊 多维度分析报告摘要\n\n"
            
            paths = {
                "基础统计": "data/intermediate/basic_stats.json",
                "核心洞察": "data/intermediate/analysis_results.json",
                "语义分析": "data/intermediate/semantic_analysis.json"
            }
            
            for name, path in paths.items():
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        report += f"#### 【{name}】\n```json\n{json.dumps(content[:2] if isinstance(content, list) else content, indent=2, ensure_ascii=False)}\n```\n\n"
            
            self.analysis_text.setMarkdown(report)
        except:
            pass

    def open_report(self):
        report_path = os.path.abspath("data/output/thesis_draft.md")
        if os.path.exists(report_path):
            os.startfile(report_path)
            
    def open_survey(self):
        survey_path = os.path.abspath("data/output/survey.html")
        if os.path.exists(survey_path):
            os.startfile(survey_path)

    def open_xlsx(self):
        xlsx_path = os.path.abspath("data/output/research_data.xlsx")
        if os.path.exists(xlsx_path):
            os.startfile(xlsx_path)

    def reset_prompts(self):
        reply = QMessageBox.question(self, '重置', '确定要将所有提示词恢复为系统默认吗？',
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            for key, editor in self.prompt_widgets.items():
                editor.setPlainText(DEFAULT_PROMPTS.get(key, ""))

    def load_config(self):
        self.config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except:
                pass

    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except:
            pass

    def closeEvent(self, event):
        """处理窗口关闭时的线程安全"""
        if hasattr(self, 'workflow_thread') and self.workflow_thread and self.workflow_thread.isRunning():
            reply = QMessageBox.question(self, '退出', '工作流正在运行，确定要强行退出吗？',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.workflow_thread.stop()
                self.workflow_thread.terminate()
                self.workflow_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PaperAnvilGUI()
    window.show()
    sys.exit(app.exec())
