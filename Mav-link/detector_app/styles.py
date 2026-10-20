"""Cyber-dark theme for the standalone detector."""

CYBER_DARK_THEME = """
QWidget { background-color: #0b0f19; color: #e2e8f0; font-family: "Segoe UI", "Inter", sans-serif; font-size: 12px; }
QLabel { background-color: transparent; }
QFrame#HeaderFrame { background-color: #0f172a; border-bottom: 1px solid #1e293b; }
QLabel#AppTitle { color: #f8fafc; font-size: 20px; font-weight: 700; }
QLabel#AppSubtitle { color: #94a3b8; font-size: 11px; font-weight: 600; }
QFrame#SidebarFrame { background-color: #0a0f1d; border-right: 1px solid #1e293b; }
QFrame.card { background-color: #0d1322; border: 1px solid #1e293b; border-radius: 8px; }
QLabel#CardTitle { color: #38bdf8; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QPushButton { background-color: #1e293b; color: #f1f5f9; border: 1px solid #334155; border-radius: 6px; padding: 7px 10px; font-weight: 600; }
QPushButton:hover { border-color: #38bdf8; color: #38bdf8; }
QPushButton:disabled { color: #64748b; }
QPushButton#PrimaryButton { background-color: #047857; color: #ffffff; border: 1px solid #10b981; font-size: 14px; font-weight: 800; padding: 10px; }
QPushButton#PrimaryButton:hover { background-color: #059669; }
QPushButton#SecondaryButton { color: #38bdf8; }
QLineEdit, QComboBox { background-color: #0f172a; border: 1px solid #334155; border-radius: 5px; padding: 6px; color: #e2e8f0; }
QCheckBox { background: transparent; }
QTabWidget::pane { background-color: #0d1322; border: 1px solid #1e293b; }
QTabBar::tab { background: transparent; color: #94a3b8; padding: 7px 11px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #38bdf8; border-bottom-color: #38bdf8; }
QTableWidget { background-color: #0d1322; alternate-background-color: #10192e; color: #e2e8f0; border: 1px solid #1e293b; gridline-color: #162036; }
QTableWidget::item { padding: 5px; }
QHeaderView::section { background-color: #10192e; color: #94a3b8; padding: 7px 5px; font-weight: 700; border: none; border-bottom: 1px solid #1e293b; }
QProgressBar { background: #0f172a; border: 1px solid #1e293b; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background: #10b981; border-radius: 5px; }
QFrame#ConsoleFrame { background: #080c14; border: 1px solid #1e293b; border-radius: 7px; }
QPlainTextEdit#ConsoleLog { background-color: #06090e; color: #38bdf8; border: 1px solid #162036; font-family: Consolas, monospace; font-size: 10px; }
"""
