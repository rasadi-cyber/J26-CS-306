"""
Cyber-Forensics Dark Theme Stylesheet (QSS) and Design Tokens
"""

CYBER_DARK_THEME = """
/* Global Application Styles */
QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: "Segoe UI", "Inter", "Roboto", -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
}

QLabel {
    background-color: transparent;
}

/* Tooltip */
QToolTip {
    background-color: #1e293b;
    color: #38bdf8;
    border: 1px solid #334155;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #0f172a;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #334155;
    min-height: 24px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #0ea5e9;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #0f172a;
    height: 10px;
    margin: 0px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #334155;
    min-width: 24px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #0ea5e9;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Header Bar */
#HeaderFrame {
    background-color: #0f172a;
    border-bottom: 1px solid #1e293b;
    padding: 8px 16px;
}

#AppTitle {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

#AppSubtitle {
    color: #64748b;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

#PageTabs {
    background-color: transparent;
}
QTabBar::tab {
    background-color: transparent;
    color: #94a3b8;
    border: 1px solid transparent;
    border-bottom: 2px solid transparent;
    border-radius: 5px 5px 0 0;
    padding: 7px 14px;
    margin-right: 4px;
    font-size: 12px;
    font-weight: 600;
}
QTabBar::tab:hover {
    background-color: #131d31;
    color: #e2e8f0;
}
QTabBar::tab:selected {
    background-color: #0d1322;
    color: #38bdf8;
    border: 1px solid #1e293b;
    border-bottom: 2px solid #38bdf8;
}

QTabWidget::pane {
    background-color: #0d1322;
    border: 1px solid #1e293b;
    border-radius: 6px;
    top: -1px;
}
QTabWidget#DetectorTabs QTabBar::tab {
    padding: 6px 11px;
    margin-right: 2px;
}

#StatusBadge {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    border-radius: 12px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

#SearchInput {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    min-width: 220px;
}
#SearchInput:focus {
    border: 1px solid #38bdf8;
    background-color: #0f172a;
}

/* Left Navigation Sidebar */
#SidebarFrame {
    background-color: #0a0f1d;
    border-right: 1px solid #1e293b;
    min-width: 260px;
    max-width: 320px;
}

#BrandBox {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e293b, stop:1 #0f172a);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
}

#BrandTitle {
    color: #10b981;
    font-weight: 700;
    font-size: 13px;
}

#BrandSubtitle {
    color: #64748b;
    font-size: 11px;
}

/* Cards & Containers */
QFrame.card {
    background-color: #131d31;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 12px;
}

QFrame.card:hover {
    border-color: #2e4369;
}

#StatCard {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #162238, stop:1 #0f182a);
    border: 1px solid #23334d;
    border-radius: 8px;
    padding: 10px 14px;
}

#StatCardTitle {
    color: #94a3b8;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

#StatCardValue {
    color: #f8fafc;
    font-size: 22px;
    font-weight: 700;
    margin-top: 4px;
}

/* Buttons */
QPushButton {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #273549;
    border-color: #0ea5e9;
    color: #38bdf8;
}
QPushButton:pressed {
    background-color: #0f172a;
}
QPushButton:disabled {
    background-color: #0f172a;
    border-color: #1e293b;
    color: #475569;
}

/* Primary Action Buttons */
QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #0284c7);
    color: #ffffff;
    border: 1px solid #10b981;
    font-weight: 700;
    font-size: 13px;
    border-radius: 6px;
    padding: 10px 16px;
}
QPushButton#PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #0ea5e9);
    border-color: #34d399;
}
QPushButton#PrimaryButton:pressed {
    background: #047857;
}

QPushButton#ExportButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #4f46e5);
    color: #ffffff;
    border: 1px solid #3b82f6;
    font-weight: 700;
    font-size: 13px;
    border-radius: 6px;
    padding: 10px 16px;
}
QPushButton#ExportButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #6366f1);
}

QPushButton#SecondaryButton {
    background-color: #1e293b;
    color: #38bdf8;
    border: 1px solid #0284c7;
    font-weight: 600;
    padding: 8px 14px;
}
QPushButton#SecondaryButton:hover {
    background-color: #0c4a6e;
}

QPushButton#ConsoleActionButton {
    min-width: 48px;
    padding: 2px 8px;
    font-size: 11px;
}

/* Nav Item Buttons */
QPushButton.nav-btn {
    background-color: transparent;
    border: none;
    border-radius: 6px;
    color: #94a3b8;
    text-align: left;
    padding: 8px 12px;
    font-size: 13px;
    font-weight: 500;
}
QPushButton.nav-btn:hover {
    background-color: #1e293b;
    color: #f1f5f9;
}
QPushButton.nav-btn.active {
    background-color: #064e3b;
    color: #34d399;
    border-left: 3px solid #10b981;
    font-weight: 700;
}

/* Radio & Checkboxes */
QRadioButton, QCheckBox {
    color: #cbd5e1;
    spacing: 8px;
    font-size: 12px;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #475569;
    border-radius: 3px;
    background-color: #0f172a;
}
QRadioButton::indicator {
    border-radius: 8px;
}
QRadioButton::indicator:checked {
    background-color: #10b981;
    border: 3px solid #0f172a;
}
QCheckBox::indicator:checked {
    background-color: #0ea5e9;
    border: 1px solid #38bdf8;
}

/* GroupBox */
QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    font-size: 12px;
    font-weight: 700;
    color: #94a3b8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #38bdf8;
}

/* Data Table */
QTableWidget {
    background-color: #0d1322;
    border: 1px solid #1e293b;
    border-radius: 8px;
    gridline-color: #162036;
    color: #e2e8f0;
    font-size: 13px;
}
QTableWidget::item {
    padding: 6px 8px;
    border-bottom: 1px solid #162036;
}
QTableWidget::item:selected {
    background-color: #1e3a8a;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #10192e;
    color: #94a3b8;
    padding: 9px 8px;
    font-weight: 700;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: none;
    border-bottom: 2px solid #1e293b;
    border-right: 1px solid #162036;
}

/* Progress Bar */
QProgressBar {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    height: 16px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #06b6d4);
    border-radius: 5px;
}

/* Console Log Pane */
QPlainTextEdit#ConsoleLog {
    background-color: #080c14;
    color: #38bdf8;
    font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
    font-size: 11px;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 8px;
}

/* ComboBox */
QComboBox {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
}
QComboBox:hover {
    border-color: #0ea5e9;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #f1f5f9;
    border: 1px solid #334155;
    selection-background-color: #0284c7;
}

/* Footer Frame */
#FooterFrame {
    background-color: #0a0f1d;
    border-top: 1px solid #1e293b;
    padding: 6px 16px;
}
#FooterText {
    color: #475569;
    font-size: 11px;
}
"""
