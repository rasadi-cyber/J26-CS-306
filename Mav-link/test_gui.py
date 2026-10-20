import sys
import traceback
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QEventLoop, QTimer
from ui.main_window import MainWindow

def test():
    try:
        app = QApplication(sys.argv)
        win = MainWindow()
        print("MainWindow created.")
        win._on_file_selected("sample_uav_flight.pcap")
        print("File selected.")
        win._execute_parsing()
        print("Parsing executed.")

        loop = QEventLoop()
        
        def on_done(df):
            print("on_done called with df len:", len(df))
            loop.quit()

        def on_fail(err):
            print("on_fail called with error:", err)
            loop.quit()

        win.worker.parsing_completed.connect(on_done)
        win.worker.parsing_failed.connect(on_fail)

        # 5 second safety timer
        QTimer.singleShot(10000, loop.quit)
        loop.exec_()

        print("Final DataFrame shape in Window:", win.current_df.shape)
        assert not win.current_df.empty, "DataFrame should not be empty!"
        print("GUI Integration Test Passed Successfully!")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test()
