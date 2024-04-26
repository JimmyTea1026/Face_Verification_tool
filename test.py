import os
import subprocess
import threading

def monitor_stderr(process, flag):
    """監控錯誤輸出的線程函數"""
    for line in iter(process.stderr.readline, ''):
        if line:
            print(f"{line.strip()}")
            flag.set()  # 設置標誌，表示發生了錯誤

def main():
    # 創建子進程
    process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.PIPE)
    
    # 錯誤標誌
    error_flag = threading.Event()

    # 創建並啟動監控 stderr 的線程
    stderr_thread = threading.Thread(target=monitor_stderr, args=(process, error_flag))
    stderr_thread.daemon = True
    stderr_thread.start()

    # 主線程繼續執行其他任務
    test_images = os.listdir('./test')
    for img in test_images:
        img_path = os.path.join('./test', img)
        process.stdin.write(img_path + '\n')
        process.stdin.flush()

        # 檢查是否有錯誤發生
        # if error_flag.is_set():
        #     print("檢測到錯誤")

        # 讀取並處理 stdout
        stdout_line = process.stdout.readline()
        if stdout_line:
            print(f"輸出: {stdout_line.strip()}")

    # 發送結束信號到子進程
    process.stdin.write('exit\n')
    process.stdin.flush()

    # 等待子進程結束
    process.wait()

    # 等待 stderr 線程結束
    stderr_thread.join()

if __name__ == "__main__":
    main()