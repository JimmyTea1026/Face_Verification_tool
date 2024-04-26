import json
import os
import subprocess

def main():
    # 創建子進程
    process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.PIPE)
    stdout_line = process.stdout.readline()
    if stdout_line:
        print(f"輸出: {stdout_line.strip()}")


    # 主線程繼續執行其他任務
    test_images = os.listdir('./test')
    for i, img in enumerate(test_images):
        img_path = os.path.join('./test', img)
        input_data = {"id": i, "img_path": img_path}
        json_input = json.dumps(input_data)
        process.stdin.write(json_input + '\n')
        process.stdin.flush()

        # 讀取並處理 stdout
        stdout_line = process.stdout.readline()
        if stdout_line:
            print(f"輸出: {stdout_line.strip()}")

    # 發送結束信號到子進程
    process.stdin.write('exit\n')
    process.stdin.flush()
    # 等待子進程結束
    process.wait()

if __name__ == "__main__":
    main()