import threading
import time
import webbrowser

import uvicorn


def open_browser() -> None:
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")


def main() -> None:
    print("Open http://localhost:8000")
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
