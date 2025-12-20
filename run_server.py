# run_server.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        # These patterns are handled by uvicorn itself, not the shell
        reload_excludes=[
            "tests",
            "tests/*",
            "__pycache__",
            "__pycache__/*",
        ],
    )
