"""Console entry point: run the proxy with uvicorn on port 4001."""

import uvicorn


def main() -> None:
    from gemma4_delimiter_proxy.proxy import app

    uvicorn.run(app, host="0.0.0.0", port=4001)


if __name__ == "__main__":
    main()
