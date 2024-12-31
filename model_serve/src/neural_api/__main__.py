import uvicorn
from neural_api.gateway import create_app


def main() -> create_app:
    """Main entry point for the application"""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
